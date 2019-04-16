#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"


// Questo algoritmo è una modifica del Komura Equivalence (KE) che esegue le operazioni in due livelli (stage). 
// Inizialmente esegue le operazioni nel blocco usando la shared memory e poi merga le etichette sui bordi dei 
// blocchi. Varie prove hanno mostrato che sulla quadro va peggio della versione BUF.



#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;


// Algorithm itself has good performances, but memory allocation is a problem.
// I will try to reduce it.
namespace {

    // Only use it with unsigned numeric types
    template <typename T>
    __device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
        return (bitmap >> pos) & 1;
    }

    __device__ __forceinline__ void SetBit(unsigned char &bitmap, unsigned char pos) {
        bitmap |= (1 << pos);
    }


    // Risale alla radice dell'albero a partire da un suo nodo n
    __device__ unsigned Find(const int *s_buf, unsigned n) {
        // Attenzione: non invocare la find su un pixel di background
        while (s_buf[n] != n) {
            n = s_buf[n];
        }
        return n;
    }

    // Unisce gli alberi contenenti i nodi a e b, collegandone le radici
    __device__ void Union(int *s_buf, unsigned a, unsigned b) {

        bool done;

        do {

            a = Find(s_buf, a);
            b = Find(s_buf, b);

            if (a < b) {
                int old = atomicMin(s_buf + b, a);
                done = (old == b);
                b = old;
            }
            else if (b < a) {
                int old = atomicMin(s_buf + a, b);
                done = (old == a);
                a = old;
            }
            else {
                done = true;
            }

        } while (!done);

    }


    __global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
        unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
        unsigned img_index = row * img.step + col;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        __shared__ int buf[BLOCK_ROWS * BLOCK_COLS];
        unsigned buf_index = threadIdx.y * BLOCK_COLS + threadIdx.x;

        if (row < labels.rows && col < labels.cols) {
            buf[buf_index] = buf_index;
        }

        __syncthreads();

        if (row < labels.rows && col < labels.cols) {

            //	0|1 2|3
            // --+---+--
            //  4|A B|
            //  5|C D|
            // --+---+

            unsigned char P = 0;

            if ((threadIdx.x > 0 || threadIdx.y > 0)) {
                if (img[img_index]) {
                    P |= 0x37;				// 00110111
                }
            }

            if ((threadIdx.y > 0 || threadIdx.x < BLOCK_COLS - 1) && (col + 1 < img.cols)) {
                if (img[img_index + 1]) {
                    P |= 0x0E;				// 00001110
                }
            }

            if ((threadIdx.x > 0) && (row + 1 < img.rows)) {
                if (img[img_index + img.step]) {
                    P |= 0x30;				// 00110000
                }
            }

            if (threadIdx.x == 0) {
                P &= 0xCE;					// 11001110
            }
            if (col + 1 >= img.cols) {
                P &= 0xF3;					// 11110011
            }
            else if ((threadIdx.x + 1 == BLOCK_COLS) || (col + 2 >= img.cols)) {
                P &= 0xF7;					// 11110111
            }

            if (threadIdx.y == 0) {
                P &= 0xF0;					// 11110000
            }
            if (row + 1 >= img.rows) {
                P &= 0xDF;					// 11011111
            }

            // P is now ready to be used to find neighbour blocks (or it should be)
            // P value avoids range errors

            if (P > 0) {

                if (HasBit(P, 0) && img[img_index - img.step - 1]) {
                    Union(buf, buf_index, buf_index - BLOCK_COLS - 1);
                }

                if ((HasBit(P, 1) && img[img_index - img.step]) || (HasBit(P, 2) && img[img_index + 1 - img.step])) {
                    Union(buf, buf_index, buf_index - BLOCK_COLS);
                }

                if (HasBit(P, 3) && img[img_index + 2 - img.step]) {
                    Union(buf, buf_index, buf_index + 1 - BLOCK_COLS);
                }

                if ((HasBit(P, 4) && img[img_index - 1]) || (HasBit(P, 5) && img[img_index + img.step - 1])) {
                    Union(buf, buf_index, buf_index - 1);
                }
            }
        }

        __syncthreads();

        // Local compression
        if (row < labels.rows && col < labels.cols) {
            unsigned f = Find(buf, buf_index);
            unsigned f_row = f / BLOCK_COLS;
            unsigned f_col = f % BLOCK_COLS;
            unsigned global_f = 2 * (blockIdx.y * BLOCK_ROWS + f_row) * (labels.step / labels.elem_size) + 2 * (blockIdx.x * BLOCK_COLS + f_col);
            labels.data[labels_index] = global_f;
        }
    }

    __global__ void GlobalMerge(cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
        unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
        unsigned img_index = row * img.step + col;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {

            unsigned char P = 0;

            if (((threadIdx.x == 0 && col > 0) || (threadIdx.y == 0 && row > 0))) {
                if (img[img_index]) {
                    P |= 0x37;				// 00110111
                }
            }

            if (((threadIdx.y == 0 && row > 0) || (threadIdx.x == BLOCK_COLS - 1 && col + 2 < img.cols)) && (col + 1 < img.cols)) {
                if (img[img_index + 1]) {
                    P |= 0x0E;				// 00001110
                }
            }

            if ((threadIdx.x == 0 && col > 0) && (row + 1 < img.rows)) {
                if (img[img_index + img.step]) {
                    P |= 0x30;				// 00110000
                }
            }

            if (col == 0) {
                P &= 0xCE;					// 11001110
            }
            if (col + 1 >= img.cols) {
                P &= 0xF3;					// 11110011
            }
            else if (col + 2 >= img.cols) {
                P &= 0xF7;					// 11110111
            }

            if (row == 0) {
                P &= 0xF0;					// 11110000
            }
            if (row + 1 >= img.rows) {
                P &= 0xDF;					// 11011111
            }

            // P is now ready to be used to find neighbour blocks (or it should be)
            // P value avoids range errors

            if (P > 0) {

                if (HasBit(P, 0) && img[img_index - img.step - 1]) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) - 2);
                }

                if ((HasBit(P, 1) && img[img_index - img.step]) || (HasBit(P, 2) && img[img_index + 1 - img.step])) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));
                }

                if (HasBit(P, 3) && img[img_index + 2 - img.step]) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2);
                }

                if ((HasBit(P, 4) && img[img_index - 1]) || (HasBit(P, 5) && img[img_index + img.step - 1])) {
                    Union(labels.data, labels_index, labels_index - 2);
                }
            }

        }
    }

    __global__ void Compression(cuda::PtrStepSzi labels) {

        unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
        unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {
            labels[labels_index] = Find(labels.data, labels_index);
        }
    }

    __global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
        unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;
        unsigned img_index = row * (img.step / img.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {

            int label = labels[labels_index] + 1;

            if (img[img_index])
                labels[labels_index] = label;
            else {
                labels[labels_index] = 0;
            }

            if (col + 1 < labels.cols) {
                if (img[img_index + 1])
                    labels[labels_index + 1] = label;
                else {
                    labels[labels_index + 1] = 0;
                }

                if (row + 1 < labels.rows) {
                    if (img[img_index + img.step + 1])
                        labels[labels_index + (labels.step / labels.elem_size) + 1] = label;
                    else {
                        labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
                    }
                }
            }

            if (row + 1 < labels.rows) {
                if (img[img_index + img.step])
                    labels[labels_index + (labels.step / labels.elem_size)] = label;
                else {
                    labels[labels_index + (labels.step / labels.elem_size)] = 0;
                }
            }

        }

    }

}

class BKE_2S : public GpuLabeling2D<CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    unsigned char *last_pixel_;
    bool last_pixel_allocated_;

public:
    BKE_2S() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);

        last_pixel_allocated_ = false;
        if ((d_img_.rows == 1 || d_img_.cols == 1) && !((d_img_.rows + d_img_.cols) % 2)) {
            cudaMalloc(&last_pixel_, sizeof(unsigned char));
            last_pixel_allocated_ = true;
        }
        else {
            // last_pixel_ = d_img_labels_.data + d_img_labels_.step + sizeof(unsigned int);
            last_pixel_ = d_img_labels_.data + ((d_img_labels_.rows - 2) * d_img_labels_.step) + (d_img_labels_.cols - 2) * d_img_labels_.elemSize();
        }

        grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        LocalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //cuda::GpuMat d_expanded_connections;
        //d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
        //ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
        //Mat1b expanded_connections;
        //d_expanded_connections.download(expanded_connections);
        //d_expanded_connections.release();

        //Mat1i local_labels;
        //cuda::GpuMat d_local_merge;
        //d_img_labels_.copyTo(d_local_merge);
        //FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_local_merge);
        //d_local_merge.download(local_labels);

        GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //Mat1i block_info_final;
        //d_img_labels_.download(block_info_final);

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //d_img_labels_.download(img_labels_);
        if (last_pixel_allocated_) {
            cudaFree(last_pixel_);
        }
        cudaDeviceSynchronize();
    }


private:
    void Alloc() {
        d_img_labels_.create(d_img_.size(), CV_32SC1);
    }

    void Dealloc() {
    }

    double MemoryTransferHostToDevice() {
        perf_.start();
        d_img_.upload(img_);
        perf_.stop();
        return perf_.last();
    }

    void MemoryTransferDeviceToHost() {
        d_img_labels_.download(img_labels_);
    }

    void AllScans() {

        grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        LocalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //cuda::GpuMat d_expanded_connections;
        //d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
        //ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
        //Mat1b expanded_connections;
        //d_expanded_connections.download(expanded_connections);
        //d_expanded_connections.release();

        //Mat1i init_labels;
        //d_block_labels_.download(init_labels);

        GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //Mat1i block_info_final;
        //d_img_labels_.download(block_info_final);		

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        // d_img_labels_.download(img_labels_);
        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
        perf_.start();
        Alloc();
        perf_.stop();
        double alloc_timing = perf_.last();

        perf_.start();
        AllScans();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        double dealloc_timing = perf_.last();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
    }

};

REGISTER_LABELING(BKE_2S);
