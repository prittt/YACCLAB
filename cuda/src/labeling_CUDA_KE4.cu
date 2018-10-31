#include <opencv2\cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

// Il minimo per entrambi è 4
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {


    // Return the root of a tree
    __device__ unsigned Find(const int *s_buf, unsigned n) {

        unsigned label = s_buf[n];

        while (label - 1 != n) {
            n = label - 1;
            label = s_buf[n];
        }

        return n;

    }

    // Links together trees containing a and b
    __device__ void Union(int *s_buf, unsigned a, unsigned b) {

        bool done;

        do {

            a = Find(s_buf, a);
            b = Find(s_buf, b);

            if (a < b) {
                int old = atomicMin(s_buf + b, a + 1);
                done = (old == b + 1);
                b = old - 1;
            }
            else if (b < a) {
                int old = atomicMin(s_buf + a, b + 1);
                done = (old == a + 1);
                a = old - 1;
            }
            else {
                done = true;
            }

        } while (!done);

    }


    // Init phase.
    // Labels start at value 1, to differentiate them from background, that has value 0.
    __global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
        unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
        unsigned img_index = row * img.step + col;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < img.rows && col < img.cols) {

            if (img.data[img_index]) {

                if (row > 0 && img.data[img_index - img.step]) {
                    labels.data[labels_index] = labels_index - (labels.step / labels.elem_size) + 1;
                }

                else if (col > 0 && img.data[img_index - 1]) {
                    labels.data[labels_index] = labels_index;
                }

                else {
                    labels.data[labels_index] = labels_index + 1;
                }
            }
        }
    }


    // Analysis phase.
    __global__ void Analyze(cuda::PtrStepSzi labels) {

        unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
        unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {

            unsigned label = labels[labels_index];

            if (label) {

                unsigned index = labels_index;

                while (label - 1 != index) {
                    index = label - 1;
                    label = labels[index];
                }

                labels[labels_index] = label;
            }
        }
    }

    __global__ void Reduce(const cuda::PtrStepb img, cuda::PtrStepSzi labels) {

        unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
        unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
        unsigned img_index = row * img.step + col;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {

            if (img.data[img_index]) {

                if (col > 0 && img.data[img_index - 1]) {
                    Union(labels.data, labels_index, labels_index - 1);
                }

            }

        }


    }

}

class CUDA_KE4 : public GpuLabeling2D<CONN_4> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    CUDA_KE4() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);

        grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

        Reduce << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

        cudaDeviceSynchronize();
    }


private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        perf_.stop();
        return perf_.last();
    }

    double Dealloc() {
        perf_.start();
        perf_.stop();
        return perf_.last();
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
        d_img_labels_.create(d_img_.size(), CV_32SC1);

        grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

        Reduce << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        AllScans();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        double dealloc_timing = Dealloc();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);

    }

};

REGISTER_LABELING(CUDA_KE4);

