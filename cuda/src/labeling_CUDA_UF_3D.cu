#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

// Oliveira in 3D (nostro, pensiamo di essere i primi)

#define BLOCK_X 4
#define BLOCK_Y 4
#define BLOCK_Z 4

using namespace cv;

namespace {

    // Risale alla radice dell'albero a partire da un suo nodo n
    __device__ unsigned Find(const int *s_buf, unsigned n) {
        // Attenzione: non invocare la find su un pixel di background

        unsigned label = s_buf[n];

        assert(label > 0);

        while (label - 1 != n) {
            n = label - 1;
            label = s_buf[n];

            assert(label > 0);
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

    __global__ void Initialization(const cuda::PtrStepSz3b img, cuda::PtrStepSz3i labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned img_index = z * (img.stepz / img.elem_size) + y * (img.stepy / img.elem_size) + x;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {
            if (img[img_index]) {
                labels[labels_index] = labels_index + 1;
            }
            else {
                labels[labels_index] = 0;
            }
        }
    }


    __global__ void Merge(cuda::PtrStepSz3i labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {
            if (labels[labels_index]) {
                if (z > 0) {
                    unsigned current_plane = labels_index - (labels.stepz / labels.elem_size);
                    if (y > 0) {
                        unsigned current_row = current_plane - (labels.stepy / labels.elem_size);
                        if (x > 0 && labels[current_row - 1]) {
                            Union(labels.data, labels_index, current_row - 1);
                        }
                        if (labels[current_row]) {
                            Union(labels.data, labels_index, current_row);
                        }
                        if (x + 1 < labels.x && labels[current_row + 1]) {
                            Union(labels.data, labels_index, current_row + 1);
                        }
                    }
                    {
                        unsigned current_row = current_plane;
                        if (x > 0 && labels[current_row - 1]) {
                            Union(labels.data, labels_index, current_row - 1);
                        }
                        if (labels[current_row]) {
                            Union(labels.data, labels_index, current_row);
                        }
                        if (x + 1 < labels.x && labels[current_row + 1]) {
                            Union(labels.data, labels_index, current_row + 1);
                        }
                    }
                    if (y + 1 < labels.y) {
                        unsigned current_row = current_plane + (labels.stepy / labels.elem_size);
                        if (x > 0 && labels[current_row - 1]) {
                            Union(labels.data, labels_index, current_row - 1);
                        }
                        if (labels[current_row]) {
                            Union(labels.data, labels_index, current_row);
                        }
                        if (x + 1 < labels.x && labels[current_row + 1]) {
                            Union(labels.data, labels_index, current_row + 1);
                        }
                    }
                }
                {
                    if (y > 0) {
                        unsigned current_row = labels_index - (labels.stepy / labels.elem_size);
                        if (x > 0 && labels[current_row - 1]) {
                            Union(labels.data, labels_index, current_row - 1);
                        }
                        if (labels[current_row]) {
                            Union(labels.data, labels_index, current_row);
                        }
                        if (x + 1 < labels.x && labels[current_row + 1]) {
                            Union(labels.data, labels_index, current_row + 1);
                        }
                    }
                    {
                        if (x > 0 && labels[labels_index - 1]) {
                            Union(labels.data, labels_index, labels_index - 1);
                        }
                    }
                }
            }
        }
    }


    __global__ void PathCompression(cuda::PtrStepSz3i labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {
            unsigned int val = labels[labels_index];
            if (val) {
                labels[labels_index] = Find(labels.data, labels_index) + 1;
            }
        }
    }

}


class UF_3D : public GpuLabeling3D<Connectivity3D::CONN_26> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    UF_3D() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);

        grid_size_ = dim3((d_img_.x + BLOCK_X - 1) / BLOCK_X, (d_img_.y + BLOCK_Y - 1) / BLOCK_Y, (d_img_.z + BLOCK_Z - 1) / BLOCK_Z);
        block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

        //cuda::PtrStep3b ptr_step_prima(d_img_labels_);

        // Phase 1
        // Etichetta i pixel localmente al blocco		
        Initialization << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //cuda::PtrStepSz3i ptr_step_size(d_img_labels_);

        // Immagine di debug della prima fase
        //cuda::GpuMat d_local_labels;
        //d_img_labels_.copyTo(d_local_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        //Mat1i local_labels(img_.size());
        //d_local_labels.download(local_labels);

        // Phase 2
        // Collega tra loro gli alberi union-find dei diversi blocchi
        Merge << <grid_size_, block_size_ >> > (d_img_labels_);

        // Immagine di debug della seconda fase
        //cuda::GpuMat d_global_labels;
        //d_img_labels_.copyTo(d_global_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //Mat1i global_labels(img_.size());
        //d_global_labels.download(global_labels);

        // Phase 3
        // Collassa gli alberi union-find sulle radici
        PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);

        cudaDeviceSynchronize();

        //d_img_labels_.download(img_labels_);
        //Mat errors;
        //bool correct = CheckLabeledVolume(img_, img_labels_, errors);
        //volwrite("C:\\Users\\Stefano\\Desktop\\debug\\UF_errors", errors);

    }


private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);
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

    void LocalScan() {
        grid_size_ = dim3((d_img_.x + BLOCK_X - 1) / BLOCK_X, (d_img_.y + BLOCK_Y - 1) / BLOCK_Y, (d_img_.z + BLOCK_Z - 1) / BLOCK_Z);
        block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
        Initialization << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);
        cudaDeviceSynchronize();
    }

    void GlobalScan() {
        Merge << <grid_size_, block_size_ >> > (d_img_labels_);
        PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);
        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        LocalScan();
        GlobalScan();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        double dealloc_timing = Dealloc();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);

    }

};

REGISTER_LABELING(UF_3D);
