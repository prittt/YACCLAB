// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "labeling_algorithms.h"
#include "register.h"

// Rasmusson2013

#define BLOCK_SIZE 32   // this must be multiple of the warp size (leave it to 32)
#define PATCH_SIZE (BLOCK_SIZE + 2)

using namespace cv;
using namespace std;

namespace {

// This kernel makes use of a (BLOCK_SIZE + 2) X (BLOCK_SIZE + 2) array in shared memory 
// The paper actually says (BLOCK_SIZE + 1) X (BLOCK_SIZE + 1), but I can't manage to make it work that way
__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
    const unsigned img_patch_index = (threadIdx.y + 1) * PATCH_SIZE + threadIdx.x + 1;
    const unsigned local_linear_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    __shared__ unsigned char img_patch[PATCH_SIZE * PATCH_SIZE];

    const bool in_limits = r < img.rows&& c < img.cols;

    // Load 34 x 34 matrix from input image

    // Convert local_linear_index to coordinates of the 34 x 34 matrix

    // Round 1
    const int patch_r1 = local_linear_index / PATCH_SIZE;
    const int patch_c1 = local_linear_index % PATCH_SIZE;
    const int patch_img_r1 = blockIdx.y * BLOCK_SIZE - 1 + patch_r1;
    const int patch_img_c1 = blockIdx.x * BLOCK_SIZE - 1 + patch_c1;
    const int patch_img_index1 = patch_img_r1 * img.step + patch_img_c1;
    const bool patch_in_limits1 = patch_img_r1 >= 0 && patch_img_c1 >= 0 && patch_img_r1 < img.rows&& patch_img_c1 < img.cols;
    img_patch[patch_r1 * PATCH_SIZE + patch_c1] = patch_in_limits1 ? img[patch_img_index1] : 0;

    // Round 2
    const int patch_r2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) / PATCH_SIZE;
    const int patch_c2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) % PATCH_SIZE;
    if (patch_r2 < PATCH_SIZE) {
        const int patch_img_r2 = blockIdx.y * BLOCK_SIZE - 1 + patch_r2;
        const int patch_img_c2 = blockIdx.x * BLOCK_SIZE - 1 + patch_c2;
        const int patch_img_index2 = patch_img_r2 * img.step + patch_img_c2;
        const bool patch_in_limits2 = patch_img_r2 >= 0 && patch_img_c2 >= 0 && patch_img_r2 < img.rows&& patch_img_c2 < img.cols;
        img_patch[patch_r2 * PATCH_SIZE + patch_c2] = patch_in_limits2 ? img[patch_img_index2] : 0;
    }

    __syncthreads();


    if (in_limits) {

        unsigned int connections = 0;
        unsigned label = 0;
        if (img_patch[img_patch_index]) {
            label = labels_index + 1;

            // Enrich label with connections information

            if (img_patch[img_patch_index - PATCH_SIZE - 1]) {
                connections |= (1u << 31);
            }
            if (img_patch[img_patch_index - PATCH_SIZE]) {
                connections |= (1u << 30);
            }
            if (img_patch[img_patch_index - PATCH_SIZE + 1]) {
                connections |= (1u << 29);
            }
            if (img_patch[img_patch_index + 1]) {
                connections |= (1u << 28);
            }

            label |= connections;

        }
        labels[labels_index] = label;
    }
}

__global__ void Propagate(cuda::PtrStepSzi labels, char* changed) {

    bool thread_changed = false;
    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
    const unsigned labels_patch_index = (threadIdx.y + 1) * PATCH_SIZE + threadIdx.x + 1;
    const unsigned local_linear_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    __shared__ unsigned labels_patch[PATCH_SIZE * PATCH_SIZE];
    __shared__ bool something_changed[1];

    const bool in_limits = r < labels.rows&& c < labels.cols;

    // Load 34 x 34 matrix from input image

    // Convert local_linear_index to coordinates of the 34 x 34 matrix
    // 2 rounds are enough only for BLOCK_SIZE >= 5

    // Round 1
    const int patch_r1 = local_linear_index / PATCH_SIZE;
    const int patch_c1 = local_linear_index % PATCH_SIZE;
    const int patch_labels_r1 = blockIdx.y * BLOCK_SIZE - 1 + patch_r1;
    const int patch_labels_c1 = blockIdx.x * BLOCK_SIZE - 1 + patch_c1;
    const int patch_labels_index1 = patch_labels_r1 * (labels.step / labels.elem_size) + patch_labels_c1;
    const bool patch_in_limits1 = patch_labels_r1 >= 0 && patch_labels_c1 >= 0 && patch_labels_r1 < labels.rows&& patch_labels_c1 < labels.cols;
    labels_patch[patch_r1 * PATCH_SIZE + patch_c1] = patch_in_limits1 ? labels[patch_labels_index1] : 0;

    // Round 2
    const int patch_r2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) / PATCH_SIZE;
    if (patch_r2 < PATCH_SIZE) {
        const int patch_c2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) % PATCH_SIZE;
        const int patch_labels_r2 = blockIdx.y * BLOCK_SIZE - 1 + patch_r2;
        const int patch_labels_c2 = blockIdx.x * BLOCK_SIZE - 1 + patch_c2;
        const int patch_labels_index2 = patch_labels_r2 * (labels.step / labels.elem_size) + patch_labels_c2;
        const bool patch_in_limits2 = patch_labels_r2 >= 0 && patch_labels_c2 >= 0 && patch_labels_r2 < labels.rows&& patch_labels_c2 < labels.cols;
        labels_patch[patch_r2 * PATCH_SIZE + patch_c2] = patch_in_limits2 ? labels[patch_labels_index2] : 0;
    }

    do {

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            something_changed[0] = false;
        }

        __syncthreads();

        thread_changed = false;

        // Primary/Secondary Optimization
        // Find the primary pixel of the sub-component, and add its label to the propagation.
        if (true) {
            const unsigned label = labels_patch[labels_patch_index];
            unsigned min_label = label & 0x0FFFFFFF;

            if (min_label) {
                const int primary_r = ((label & 0x0FFFFFFF) - 1) / (labels.step / labels.elem_size);
                const int primary_c = ((label & 0x0FFFFFFF) - 1) % (labels.step / labels.elem_size);

                // Check if the primary pixel is in the same block
                // If it is, take its current label as the minimum
                if (primary_r >= (blockIdx.y * BLOCK_SIZE - 1) && primary_r <= (blockIdx.y + 1) * BLOCK_SIZE &&
                    primary_c >= (blockIdx.x * BLOCK_SIZE - 1) && primary_c <= (blockIdx.x + 1) * BLOCK_SIZE) {
                    const int primary_local_r = primary_r - blockIdx.y * BLOCK_SIZE;
                    const int primary_local_c = primary_c - blockIdx.x * BLOCK_SIZE;
                    min_label = min(min_label, labels_patch[(primary_local_r + 1) * PATCH_SIZE + (primary_local_c + 1)] & 0x0FFFFFFF);
                }
            }

            if (min_label < (label & 0x0FFFFFFF)) {
                labels_patch[labels_patch_index] = min_label | (label & 0xF0000000);
                thread_changed = true;
            }

        }

        __syncthreads();

        // Propagation sizes are calculated in every propagation step

        if (true) {
            // UP-LEFT

            // This is a bit convoluted, because we need pixels on a diagonal line 
            // to be processed by threads belonging to the same warp.

            // The pixel-warp mapping is the following (for WARP_SIZE = BLOCK_SIZE = 4):
            // +---+---+---+---+
            // | 0 | 1 | 2 | 3 |
            // +---+---+---+---+
            // | 3 | 0 | 1 | 2 |
            // +---+---+---+---+
            // | 2 | 3 | 0 | 1 |
            // +---+---+---+---+
            // | 1 | 2 | 3 | 0 |
            // +---+---+---+---+

            const unsigned patch_r_dir = threadIdx.x;
            const unsigned patch_c_dir = (threadIdx.x + threadIdx.y) % BLOCK_SIZE;
            const unsigned patch_index_dir = (patch_r_dir + 1) * PATCH_SIZE + patch_c_dir + 1;
            unsigned label_dir = labels_patch[patch_index_dir];
            unsigned min_label = label_dir & 0x0FFFFFFF;
            unsigned prop = ((label_dir >> 31) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_up_sync(0xffffffff, prop, prop);

                if (static_cast<int>(patch_r_dir) - static_cast<int>(prop) >= 0 &&
                    static_cast<int>(patch_c_dir) - static_cast<int>(prop) >= 0) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir - PATCH_SIZE - 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir - prop * (PATCH_SIZE + 1)];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            // DOWN-RIGHT
            prop = ((labels_patch[patch_index_dir + PATCH_SIZE + 1] >> 31) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_down_sync(0xffffffff, prop, prop);

                if (patch_r_dir + prop < BLOCK_SIZE && patch_c_dir + prop < BLOCK_SIZE) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir + PATCH_SIZE + 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir + prop * (PATCH_SIZE + 1)];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            if (min_label < (label_dir & 0x0FFFFFFF)) {
                labels_patch[patch_index_dir] = min_label | (label_dir & 0xF0000000);
                thread_changed = true;
            }

        }

        __syncthreads();

        if (true) {
            // UP-RIGHT

            // This is a bit convoluted, because we need pixels on a diagonal line 
            // to be processed by threads belonging to the same warp.

            // The pixel-warp mapping is the following (for WARP_SIZE = BLOCK_SIZE = 4):
            // +---+---+---+---+
            // | 1 | 2 | 3 | 0 |
            // +---+---+---+---+
            // | 2 | 3 | 0 | 1 |
            // +---+---+---+---+
            // | 3 | 0 | 1 | 2 |
            // +---+---+---+---+
            // | 0 | 1 | 2 | 3 |
            // +---+---+---+---+

            const unsigned patch_r_dir = threadIdx.x;
            const unsigned patch_c_dir = (BLOCK_SIZE - 1 - threadIdx.x + threadIdx.y) % BLOCK_SIZE;
            const unsigned patch_index_dir = (patch_r_dir + 1) * PATCH_SIZE + patch_c_dir + 1;
            unsigned label_dir = labels_patch[patch_index_dir];
            unsigned min_label = label_dir & 0x0FFFFFFF;
            unsigned prop = ((label_dir >> 29) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_up_sync(0xffffffff, prop, prop);

                if (static_cast<int>(patch_r_dir) - static_cast<int>(prop) >= 0 &&
                    patch_c_dir + prop < BLOCK_SIZE) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir - PATCH_SIZE + 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir - prop * (PATCH_SIZE - 1)];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            // DOWN-LEFT
            prop = ((labels_patch[patch_index_dir + PATCH_SIZE - 1] >> 29) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_down_sync(0xffffffff, prop, prop);

                if (patch_r_dir + prop < BLOCK_SIZE && static_cast<int>(patch_c_dir) - static_cast<int>(prop) >= 0) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir + PATCH_SIZE - 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir + prop * (PATCH_SIZE - 1)];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            if (min_label < (label_dir & 0x0FFFFFFF)) {
                labels_patch[patch_index_dir] = min_label | (label_dir & 0xF0000000);
                thread_changed = true;
            }

        }

        __syncthreads();

        if (true) {
            // UP

            // warp x takes care of COLUMN x, up to down
            unsigned patch_index_dir = (threadIdx.x + 1) * PATCH_SIZE + threadIdx.y + 1;
            unsigned label_dir = labels_patch[patch_index_dir];
            unsigned min_label = label_dir & 0x0FFFFFFF;
            unsigned prop = ((label_dir >> 30) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_up_sync(0xffffffff, prop, prop);

                if (static_cast<int>(threadIdx.x) - static_cast<int>(prop) >= 0) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir - PATCH_SIZE];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir - prop * PATCH_SIZE];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            // DOWN
            prop = ((labels_patch[patch_index_dir + PATCH_SIZE] >> 30) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_down_sync(0xffffffff, prop, prop);

                if (threadIdx.x + prop < BLOCK_SIZE) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir + PATCH_SIZE];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir + prop * PATCH_SIZE];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            if (min_label < (label_dir & 0x0FFFFFFF)) {
                labels_patch[patch_index_dir] = min_label | (label_dir & 0xF0000000);
                thread_changed = true;
            }

        }

        __syncthreads();

        if (true) {
            // RIGHT

            // patch_index_dir changes for every direction
            unsigned patch_index_dir = (threadIdx.y + 1) * PATCH_SIZE + threadIdx.x + 1;
            unsigned label_dir = labels_patch[patch_index_dir];
            unsigned min_label = label_dir & 0x0FFFFFFF;
            unsigned prop = ((label_dir >> 28) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_down_sync(0xffffffff, prop, prop);

                if (threadIdx.x + prop < BLOCK_SIZE) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir + 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir + prop];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            // LEFT
            prop = ((labels_patch[patch_index_dir - 1] >> 28) & 1);

            // 5 iterations are enough for the longest propagation
            // Maybe there is a way to end the cycle sooner
            for (int i = 0; i < 5; ++i) {

                unsigned delta = __shfl_up_sync(0xffffffff, prop, prop);

                if (static_cast<int>(threadIdx.x) - static_cast<int>(prop) >= 0) {
                    prop += delta;
                }
            }

            if (prop > 0) {
                // A propagation size of 1 must be mantained
                const unsigned close_label = labels_patch[patch_index_dir - 1];
                min_label = min(min_label, close_label & 0x0FFFFFFF);
                // The farthest label is gathered
                const unsigned far_label = labels_patch[patch_index_dir - prop];
                min_label = min(min_label, far_label & 0x0FFFFFFF);
            }

            if (min_label < (label_dir & 0x0FFFFFFF)) {
                labels_patch[patch_index_dir] = min_label | (label_dir & 0xF0000000);
                thread_changed = true;
            }

        }

        if (thread_changed) {
            something_changed[0] = true;
        }

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && something_changed[0]) {
            *changed = 1;
        }

    }
    //while (something_changed[0]);
    while (false);  // change this with the previous line to add an internal loop - it doesn't seem efficient

    if (in_limits) {

        labels[labels_index] = labels_patch[labels_patch_index];

    }

}


__global__ void End(cuda::PtrStepSzi labels) {

    unsigned global_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    unsigned global_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

    if (global_row < labels.rows && global_col < labels.cols) {
        labels.data[labels_index] &= 0x0FFFFFFF;
    }
}

}

class RASMUSSON : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    char* d_changed_ptr_;

public:
    RASMUSSON() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_img_.rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        block_size_ = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

        char changed = 1;
        char* d_changed_ptr;
        cudaMalloc(&d_changed_ptr, 1);

        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        while (changed) {
            changed = 0;

            cudaMemset(d_changed_ptr, 0, 1);

            Propagate << <grid_size_, block_size_ >> > (d_img_labels_, d_changed_ptr);

            cudaMemcpy(&changed, d_changed_ptr, 1, cudaMemcpyDeviceToHost);
        }

        End << <grid_size_, block_size_ >> > (d_img_labels_);

        cudaDeviceSynchronize();
    }


private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);

        grid_size_ = dim3((d_img_.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_img_.rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        block_size_ = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

        cudaMalloc(&d_changed_ptr_, 1);
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

    }

    void GlobalScan() {

        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
        // This doesn't really make sense, there are not two separate scans

        double alloc_timing = Alloc();

        perf_.start();
        LocalScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        GlobalScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        double dealloc_timing = Dealloc();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);

    }

};

REGISTER_LABELING(RASMUSSON);

