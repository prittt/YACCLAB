// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// 
// The algorithm is presented in "A new Direct Connected
// Component Labeling and Analysis Algorithms for GPUs",
// A. Hennequin, L. Lacassagne, L. Cabaret, Q. Meunier, 
// DASIP, 2018
//
// This is a modification of the algorithm, to work with
// 8-connectivity. The original 4-conn algorithm is in
// labeling_hennequin_2018_HA4.cu


#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

//// README
// The algorithm has been modified, in order to make labels start at 0


#define WARP_SIZE 32
#define BLOCK_H 4

using namespace cv;

namespace {

__device__ void merge(int* L, int label_1, int label_2) {

    while (label_1 != label_2 && (label_1 != L[label_1] - 1)) {
        assert(label_1 >= 0);
        label_1 = L[label_1] - 1;
    }
    while (label_1 != label_2 && (label_2 != L[label_2] - 1)) {
        assert(label_2 >= 0);
        label_2 = L[label_2] - 1;
    }
    while (label_1 != label_2) {
        assert(label_1 >= 0 && label_2 >= 0);
        if (label_1 < label_2) {
            unsigned int tmp = label_1;
            label_1 = label_2;
            label_2 = tmp;
        }
        unsigned int label_3 = atomicMin(L + label_1, label_2 + 1) - 1;
        if (label_1 == label_3) {
            label_1 = label_2;
        }
        else {
            label_1 = label_3;
        }
    }
}


__device__ unsigned int start_distance(unsigned int pixels, unsigned int tx) {

    return __clz(~(pixels << (32 - tx)));

}

__device__ unsigned int end_distance(unsigned int pixels, unsigned int tx) {

    return __ffs(~(pixels >> (tx + 1)));

}

__global__ void StripLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    __shared__ unsigned int shared_pixels[BLOCK_H];

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < labels.rows) {

        unsigned int img_line_base = y * img.step + threadIdx.x;
        unsigned int labels_line_base = y * (labels.step / labels.elem_size) + threadIdx.x;

        int distance_y = 0;
        int distance_y_1 = 0;

        for (int i = 0; i < img.cols; i += WARP_SIZE) {

            const unsigned int x = threadIdx.x + i;
            if (x < labels.cols) {

                unsigned int mask = 0xFFFFFFFF;
                const unsigned int involved_cols = img.cols - i;
                if (involved_cols < 32) {
                    mask = mask >> (32 - involved_cols);
                }

                const unsigned img_index = img_line_base + i;
                const unsigned labels_index = labels_line_base + i;

                int p_y = img.data[img_index];

                unsigned pixels_y = __ballot_sync(mask, p_y);
                unsigned s_dist_y = start_distance(pixels_y, threadIdx.x);

                if (p_y && s_dist_y == 0) {
                    labels.data[labels_index] = labels_index - ((threadIdx.x == 0) ? distance_y : 0) + 1;
                }

#if __CUDA_ARCH__ < 700
                __syncthreads();
#endif

                if (threadIdx.x == 0) {
                    shared_pixels[threadIdx.y] = pixels_y;
                }

                __syncthreads();

                unsigned int pixels_y_1 = (threadIdx.y > 0) ? shared_pixels[threadIdx.y - 1] : 0;
                unsigned int p_y_1 = (pixels_y_1 >> threadIdx.x) & 1u;
                unsigned int s_dist_y_1 = start_distance(pixels_y_1, threadIdx.x);

                if (threadIdx.x == 0) {
                    s_dist_y = distance_y;
                    s_dist_y_1 = distance_y_1;
                }

                // Added for 8-connectivity
                const unsigned int pixels_y_shifted = (pixels_y << 1) | (distance_y > 0);
                const unsigned int pixels_y_1_shifted = (pixels_y_1 << 1) | (distance_y_1 > 0);

                if (p_y && p_y_1 && (s_dist_y == 0 || s_dist_y_1 == 0)) {
                    int label_1 = labels_index - s_dist_y;
                    int label_2 = labels_index - s_dist_y_1 - (labels.step / labels.elem_size);
                    merge(labels.data, label_1, label_2);
                }
                // Added for 8-connectivity
                else if (p_y && s_dist_y == 0 && ((pixels_y_1_shifted >> threadIdx.x) & 1u)) {
                    unsigned int s_dist_y_1_prev = (threadIdx.x == 0) ? (distance_y_1 - 1) : start_distance(pixels_y_1, threadIdx.x - 1);
                    int label_1 = labels_index;
                    int label_2 = labels_index - (labels.step / labels.elem_size) - 1 - s_dist_y_1_prev;
                    merge(labels.data, label_1, label_2);
                }
                else if (p_y_1 && s_dist_y_1 == 0 && ((pixels_y_shifted >> threadIdx.x) & 1u)) {
                    unsigned int s_dist_y_prev = (threadIdx.x == 0) ? (distance_y - 1) : (start_distance(pixels_y, threadIdx.x - 1));
                    int label_1 = labels_index - 1 - s_dist_y_prev;
                    int label_2 = labels_index - (labels.step / labels.elem_size);
                    merge(labels.data, label_1, label_2);
                }

                int d = start_distance(pixels_y_1, 32);
                distance_y_1 = d + ((d == 32) ? distance_y_1 : 0);
                d = start_distance(pixels_y, 32);
                distance_y = d + ((d == 32) ? distance_y : 0);

            }
        }
    }
}

__global__ void StripMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    const unsigned int warp_starting_x = blockIdx.x * (blockDim.x * blockDim.z - WARP_SIZE) + threadIdx.z * WARP_SIZE;
    const unsigned int y = (blockIdx.y + 1) * BLOCK_H;
    const unsigned int x = warp_starting_x + threadIdx.x;

    __shared__ unsigned last_dist_vec[32];  // Magic number could be removed
    __shared__ unsigned last_dist_up_vec[32];

    if (y < labels.rows && x < labels.cols) {

        unsigned int mask = 0xFFFFFFFF;
        if (img.cols - warp_starting_x < 32) {
            mask = mask >> (32 - (img.cols - warp_starting_x));
        }

        const unsigned int img_index = y * img.step + x;
        const unsigned int labels_index = y * (labels.step / labels.elem_size) + x;

        const unsigned int img_index_up = img_index - img.step;
        const unsigned int labels_index_up = labels_index - (labels.step / labels.elem_size);

        const int p = img.data[img_index];
        const int p_up = img.data[img_index_up];

        const unsigned int pixels = __ballot_sync(mask, p);
        const unsigned int pixels_up = __ballot_sync(mask, p_up);

        const unsigned int s_dist = start_distance(pixels, threadIdx.x);
        const unsigned int s_dist_up = start_distance(pixels_up, threadIdx.x);

        if (threadIdx.x == WARP_SIZE - 1) {
            last_dist_vec[threadIdx.z] = start_distance(pixels, 32);
            last_dist_up_vec[threadIdx.z] = start_distance(pixels_up, 32);
        }

        __syncthreads();

        if (blockIdx.x == 0 || threadIdx.z > 0) {    // There is a 32-pixel overlapping between 1024-long lines

            const unsigned int last_dist = threadIdx.z > 0 ? last_dist_vec[threadIdx.z - 1] : 0;
            const unsigned int last_dist_up = threadIdx.z > 0 ? last_dist_up_vec[threadIdx.z - 1] : 0;

            const unsigned int p_prev = threadIdx.x > 0 ? ((pixels >> (threadIdx.x - 1)) & 1u) : last_dist;
            const unsigned int p_up_prev = threadIdx.x > 0 ? ((pixels_up >> (threadIdx.x - 1)) & 1u) : last_dist_up;

            if (p && p_up) {
                if (s_dist == 0 || s_dist_up == 0) {
                    merge(labels.data, labels_index - s_dist, labels_index_up - s_dist_up);
                }
            }
            else if (p && p_up_prev && s_dist == 0) {
                const unsigned int s_dist_up_prev = (threadIdx.x == 0) ? (last_dist_up - 1) : start_distance(pixels_up, threadIdx.x - 1);
                merge(labels.data, labels_index, labels_index_up - 1 - s_dist_up_prev);
            }
            else if (p_prev && p_up && s_dist_up == 0) {
                const unsigned int s_dist_prev = (threadIdx.x == 0) ? (last_dist - 1) : (start_distance(pixels, threadIdx.x - 1));
                merge(labels.data, labels_index - 1 - s_dist_prev, labels_index_up);
            }

        }
    }

}

__global__ void Relabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < labels.cols && y < labels.rows) {

        unsigned int mask = 0xFFFFFFFF;
        if (img.cols - blockIdx.x * blockDim.x < 32) {
            mask = mask >> (32 - (img.cols - blockIdx.x * blockDim.x));
        }

        const unsigned int img_index = y * img.step + x;
        const unsigned int labels_index = y * (labels.step / labels.elem_size) + x;

        const int p = img.data[img_index];
        const unsigned int pixels = __ballot_sync(mask, p);
        const unsigned int s_dist = start_distance(pixels, threadIdx.x);
        int label = 0;

        if (p && s_dist == 0) {
            label = labels.data[labels_index] - 1;
            while (label != labels.data[label] - 1) {
                label = labels.data[label] - 1;
            }
        }

        label = __shfl_sync(mask, label, threadIdx.x - s_dist);

        if (p) {
            labels.data[labels_index] = label + 1;
        }
    }

}

}

class HA8 : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    HA8() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            1,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        StripLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        const unsigned int horizontal_warps = std::min((d_img_.cols + WARP_SIZE - 1) / WARP_SIZE, 32);
        block_size_ = dim3(WARP_SIZE, 1, horizontal_warps);
        grid_size_ = dim3(
            std::max((d_img_.cols + WARP_SIZE * 30 - 1) / (WARP_SIZE * 31), 1),
            (d_img_.rows - 1) / BLOCK_H,
            1);

        StripMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            (d_img_.cols + block_size_.x - 1) / block_size_.x,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        Relabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        cudaDeviceSynchronize();
    }


private:
    double Alloc() {

        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);
        cudaDeviceSynchronize();

        double t = perf_.stop();

        perf_.start();
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);
        cudaDeviceSynchronize();

        t -= perf_.stop();
        return t;
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

        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            1,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        StripLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        const unsigned int horizontal_warps = std::min((d_img_.cols + WARP_SIZE - 1) / WARP_SIZE, 32);
        block_size_ = dim3(WARP_SIZE, 1, horizontal_warps);
        grid_size_ = dim3(
            std::max((d_img_.cols + WARP_SIZE * 30 - 1) / (WARP_SIZE * 31), 1),
            (d_img_.rows - 1) / BLOCK_H,
            1);

        StripMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            (d_img_.cols + block_size_.x - 1) / block_size_.x,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        Relabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

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

        perf_.start();
        Dealloc();
        perf_.stop();
        double dealloc_timing = perf_.last();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
    }

};

REGISTER_LABELING(HA8);
