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

__device__ void merge(int* L, unsigned int label_1, unsigned int label_2) {

    while (label_1 != label_2 && (label_1 != L[label_1] - 1)) {
        label_1 = L[label_1] - 1;
    }
    while (label_1 != label_2 && (label_2 != L[label_2] - 1)) {
        label_2 = L[label_2] - 1;
    }
    while (label_1 != label_2) {
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
                if (img.cols - i < 32) {
                    mask = mask >> (32 - (img.cols - i));
                }

                const unsigned img_index = img_line_base + i;
                const unsigned labels_index = labels_line_base + i;

                int p_y = img.data[img_index];

                unsigned pixels_y = __ballot_sync(mask, p_y);
                unsigned s_dist_y = start_distance(pixels_y, threadIdx.x);

                if (p_y && s_dist_y == 0) {
                    labels.data[labels_index] = labels_index - ((threadIdx.x == 0) ? distance_y : 0) + 1;
                }

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

                if (p_y && p_y_1 && (s_dist_y == 0 || s_dist_y_1 == 0)) {
                    int label_1 = labels_index - s_dist_y;
                    int label_2 = labels_index - s_dist_y_1 - (labels.step / labels.elem_size);
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

    unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_H;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < labels.rows && x < labels.cols && y > 0) {

        unsigned int mask = 0xFFFFFFFF;
        if (img.cols - blockIdx.x * blockDim.x < 32) {
            mask = mask >> (32 - (img.cols - blockIdx.x * blockDim.x));
        }

        const unsigned int img_index = y * img.step + x;
        const unsigned int labels_index = y * (labels.step / labels.elem_size) + x;

        const unsigned int img_index_up = img_index - img.step;
        const unsigned int labels_index_up = labels_index - (labels.step / labels.elem_size);

        const int p = img.data[img_index];
        const int p_up = img.data[img_index_up];

        const unsigned int pixels = __ballot_sync(mask, p);
        const unsigned int pixels_up = __ballot_sync(mask, p_up);

        if (p && p_up) {
            const unsigned int s_dist = start_distance(pixels, threadIdx.x);
            const unsigned int s_dist_up = start_distance(pixels_up, threadIdx.x);
            if (s_dist == 0 || s_dist_up == 0) {
                merge(labels.data, labels_index - s_dist, labels_index_up - s_dist_up);
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

class HA4 : public GpuLabeling2D<Connectivity2D::CONN_4> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    HA4() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            1,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        StripLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            (d_img_.cols + block_size_.x - 1) / block_size_.x,
            ((d_img_.rows + BLOCK_H - 1) / BLOCK_H + block_size_.y - 1) / block_size_.y,
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

        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            1,
            (d_img_.rows + block_size_.y - 1) / block_size_.y,
            1);

        StripLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        block_size_ = dim3(WARP_SIZE, BLOCK_H, 1);
        grid_size_ = dim3(
            (d_img_.cols + block_size_.x - 1) / block_size_.x,
            ((d_img_.rows + BLOCK_H - 1) / BLOCK_H + block_size_.y - 1) / block_size_.y,
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

REGISTER_LABELING(HA4);
