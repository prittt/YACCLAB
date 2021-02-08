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

// Stava and Benes

#define TILE_SIZE 16

#define TILE_GRID_SIZE 4
#define THREAD_PER_TILE 16

using namespace cv;

namespace {

    // Returns the root index of the UFTree
    __device__ unsigned Find(const int* s_buf, unsigned n) {
        // Warning: do not call Find on a background pixel

        unsigned label = s_buf[n];

        assert(label > 0);

        while (label - 1 != n) {
            n = label - 1;
            label = s_buf[n];

            assert(label > 0);
        }

        return n;

    }

    __device__ void Union(int* s_buf, unsigned index_a, unsigned index_b, char* changed) {

        unsigned a = s_buf[index_a];
        if (!a) return;
        unsigned b = s_buf[index_b];
        if (!b) return;
        --a;
        --b;

        a = Find(s_buf, a);
        b = Find(s_buf, b);

        if (a != b) {
            *changed = 1;
        }

        if (a < b) {
            atomicMin(s_buf + b, a + 1);
        }
        else if (b < a) {
            atomicMin(s_buf + a, b + 1);
        }
    }

    
    // Perform local CCL on 16x16 tiles
    __global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        const unsigned r = threadIdx.y;
        const unsigned c = threadIdx.x;
        const unsigned local_index = r * blockDim.x + c;

        const unsigned global_row = blockIdx.y * blockDim.y + r;
        const unsigned global_col = blockIdx.x * blockDim.x + c;
        const unsigned img_index = global_row * img.step + global_col;

        __shared__ int s_buf[TILE_SIZE * TILE_SIZE];
        __shared__ unsigned char s_img[TILE_SIZE * TILE_SIZE];

        __shared__ char changed[1];

        bool in_limits = (global_row < img.rows&& global_col < img.cols);

        s_img[local_index] = in_limits ? img[img_index] : 0;
        unsigned char v = s_img[local_index];

        int label = v ? local_index + 1 : 0;

        __syncthreads();

        while (1) {

            // Pass 1 of the CCL algorithm
            s_buf[local_index] = label;

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                changed[0] = 0;
            }

            int new_label = label;

            __syncthreads();

            // Find the minimal label from the neighboring elements

            if (label) {

                if (r > 0 && c > 0 && s_img[local_index - TILE_SIZE - 1]) {
                    new_label = min(new_label, s_buf[local_index - TILE_SIZE - 1]);
                }
                if (r > 0 && s_img[local_index - TILE_SIZE]) {
                    new_label = min(new_label, s_buf[local_index - TILE_SIZE]);
                }
                if (r > 0 && c < TILE_SIZE - 1 && s_img[local_index - TILE_SIZE + 1]) {
                    new_label = min(new_label, s_buf[local_index - TILE_SIZE + 1]);
                }
                if (c > 0 && s_img[local_index - 1]) {
                    new_label = min(new_label, s_buf[local_index - 1]);
                }
                if (c < TILE_SIZE - 1 && s_img[local_index + 1]) {
                    new_label = min(new_label, s_buf[local_index + 1]);
                }
                if (r < TILE_SIZE - 1 && c > 0 && s_img[local_index + TILE_SIZE - 1]) {
                    new_label = min(new_label, s_buf[local_index + TILE_SIZE - 1]);
                }
                if (r < TILE_SIZE - 1 && s_img[local_index + TILE_SIZE]) {
                    new_label = min(new_label, s_buf[local_index + TILE_SIZE]);
                }
                if (r < TILE_SIZE - 1 && c < TILE_SIZE - 1 && s_img[local_index + TILE_SIZE + 1]) {
                    new_label = min(new_label, s_buf[local_index + TILE_SIZE + 1]);
                }

            }

            __syncthreads();

            // If the new label is smaller than the old one merge the equivalence trees

            if (new_label < label) {

                atomicMin(s_buf + label - 1, new_label);
                changed[0] = 1;

            }

            __syncthreads();


            if (changed[0] == 0)
                break;

            if (label) {

                // Pass 2 of the CCL algorithm

                label = Find(s_buf, label - 1) + 1;

            }

            __syncthreads();

        }

        if (in_limits) {
            // Store the result to the device memory
            int global_label = 0;

            if (v) {
                unsigned f_row = (label - 1) / TILE_SIZE;
                unsigned f_col = (label - 1) % TILE_SIZE;
                global_label = (blockIdx.y * TILE_SIZE + f_row) * (labels.step / labels.elem_size) + (blockIdx.x * TILE_SIZE + f_col) + 1;
            }
            labels.data[global_row * labels.step / sizeof(int) + global_col] = global_label;
        }

    }


    __global__ void GlobalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, uint32_t subBlockDim) {

        // Coordinates of the top-left pixel of the current block of tiles
        unsigned block_row = blockIdx.y * blockDim.y * subBlockDim;
        unsigned block_col = blockIdx.x * blockDim.x * subBlockDim;

        // Coordinates of the top-left pixel of the current tile
        unsigned tile_row = block_row + threadIdx.y * subBlockDim;
        unsigned tile_col = block_col + threadIdx.x * subBlockDim;

        unsigned repetitions = (subBlockDim + blockDim.z - 1) / blockDim.z;

        __shared__ char changed[1];

        while (1) {

            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                changed[0] = 0;
            }

            __syncthreads();

            // Process the bottom horizontal border
            for (unsigned i = 0; i < repetitions; i++) {

                unsigned r = tile_row + subBlockDim - 1;
                unsigned c = tile_col + i * blockDim.z + threadIdx.z;

                if (threadIdx.y < blockDim.y - 1 && r < img.rows - 1 && c < img.cols && c < tile_col + subBlockDim) {

                    if (c > block_col) {
                        Union(labels.data, r * labels.step / sizeof(int) + c, (r + 1) * labels.step / sizeof(int) + c - 1, changed);
                    }

                    Union(labels.data, r * labels.step / sizeof(int) + c, (r + 1) * labels.step / sizeof(int) + c, changed);

                    if (c < img.cols - 1 && c < block_col + blockDim.x * subBlockDim - 1) {
                        Union(labels.data, r * labels.step / sizeof(int) + c, (r + 1) * labels.step / sizeof(int) + c + 1, changed);
                    }

                }

            }

            // Process the right vertical border
            for (unsigned i = 0; i < repetitions; i++) {

                unsigned c = tile_col + subBlockDim - 1;
                unsigned r = tile_row + i * blockDim.z + threadIdx.z;

                if (threadIdx.x < blockDim.x - 1 && c < img.cols - 1 && r < img.rows && r < tile_row + subBlockDim) {

                    if (r > block_row) {
                        Union(labels.data, r * labels.step / sizeof(int) + c, (r - 1) * labels.step / sizeof(int) + c + 1, changed);
                    }

                    Union(labels.data, r * labels.step / sizeof(int) + c, r * labels.step / sizeof(int) + c + 1, changed);

                    if (r < img.rows - 1 && r < block_row + blockDim.y * subBlockDim - 1) {
                        Union(labels.data, r * labels.step / sizeof(int) + c, (r + 1) * labels.step / sizeof(int) + c + 1, changed);
                    }

                }

            }

            __syncthreads();

            if (changed[0] == 0) {
                break;  // the tiles are merged
            }

            __syncthreads();

        }

    }


    __global__ void BorderCompression(cuda::PtrStepSzi labels, uint32_t subBlockDim) {

        // Coordinates of the top-left pixel of the current block of tiles
        const unsigned block_row = blockIdx.y * blockDim.y * subBlockDim;
        const unsigned block_col = blockIdx.x * blockDim.x * subBlockDim;

        // Coordinates of the top-left pixel of the current tile
        const unsigned tile_row = block_row + threadIdx.y * subBlockDim;
        const unsigned tile_col = block_col + threadIdx.x * subBlockDim;

        const unsigned repetitions = (subBlockDim + blockDim.z - 1) / blockDim.z;

        // Process the bottom horizontal border
        for (unsigned i = 0; i < repetitions; i++) {

            const unsigned r = tile_row + subBlockDim - 1;
            const unsigned c = tile_col + i * blockDim.z + threadIdx.z;

            if (threadIdx.y < blockDim.y - 1 && r < labels.rows - 1 && c < labels.cols && c < tile_col + subBlockDim) {

                int label = labels[r * labels.step / sizeof(int) + c];
                if (label) {
                    labels[r * labels.step / sizeof(int) + c] = Find(labels, label - 1) + 1;
                }

            }

        }

        // Process the right vertical border
        for (unsigned i = 0; i < repetitions; i++) {

            const unsigned c = tile_col + subBlockDim - 1;
            const unsigned r = tile_row + i * blockDim.z + threadIdx.z;

            if (threadIdx.x < blockDim.x - 1 && c < labels.cols - 1 && r < labels.rows && r < tile_row + subBlockDim) {

                int label = labels[r * labels.step / sizeof(int) + c];
                if (label) {
                    labels[r * labels.step / sizeof(int) + c] = Find(labels, label - 1) + 1;
                }

            }

        }

    }


    __global__ void PathCompression(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned global_row = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned global_col = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

        if (global_row < labels.rows && global_col < labels.cols) {
            unsigned char val = img[global_row * img.step + global_col];
            if (val) {
                labels[labels_index] = Find(labels.data, labels_index) + 1;
            }
        }
    }

}

class STAVA : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    STAVA() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + TILE_SIZE - 1) / TILE_SIZE, (d_img_.rows + TILE_SIZE - 1) / TILE_SIZE, 1);
        block_size_ = dim3(TILE_SIZE, TILE_SIZE, 1);

        // Phase 1
        // Label pixels locally to a tile
        LocalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        // Phase 1 output
        //cuda::GpuMat d_local_labels;
        //d_img_labels_.copyTo(d_local_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        //Mat1i local_labels(img_.size());
        //d_local_labels.download(local_labels);

        // Phase 2
        // Merges together Union-Find trees from different tiles, in a recursive manner
        uint32_t max_img_dim = max(img_.rows, img_.cols);
        uint32_t sub_block_dim = TILE_SIZE;
        uint32_t block_pixels = sub_block_dim * TILE_GRID_SIZE;

        dim3 grid_size_merge;
        dim3 block_size_merge = dim3(TILE_GRID_SIZE, TILE_GRID_SIZE, THREAD_PER_TILE);

        while (sub_block_dim < max_img_dim) {
            grid_size_merge = dim3((d_img_.cols + block_pixels - 1) / block_pixels, (d_img_.rows + block_pixels - 1) / block_pixels, 1);
            
            GlobalMerge << <grid_size_merge, block_size_merge >> > (d_img_, d_img_labels_, sub_block_dim);
            
            BorderCompression << <grid_size_merge, block_size_merge >> > (d_img_labels_, sub_block_dim);

            sub_block_dim = block_pixels;
            block_pixels *= TILE_GRID_SIZE;

            // Phase 2 output
            //cuda::GpuMat d_global_labels;
            //d_img_labels_.copyTo(d_global_labels);
            //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
            //Mat1i global_labels(img_.size());
            //d_global_labels.download(global_labels);
        }

        // Phase 3
        PathCompression << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

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

    void LocalScan() {
        grid_size_ = dim3((d_img_.cols + TILE_SIZE - 1) / TILE_SIZE, (d_img_.rows + TILE_SIZE - 1) / TILE_SIZE, 1);
        block_size_ = dim3(TILE_SIZE, TILE_SIZE, 1);
        LocalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);
        cudaDeviceSynchronize();

    }

    void GlobalScan() {
        uint32_t max_img_dim = max(img_.rows, img_.cols);
        uint32_t sub_block_dim = TILE_SIZE;
        uint32_t block_pixels = sub_block_dim * TILE_GRID_SIZE;

        dim3 grid_size_merge;
        dim3 block_size_merge = dim3(TILE_GRID_SIZE, TILE_GRID_SIZE, THREAD_PER_TILE);

        while (sub_block_dim < max_img_dim) {
            grid_size_merge = dim3((d_img_.cols + block_pixels - 1) / block_pixels, (d_img_.rows + block_pixels - 1) / block_pixels, 1);

            GlobalMerge << <grid_size_merge, block_size_merge >> > (d_img_, d_img_labels_, sub_block_dim);

            BorderCompression << <grid_size_merge, block_size_merge >> > (d_img_labels_, sub_block_dim);

            sub_block_dim = block_pixels;
            block_pixels *= TILE_GRID_SIZE;
        }

        // Phase 3
        PathCompression << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
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

REGISTER_LABELING(STAVA);
