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


#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {

//
//         This is a block-based algorithm.
// Blocks are 2x2 sized, with internal pixels named as:
//                       +---+
//                       |a b|
//                       |c d|
//                       +---+   
// 
//       Neighbour blocks of block X are named as:
//                      +-+-+-+
//                      |P|Q|R|                
//                      +-+-+-+
//                      |S|X|
//                      +-+-+
//

enum class Info : unsigned char { a = 0, b = 1, c = 2, d = 3, P = 4, Q = 5, R = 6, S = 7 };

// Only use it with unsigned numeric types
template <typename T>
__device__ __forceinline__ unsigned char HasBit(T bitmap, Info pos) {
    return (bitmap >> static_cast<unsigned char>(pos)) & 1;
}

template <typename T>
__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

// Only use it with unsigned numeric types
__device__ __forceinline__ void SetBit(unsigned char& bitmap, Info pos) {
    bitmap |= (1 << static_cast<unsigned char>(pos));
}

// Returns the root index of the UFTree
__device__ unsigned Find(const int* s_buf, unsigned n) {
    while (s_buf[n] != n) {
        n = s_buf[n];
    }
    return n;
}

__device__ unsigned FindAndCompress(int* s_buf, unsigned n) {
    unsigned id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

// Merges the UFTrees of a and b, linking one root to the other
__device__ void Union(int* s_buf, unsigned a, unsigned b) {

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


__global__ void InitLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, unsigned char* last_pixel) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned img_index = row * img.step + col;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        unsigned P = 0;

        // Bitmask representing two kinds of information
        // Bits 0, 1, 2, 3 are set if pixel a, b, c, d are foreground, respectively
        // Bits 4, 5, 6, 7 are set if block P, Q, R, S need to be merged to X in Merge phase
        unsigned char info = 0;

        char buffer alignas(int)[4];
        *(reinterpret_cast<int*>(buffer)) = 0;

        // Read pairs of consecutive values in memory at once
        if (col + 1 < img.cols) {
            // This does not depend on endianness
            *(reinterpret_cast<int16_t*>(buffer)) = *(reinterpret_cast<int16_t*>(img.data + img_index));

            if (row + 1 < img.rows) {
                *(reinterpret_cast<int16_t*>(buffer + 2)) = *(reinterpret_cast<int16_t*>(img.data + img_index + img.step));
            }
        }
        else {
            buffer[0] = img.data[img_index];

            if (row + 1 < img.rows) {
                buffer[2] = img.data[img_index + img.step];
            }
        }

        if (buffer[0]) {
            P |= 0x777;
            SetBit(info, Info::a);
        }
        if (buffer[1]) {
            P |= (0x777 << 1);
            SetBit(info, Info::b);
        }
        if (buffer[2]) {
            P |= (0x777 << 4);
            SetBit(info, Info::c);
        }
        if (buffer[3]) {
            SetBit(info, Info::d);
        }

        if (col == 0) {
            P &= 0xEEEE;
        }
        if (col + 1 >= img.cols) {
            P &= 0x3333;
        }
        else if (col + 2 >= img.cols) {
            P &= 0x7777;
        }

        if (row == 0) {
            P &= 0xFFF0;
        }
        if (row + 1 >= img.rows) {
            P &= 0x00FF;
        }
        else if (row + 2 >= img.rows) {
            P &= 0x0FFF;
        }

        // P is now ready to be used to find neighbour blocks
        // P value avoids range errors

        int father_offset = 0;

        // P square
        if (HasBit(P, 0) && img.data[img_index - img.step - 1]) {
            father_offset = -(2 * (labels.step / labels.elem_size) + 2);
        }

        // Q square
        if ((HasBit(P, 1) && img.data[img_index - img.step]) || (HasBit(P, 2) && img.data[img_index + 1 - img.step])) {
            if (!father_offset) {
                father_offset = -(2 * (labels.step / labels.elem_size));
            }
            else {
                SetBit(info, Info::Q);
            }
        }

        // R square
        if (HasBit(P, 3) && img.data[img_index + 2 - img.step]) {
            if (!father_offset) {
                father_offset = -(2 * (labels.step / labels.elem_size) - 2);
            }
            else {
                SetBit(info, Info::R);
            }
        }

        // S square
        if ((HasBit(P, 4) && img.data[img_index - 1]) || (HasBit(P, 8) && img.data[img_index + img.step - 1])) {
            if (!father_offset) {
                father_offset = -2;
            }
            else {
                SetBit(info, Info::S);
            }
        }

        labels.data[labels_index] = labels_index + father_offset;
        if (col + 1 < labels.cols) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + 1);
        }
        else if (row + 1 < labels.rows) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + labels.step / labels.elem_size);
        }
        *last_pixel = info;
    }
}

__global__ void Merge(cuda::PtrStepSzi labels, unsigned char* last_pixel) {

    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        if (col + 1 < labels.cols) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + 1);
        }
        else if (row + 1 < labels.rows) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + labels.step / labels.elem_size);
        }
        unsigned char info = *last_pixel;

        if (HasBit(info, Info::Q)) {
            Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));
        }
        if (HasBit(info, Info::R)) {
            Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2);
        }
        if (HasBit(info, Info::S)) {
            Union(labels.data, labels_index, labels_index - 2);
        }
    }
}

__global__ void Compression(cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        FindAndCompress(labels.data, labels_index);
    }
}

__global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        int label;
        unsigned char info;
        unsigned long long buffer;

        if (col + 1 < labels.cols) {
            buffer = *reinterpret_cast<unsigned long long*>(labels.data + labels_index);
            label = (buffer & (0xFFFFFFFF)) + 1;
            info = (buffer >> 32) & 0xFFFFFFFF;
        }
        else {
            label = labels[labels_index] + 1;
            if (row + 1 < labels.rows) {
                info = labels[labels_index + labels.step / labels.elem_size];
            }
            else {
                // Read from the input image
                // "a" is already in position 0
                info = img[row * img.step + col];
            }
        }

        if (col + 1 < labels.cols) {
            *reinterpret_cast<unsigned long long*>(labels.data + labels_index) =
                (static_cast<unsigned long long>(HasBit(info, Info::b) * label) << 32) | (HasBit(info, Info::a) * label);

            if (row + 1 < labels.rows) {
                *reinterpret_cast<unsigned long long*>(labels.data + labels_index + labels.step / labels.elem_size) =
                    (static_cast<unsigned long long>(HasBit(info, Info::d) * label) << 32) | (HasBit(info, Info::c) * label);
            }
        }
        else {
            labels[labels_index] = HasBit(info, Info::a) * label;

            if (row + 1 < labels.rows) {
                labels[labels_index + (labels.step / labels.elem_size)] = HasBit(info, Info::c) * label;
            }
        }

    }

}

}

class BKE : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    unsigned char* last_pixel_;
    bool last_pixel_allocated_;

public:
    BKE() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);

        last_pixel_allocated_ = false;
        if ((d_img_.rows == 1 || d_img_.cols == 1) && !((d_img_.rows + d_img_.cols) % 2)) {
            cudaMalloc(&last_pixel_, sizeof(unsigned char));
            last_pixel_allocated_ = true;
        }
        else {
            last_pixel_ = d_img_labels_.data + ((d_img_labels_.rows - 2) * d_img_labels_.step) + (d_img_labels_.cols - 2) * d_img_labels_.elemSize();
        }

        grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_pixel_);

        //Mat1i init_blocks;
        //d_img_labels_.download(init_blocks);

        //cuda::GpuMat d_init_labels = d_img_labels_.clone();
        //FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_init_labels);
        //Mat1i init_labels;
        //d_init_labels.download(init_labels);
        //d_init_labels.release();

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        //Mat1i compr_blocks;
        //d_img_labels_.download(compr_blocks);

        //cuda::GpuMat d_compr_labels = d_img_labels_.clone();
        //FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_compr_labels);
        //Mat1i compr_labels;
        //d_compr_labels.download(compr_labels);
        //d_compr_labels.release();

        //cuda::GpuMat d_expanded_connections;
        //d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
        //ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
        //Mat1b expanded_connections;
        //d_expanded_connections.download(expanded_connections);
        //d_expanded_connections.release();

        Merge << <grid_size_, block_size_ >> > (d_img_labels_, last_pixel_);

        //Mat1i merge_blocks;
        //d_img_labels_.download(merge_blocks);		

        //cuda::GpuMat d_merge_labels = d_img_labels_.clone();
        //FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_merge_labels);
        //Mat1i merge_labels;
        //d_merge_labels.download(merge_labels);
        //d_merge_labels.release();

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        //Mat1i final_blocks;
        //d_img_labels_.download(final_blocks);

        FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //d_img_labels_.download(img_labels_);
        if (last_pixel_allocated_) {
            cudaFree(last_pixel_);
        }
        cudaDeviceSynchronize();
    }

    void PerformLabelingBlocksize(int x, int y, int z) override {

        d_img_labels_.create(d_img_.size(), CV_32SC1);

        last_pixel_allocated_ = false;
        if ((d_img_.rows == 1 || d_img_.cols == 1) && !((d_img_.rows + d_img_.cols) % 2)) {
            cudaMalloc(&last_pixel_, sizeof(unsigned char));
            last_pixel_allocated_ = true;
        }
        else {
            last_pixel_ = d_img_labels_.data + ((d_img_labels_.rows - 2) * d_img_labels_.step) + (d_img_labels_.cols - 2) * d_img_labels_.elemSize();
        }

        grid_size_ = dim3((((d_img_.cols + 1) / 2) + x - 1) / x, (((d_img_.rows + 1) / 2) + y - 1) / y, 1);
        block_size_ = dim3(x, y, 1);

        BLOCKSIZE_KERNEL(InitLabeling, grid_size_, block_size_, 0, d_img_, d_img_labels_, last_pixel_)

            BLOCKSIZE_KERNEL(Compression, grid_size_, block_size_, 0, d_img_labels_)

            BLOCKSIZE_KERNEL(Merge, grid_size_, block_size_, 0, d_img_labels_, last_pixel_)

            BLOCKSIZE_KERNEL(Compression, grid_size_, block_size_, 0, d_img_labels_)

            BLOCKSIZE_KERNEL(FinalLabeling, grid_size_, block_size_, 0, d_img_, d_img_labels_)

            if (last_pixel_allocated_) {
                cudaFree(last_pixel_);
            }
    }



private:
    double Alloc() {
        
        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        if ((d_img_.rows == 1 || d_img_.cols == 1) && !((d_img_.rows + d_img_.cols) % 2)) {
            cudaMalloc(&last_pixel_, sizeof(unsigned char));
            last_pixel_allocated_ = true;
        }
        else {
            last_pixel_ = d_img_labels_.data + ((d_img_labels_.rows - 2) * d_img_labels_.step) + (d_img_labels_.cols - 2) * d_img_labels_.elemSize();
        }        
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);
        if (last_pixel_allocated_) {
            cudaMemset(last_pixel_, 0, 1);
        }
        cudaDeviceSynchronize();
        double t = perf_.stop();

        perf_.start();
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);
        if (last_pixel_allocated_) {
            cudaMemset(last_pixel_, 0, 1);
        }
        cudaDeviceSynchronize();

        t -= perf_.stop();
        return t;
    }

    void Dealloc() {
        if (last_pixel_allocated_) {
            cudaFree(last_pixel_);
        }
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

        InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_pixel_);

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        //cuda::GpuMat d_expanded_connections;
        //d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
        //ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
        //Mat1b expanded_connections;
        //d_expanded_connections.download(expanded_connections);
        //d_expanded_connections.release();

        //Mat1i init_labels;
        //d_block_labels_.download(init_labels);

        Merge << <grid_size_, block_size_ >> > (d_img_labels_, last_pixel_);

        //Mat1i block_info_final;
        //d_img_labels_.download(block_info_final);		

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

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

REGISTER_LABELING(BKE)

REGISTER_KERNELS(BKE, InitLabeling, Compression, Merge, FinalLabeling)