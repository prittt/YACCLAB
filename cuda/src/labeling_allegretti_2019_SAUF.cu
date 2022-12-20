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

// SAUF in GPU

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {

__device__ unsigned Find(const int* s_buf, unsigned n) {

    unsigned label = s_buf[n];

    assert(label > 0);

    while (label - 1 != n) {
        n = label - 1;
        label = s_buf[n];

        assert(label > 0);
    }

    return n;

}


__device__ void Union(int* s_buf, unsigned a, unsigned b) {

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



__global__ void Initialization(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {
    unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    unsigned img_index = row * (img.step / img.elem_size) + col;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        if (img[img_index] > 0) {
            labels[labels_index] = labels_index + 1;
        }
        else {
            labels[labels_index] = 0;
        }
    }
}


__global__ void Merge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    unsigned img_index = row * img.step + col;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

#define CONDITION_P col > 0 && row > 0 && img[img_index - img.step - 1] > 0
#define CONDITION_Q row > 0 && img[img_index - img.step] > 0
#define CONDITION_R col < img.cols - 1 && row > 0 && img[img_index - img.step + 1] > 0
#define CONDITION_S col > 0 && img[img_index - 1] > 0
#define CONDITION_X img[img_index] > 0

#define ACTION_1 // nothing to do 
#define ACTION_2 // LabelsSolver::NewLabel(); // new label
#define ACTION_3 Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size - 1);   //img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size);       //img_labels_row_prev[c]; // x <- q
#define ACTION_5 Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size + 1);   //img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 Union(labels.data, labels_index, labels_index - 1);                                    //img_labels_row[c - 1]; // x <- s
#define ACTION_7 Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size - 1);  \
                    Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size + 1);    //LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 Union(labels.data, labels_index, labels_index - 1);  \
                    Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size + 1);    //LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#include "labeling_wu_2009_tree.inc.h"

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X

    }
}


__global__ void Compression(cuda::PtrStepSzi labels) {

    unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        unsigned int val = labels[labels_index];
        if (val > 0) {
            labels[labels_index] = Find(labels.data, labels_index) + 1;
        }
    }
}

}

class C_SAUF : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    C_SAUF() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);


        Initialization << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);


        Merge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //Mat1i local_labels(img_.size());
        //d_img_labels_.download(local_labels);

        //cuda::GpuMat d_global_labels;
        //d_img_labels_.copyTo(d_global_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //Mat1i global_labels(img_.size());
        //d_global_labels.download(global_labels);

        Compression << <grid_size_, block_size_ >> > (d_img_labels_);

        //d_img_labels_.download(img_labels_);

        cudaDeviceSynchronize();
    }

    void PerformLabelingBlocksize(int x, int y, int z) {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + x - 1) / x, (d_img_.rows + y - 1) / y, 1);
        block_size_ = dim3(x, y, 1);

        BLOCKSIZE_KERNEL(Initialization, grid_size_, block_size_, 0, d_img_, d_img_labels_)

            BLOCKSIZE_KERNEL(Merge, grid_size_, block_size_, 0, d_img_, d_img_labels_)

            BLOCKSIZE_KERNEL(Compression, grid_size_, block_size_, 0, d_img_labels_)

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
        grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);
        Initialization << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);
        Merge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);
        Compression << <grid_size_, block_size_ >> > (d_img_labels_);
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

REGISTER_LABELING(C_SAUF);

REGISTER_KERNELS(C_SAUF, Initialization, Merge, Compression)