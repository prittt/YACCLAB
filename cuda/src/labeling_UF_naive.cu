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


// Simplified version of UF that doesn't make use of the Tile Merging technique.
// The initial phase which performs labeling inside tiles is avoided.
// This variation performs worse than the original one which uses Tiles Merging.


#define BLOCK_X 16
#define BLOCK_Y 16

using namespace cv;

namespace {

    // Returns the root index of the UFTree
    __device__ unsigned Find(const int *s_buf, unsigned n) {
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


    // Merges the UFTrees of a and b, linking one root to the other
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

    __global__ void Initialization(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned img_index = y * (img.step / img.elem_size) + x;
        unsigned labels_index = y * (labels.step / labels.elem_size) + x;

        if (x < labels.cols && y < labels.rows) {
            if (img[img_index]) {
                labels[labels_index] = labels_index + 1;
            }
            else {
                labels[labels_index] = 0;
            }
        }
    }


    __global__ void Merge(cuda::PtrStepSzi labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned labels_index = y * (labels.step / labels.elem_size) + x;

        if (x < labels.cols && y < labels.rows) {
            if (labels[labels_index]) {            

                if (y > 0) {
                    if (x > 0 && labels.data[labels_index - (labels.step / labels.elem_size) - 1]) {
                        Union(labels.data, labels_index, labels_index - (labels.step / labels.elem_size) - 1);
                    }
                    if (labels.data[labels_index - (labels.step / labels.elem_size)]) {
                        Union(labels.data, labels_index, labels_index - (labels.step / labels.elem_size));
                    }
                    if (x + 1 < labels.cols && labels.data[labels_index - (labels.step / labels.elem_size) + 1]) {
                        Union(labels.data, labels_index, labels_index - (labels.step / labels.elem_size) + 1);
                    }
                }

                if (x > 0 && labels.data[labels_index - 1]) {
                    Union(labels.data, labels_index, labels_index - 1);
                }
                
            }
        }
    }


    __global__ void PathCompression(cuda::PtrStepSzi labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned labels_index = y * (labels.step / labels.elem_size) + x;

        if (x < labels.cols && y < labels.rows) {
            unsigned int val = labels[labels_index];
            if (val) {
                labels[labels_index] = Find(labels.data, labels_index) + 1;
            }
        }
    }

}


class UF_naive : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    UF_naive() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.rows, d_img_.cols, CV_32SC1);

        grid_size_ = dim3((d_img_.cols + BLOCK_X - 1) / BLOCK_X, (d_img_.rows + BLOCK_Y - 1) / BLOCK_Y, 1);
        block_size_ = dim3(BLOCK_X, BLOCK_Y, 1);

        //cuda::PtrStep3b ptr_step_prima(d_img_labels_);

        // Phase 1
        Initialization << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        //cuda::PtrStepSz3i ptr_step_size(d_img_labels_);

        // Immagine di debug della prima fase
        //cuda::GpuMat d_local_labels;
        //d_img_labels_.copyTo(d_local_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_local_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        //Mat1i local_labels(img_.size());
        //d_local_labels.download(local_labels);

        // Phase 2
        Merge << <grid_size_, block_size_ >> > (d_img_labels_);

        // Immagine di debug della seconda fase
        //cuda::GpuMat d_global_labels;
        //d_img_labels_.copyTo(d_global_labels);
        //// PathCompression << <grid_size_, block_size_ >> > (d_global_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //Mat1i global_labels(img_.size());
        //d_global_labels.download(global_labels);

        // Phase 3
        PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);

        // d_img_labels_.download(img_labels_);

        cudaDeviceSynchronize();
    }

};

REGISTER_LABELING(UF_naive);
