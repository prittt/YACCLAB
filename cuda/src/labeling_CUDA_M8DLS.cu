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


// Modification of 8DLS (8 Directional Label equivalence Solution). Le change with respect to 8DLS is not clear.
// Apparently, the result is not always correct.
// We e-mailed the authors, but they wouldn't answer.


// Modifica dell' 8DLS (8 Directional Label equivalence Solution). La modifica proposta rispetto a DLS non � chiara, da quello che 
// abbiamo capito dal paper non sempre porta alla soluzione corretta. Abbiamo scritto una mail agli autori, attendiamo risposta.


#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {

    // Init phase.
    // Labels start at value 1, to differentiate them from background, that has value 0.
    __global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        unsigned global_row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
        unsigned global_col = blockIdx.x * BLOCK_COLS + threadIdx.x;
        unsigned img_index = global_row * img.step + global_col;
        unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

        if (global_row < img.rows && global_col < img.cols) {
            labels[labels_index] = img[img_index] ? (labels_index + 1) : 0;
        }
    }


    __global__ void Scan(cuda::PtrStepSzi labels, char *changes, unsigned it) {

        unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
        unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
        unsigned labels_index = row * (labels.step / labels.elem_size) + col;

        if (row < labels.rows && col < labels.cols) {

            unsigned label = labels[labels_index];

            if (label && (it < 2 || /*label != labels_index - 1 ||*/ labels[label - 1] != label)) {

                // Scan the 8 directions looking for the smallest label
                unsigned min_label = UINT_MAX;

                // Up
                for (unsigned i = 1; i <= row; i++) {
                    unsigned tmp = labels[labels_index - i * labels.step / labels.elem_size];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }               
                }

                // Down
                for (unsigned i = 1; row + i < labels.rows; i++) {
                    unsigned tmp = labels[labels_index + i * labels.step / labels.elem_size];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                // Left          
                for (unsigned i = 1; col >= i; i++) {
                    unsigned tmp = labels[labels_index - i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                //Right
                for (unsigned i = 1; col + i < labels.cols; i++) {
                    unsigned tmp = labels[labels_index + i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                // Up-Left
                for (unsigned i = 1; i <= row && col >= i; i++) {
                    unsigned tmp = labels[labels_index - i * labels.step / labels.elem_size - i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                // Up-Right
                for (unsigned i = 1; i <= row && col + i < labels.cols; i++) {
                    unsigned tmp = labels[labels_index - i * labels.step / labels.elem_size + i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                // Down-Left
                for (unsigned i = 1; row + i < labels.rows && col >= i; i++) {
                    unsigned tmp = labels[labels_index + i * labels.step / labels.elem_size - i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                // Down-Right
                for (unsigned i = 1; row + i < labels.rows && col + i < labels.cols; i++) {
                    unsigned tmp = labels[labels_index + i * labels.step / labels.elem_size + i];
                    if (tmp == 0) {
                        break;
                    }
                    if (tmp < min_label) {
                        min_label = tmp;
                    }
                }

                if (min_label < label) {
                    labels[labels_index] = min_label;
                    *changes = 1;
                }
              
            }
        }
    }

}

class M8DLS : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    char changes;
    char *d_changes;

public:
    M8DLS() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMalloc(&d_changes, sizeof(char));

        grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        unsigned it = 0;
        while (true) {
            changes = 0;
            cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

            Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes, it);

            d_img_labels_.download(img_labels_);

            cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

            if (!changes)
                break;

            it++;
        }
        cudaFree(d_changes);
        cudaDeviceSynchronize();
    }


private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMalloc(&d_changes, sizeof(char));
        perf_.stop();
        return perf_.last();
    }

    double Dealloc() {
        perf_.start();
        cudaFree(d_changes);
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

        char changes = 1;
        char *d_changes;
        cudaMalloc(&d_changes, sizeof(char));
        // cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        unsigned it = 1;
        while (true) {
            changes = 0;
            cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

            Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes, it);

            cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

            if (!changes)
                break;

            it++;
        }
        cudaFree(d_changes);
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

REGISTER_LABELING(M8DLS);

