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

#define BLOCK_SIZE 6
#define PATCH_SIZE (BLOCK_SIZE + 2)

using namespace cv;
using namespace std;

namespace {

    // This kernel makes use of a (BLOCK_SIZE + 2) X (BLOCK_SIZE + 2) array in shared memory 
    // The paper actually says (BLOCK_SIZE + 1) X (BLOCK_SIZE + 1), but I can't manage to make it work that way
    __global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

        const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        //const unsigned img_index = r * img.step + c;
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

            unsigned label = 0;
            if (img_patch[img_patch_index]) {
                label = labels_index + 1;

                unsigned int connections = 0;
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
                // Only half the connections are recorded
                //if (img_patch[img_patch_index + PATCH_SIZE + 1]) {
                //    connections |= (1 << 3);
                //}
                //if (img_patch[img_patch_index + PATCH_SIZE]) {
                //    connections |= (1 << 2);
                //}
                //if (img_patch[img_patch_index + PATCH_SIZE - 1]) {
                //    connections |= (1 << 1);
                //}
                //if (img_patch[img_patch_index - 1]) {
                //    connections |= (1 << 0);
                //}

                label |= connections;

                // It can't be done this way at compile time, because math functions are not defined constexpr
                // Macros could be used instead, and a technique for loop unrolling using templates is described here
                // https://stackoverflow.com/questions/15275023/clang-force-loop-unroll-for-specific-loop/15275701
                //constexpr double pi = 3.14159265358979323846;
                //for (double angle = 0; angle < 2 * pi; angle += pi / 4) {
                //	unsigned x = ceil(cos(angle));
                //	unsigned y = ceil(sin(angle));
                //}

            }
            labels[labels_index] = label;
        }
    }



    __global__ void Propagate(cuda::PtrStepSzi labels, char* changed) {

        const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
        const unsigned labels_patch_index = (threadIdx.y + 1) * PATCH_SIZE + threadIdx.x + 1;
        const unsigned local_linear_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

        __shared__ unsigned labels_patch[PATCH_SIZE * PATCH_SIZE];

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

        __syncthreads();

        const unsigned label = labels_patch[labels_patch_index];
        unsigned min_label = label & 0x0FFFFFFF;

        // Primary/Secondary Optimization
        // Find the primary pixel of the sub-component, and add its label to the propagation.
        if (min_label) {
            const int primary_r = ((label & 0x0FFFFFFF) - 1) / (labels.step / labels.elem_size);
            const int primary_c = ((label & 0x0FFFFFFF) - 1) % (labels.step / labels.elem_size);
            if (primary_r >= (blockIdx.y * BLOCK_SIZE - 1) && primary_r <= (blockIdx.y + 1) * BLOCK_SIZE &&
                primary_c >= (blockIdx.x * BLOCK_SIZE - 1) && primary_c <= (blockIdx.x + 1) * BLOCK_SIZE) {
                const int primary_local_r = primary_r - blockIdx.y * BLOCK_SIZE;
                const int primary_local_c = primary_c - blockIdx.x * BLOCK_SIZE;
                min_label = min(min_label, labels_patch[(primary_local_r + 1) * PATCH_SIZE + (primary_local_c + 1)] & 0x0FFFFFFF);
            }
        }

        __syncthreads();

        // Propagation sizes are calculated in every propagation step

        // Propagations could be shared by threads with shuffle ops, BUT
        // threads must be in the same warp. It requires a different thread 
        // organization for every direction, but it is feasible.
        // For now, shared memory is used instead.
        __shared__ unsigned propagation[BLOCK_SIZE * BLOCK_SIZE];

        // Up-Left
        int patch_r = threadIdx.y + 1;
        int patch_c = threadIdx.x + 1;
        unsigned cur_label = label;
        propagation[local_linear_index] = ((label >> 31) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x - propagation[local_linear_index];
                thread_y = threadIdx.y - propagation[local_linear_index];

                if (thread_x < 0 || thread_y < 0) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r - 1) * PATCH_SIZE + (patch_c - 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r - propagation[local_linear_index]) * PATCH_SIZE + (patch_c - propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }


        //while (((cur_label >> 31) & 1) && (--patch_r >= 0) && (--patch_c >= 0)) {
        //    cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    // This should go after the cycle, after the optimization of Pag. 209 has been applied
        //    // A propagation size of 1 must be mantained though
        //    min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //}


        // Up
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = label;

        __syncthreads();

        propagation[local_linear_index] = ((label >> 30) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_y = threadIdx.y - propagation[local_linear_index];

                if (thread_y < 0) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r - 1) * PATCH_SIZE + patch_c];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r - propagation[local_linear_index]) * PATCH_SIZE + patch_c];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }
        //patch_r = threadIdx.y + 1;
        //patch_c = threadIdx.x + 1;
        //cur_label = label;
        //while (((cur_label >> 31) & 1) && (--patch_r >= 0) && (--patch_c >= 0)) {
        //    cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    // This should go after the cycle, after the optimization of Pag. 209 has been applied
        //    // A propagation size of 1 must be mantained though
        //    min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //}

        //// Up
        //patch_r = threadIdx.y + 1;
        //patch_c = threadIdx.x + 1;
        //cur_label = label;
        //while (((cur_label >> 30) & 1) && (--patch_r >= 0)) {
        //    cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //}

        // Up-Right
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = label;

        __syncthreads();

        propagation[local_linear_index] = ((label >> 29) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x + propagation[local_linear_index];
                thread_y = threadIdx.y - propagation[local_linear_index];

                if (thread_x >= BLOCK_SIZE || thread_y < 0) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r - 1) * PATCH_SIZE + (patch_c + 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r - propagation[local_linear_index]) * PATCH_SIZE + (patch_c + propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //patch_r = threadIdx.y + 1;
        //patch_c = threadIdx.x + 1;
        //cur_label = label;
        //while (((cur_label >> 29) & 1) && (--patch_r >= 0) && (++patch_c < PATCH_SIZE)) {
        //    cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //}

        // Right
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = label;

        __syncthreads();

        propagation[local_linear_index] = ((label >> 28) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x + propagation[local_linear_index];

                if (thread_x >= BLOCK_SIZE) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[patch_r * PATCH_SIZE + (patch_c + 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[patch_r * PATCH_SIZE + (patch_c + propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //patch_r = threadIdx.y + 1;
        //patch_c = threadIdx.x + 1;
        //cur_label = label;
        //while (((cur_label >> 28) & 1) && (++patch_c < PATCH_SIZE)) {
        //    cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //}

        // The next 4 connection bits come from neighbor pixels

        // Down-Right
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = labels_patch[(patch_r + 1) * PATCH_SIZE + (patch_c + 1)];

        __syncthreads();

        propagation[local_linear_index] = ((cur_label >> 31) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x + propagation[local_linear_index];
                thread_y = threadIdx.y + propagation[local_linear_index];

                if (thread_x >= BLOCK_SIZE || thread_y >= BLOCK_SIZE) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r + 1) * PATCH_SIZE + (patch_c + 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r + propagation[local_linear_index]) * PATCH_SIZE + (patch_c + propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //while (true) {
        //    if ((cur_label >> 31) & 1) {
        //        min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //    }
        //    else {
        //        break;
        //    }
        //    patch_r++;
        //    patch_c++;
        //    if ((patch_r < PATCH_SIZE) && (patch_c < PATCH_SIZE)) {
        //        cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    }
        //    else {
        //        break;
        //    }
        //}

        // Down
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = labels_patch[(patch_r + 1) * PATCH_SIZE + patch_c];

        __syncthreads();

        propagation[local_linear_index] = ((cur_label >> 30) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_y = threadIdx.y + propagation[local_linear_index];

                if (thread_y >= BLOCK_SIZE) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r + 1) * PATCH_SIZE + patch_c];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r + propagation[local_linear_index]) * PATCH_SIZE + patch_c];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //patch_r = threadIdx.y + 2;
        //patch_c = threadIdx.x + 1;
        //cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //while (true) {
        //    if ((cur_label >> 30) & 1) {
        //        min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //    }
        //    else {
        //        break;
        //    }
        //    patch_r++;
        //    if ((patch_r < PATCH_SIZE)) {
        //        cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    }
        //    else {
        //        break;
        //    }
        //}

        // Down-Left
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = labels_patch[(patch_r + 1) * PATCH_SIZE + (patch_c - 1)];

        __syncthreads();

        propagation[local_linear_index] = ((cur_label >> 29) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x - propagation[local_linear_index];
                thread_y = threadIdx.y + propagation[local_linear_index];

                if (thread_x < 0 || thread_y >= BLOCK_SIZE) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[(patch_r + 1) * PATCH_SIZE + (patch_c - 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[(patch_r + propagation[local_linear_index]) * PATCH_SIZE + (patch_c - propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //patch_r = threadIdx.y + 2;
        //patch_c = threadIdx.x;
        //cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //while (true) {
        //    if ((cur_label >> 29) & 1) {
        //        min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //    }
        //    else {
        //        break;
        //    }
        //    patch_r++;
        //    patch_c--;
        //    if ((patch_r < PATCH_SIZE) && (patch_c >= 0)) {
        //        cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    }
        //    else {
        //        break;
        //    }
        //}

        // Left
        patch_r = threadIdx.y + 1;
        patch_c = threadIdx.x + 1;
        cur_label = labels_patch[patch_r * PATCH_SIZE + (patch_c - 1)];

        __syncthreads();

        propagation[local_linear_index] = ((cur_label >> 28) & 1);

        __syncthreads();

        if (propagation[local_linear_index]) {

            int thread_x = threadIdx.x;
            int thread_y = threadIdx.y;

            while (true) {

                thread_x = threadIdx.x - propagation[local_linear_index];

                if (thread_x < 0) {
                    break;
                }

                unsigned prop_delta = propagation[thread_y * BLOCK_SIZE + thread_x];

                if (prop_delta == 0) {
                    break;
                }

                propagation[local_linear_index] += prop_delta;
            }

            // A propagation size of 1 must be mantained
            const unsigned close_label = labels_patch[patch_r * PATCH_SIZE + (patch_c - 1)];
            min_label = min(min_label, close_label & 0x0FFFFFFF);
            // The farthest label is gathered
            const unsigned far_label = labels_patch[patch_r * PATCH_SIZE + (patch_c - propagation[local_linear_index])];
            min_label = min(min_label, far_label & 0x0FFFFFFF);
        }

        //patch_r = threadIdx.y + 1;
        //patch_c = threadIdx.x;
        //cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //while (true) {
        //    if ((cur_label >> 28) & 1) {
        //        min_label = min(min_label, cur_label & 0x0FFFFFFF);
        //    }
        //    else {
        //        break;
        //    }
        //    patch_c--;
        //    if (patch_c >= 0) {
        //        cur_label = labels_patch[patch_r * PATCH_SIZE + patch_c];
        //    }
        //    else {
        //        break;
        //    }
        //}


        if (min_label < (label & 0x0FFFFFFF)) {
            labels_patch[labels_patch_index] = min_label | (label & 0xF0000000);
            *changed = 1;
        }

        __syncthreads();

        if (in_limits) {

            labels[labels_index] = labels_patch[labels_patch_index];

        }
        // A propagation cycle could be added inside the thread block
    }


    __global__ void End(cuda::PtrStepSzi labels) {

        unsigned global_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        unsigned global_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

        if (global_row < labels.rows && global_col < labels.cols) {
            *(reinterpret_cast<unsigned char*>(labels.data + labels_index) + 3) &= 0x0F;   // Assuming little endian
        }
    }

}

class RASMUSSON : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;

public:
    RASMUSSON() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_img_.rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        block_size_ = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

        char changed = 1;
        char* d_changed_ptr;
        cudaMalloc(&d_changed_ptr, 1);

        // Phase 1
        // CCL on tiles		
        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        // Immagine di debug della prima fase
        cuda::GpuMat d_local_labels;
        d_img_labels_.copyTo(d_local_labels);
        // PathCompression << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        // ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
        Mat1i local_labels(img_.size());
        d_local_labels.download(local_labels);

        // Phase 2
        while (changed) {
            // changed = 0;
            // cudaMemcpy(d_changed_ptr, &changed, 1, cudaMemcpyDefault);

            cudaMemset(d_changed_ptr, 0, 1);

            Propagate << <grid_size_, block_size_ >> > (d_img_labels_, d_changed_ptr);
            //Propagate << <dim3(1, 1, 1), block_size_ >> > (d_img_labels_, d_changed_ptr);

            cudaMemcpy(&changed, d_changed_ptr, 1, cudaMemcpyDeviceToHost);

            cuda::GpuMat d_global_labels;
            d_img_labels_.copyTo(d_global_labels);
            End << <grid_size_, block_size_ >> > (d_global_labels);
            //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
            // ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
            Mat1i global_labels(img_.size());
            d_global_labels.download(global_labels);
        }
        // Merges UFTrees of different tiles

        // Immagine di debug della seconda fase
        //cuda::GpuMat d_global_labels;
        //d_img_labels_.copyTo(d_global_labels);
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
        //Mat1i global_labels(img_.size());
        //d_global_labels.download(global_labels);

        // Phase 3
        // Collapse UFTrees
        //PathCompression << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        End << <grid_size_, block_size_ >> > (d_img_labels_);

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

    }

    void GlobalScan() {

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

REGISTER_LABELING(RASMUSSON);

