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

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4


using namespace cv;


// Algorithm itself has good performances, but memory allocation is a problem.
// I will try to reduce it.
namespace {

    // Only use it with unsigned numeric types
    template <typename T>
    __device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
        return (bitmap >> pos) & 1;
    }

    // Only use it with unsigned numeric types
    template <typename T>
    __device__ __forceinline__ void SetBit(T &bitmap, unsigned char pos) {
        bitmap |= (1 << pos);
    }

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


    // Init phase.
    // Labels start at value 1.
    __global__ void Init(const cuda::PtrStepSz3b img, cuda::PtrStepSz3i block_conn, cuda::PtrStepSz3i block_labels) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned img_index = 2*z * (img.stepz / img.elem_size) + 2*y * (img.stepy / img.elem_size) + 2*x;
        unsigned conn_index = z * (block_conn.stepz / block_conn.elem_size) + y * (block_conn.stepy / block_conn.elem_size) + x;
        unsigned labels_index = z * (block_labels.stepz / block_labels.elem_size) + y * (block_labels.stepy / block_labels.elem_size) + x;

        if (x < block_conn.x && y < block_conn.y && z < block_conn.z) {

#define P0 0x77707770777UL

            unsigned long long P = 0UL;

            if (img[img_index]) {
                P |= P0;
            }

            if (2 * x + 1 < img.x) {

                if (img[img_index + 1]) {
                    P |= (P0 << 1);
                }

                if (2 * y + 1 < img.y && img[img_index + img.stepy / img.elem_size + 1]) {
                    P |= (P0 << 5);
                }

            }

            if (2 * y + 1 < img.y) {

                if (img[img_index + img.stepy / img.elem_size]) {
                    P |= (P0 << 4);
                }

            }

            if (2 * z + 1 < img.z) {
                if (img[img_index + img.stepz / img.elem_size]) {
                    P |= P0 << 16;
                }

                if (2 * x + 1 < img.x) {

                    if (img[img_index + img.stepz / img.elem_size + 1]) {
                        P |= (P0 << 17);
                    }

                    if (2 * y + 1 < img.y && img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size + 1]) {
                        P |= (P0 << 21);
                    }

                }

                if (2 * y + 1 < img.y) {

                    if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size]) {
                        P |= (P0 << 20);
                    }

                }
            }

#undef P0

            // checks on borders

            if (x == 0) {
                P &= 0xEEEEEEEEEEEEEEEE;
            }
            if (2 * x + 1 >= img.x) {
                P &= 0x3333333333333333;
            }
            else if (2 * x + 2 >= img.x) {
                P &= 0x7777777777777777;
            }

            if (y == 0) {
                P &= 0xFFF0FFF0FFF0FFF0;
            }
            if (2 * y + 1 >= img.y) {
                P &= 0x00FF00FF00FF00FF;
            }
            else if (2 * y + 2 >= img.y) {
                P &= 0x0FFF0FFF0FFF0FFF;
            }

            if (z == 0) {
                P &= 0xFFFFFFFFFFFF0000;
            }
            if (2 * z + 1 >= img.z) {
                P &= 0x00000000FFFFFFFF;
            }
            else if (2 * z + 2 >= img.z) {
                P &= 0x0000FFFFFFFFFFFF;
            }

            // P is now ready to be used to find neighbour blocks (or it should be)
            // P value avoids range errors

            unsigned int conn_bitmask = 0;

            if (P > 0) {

                block_labels[labels_index] = labels_index + 1;

                // Lower plane
                unsigned char * plane_data = img.data + img_index - (img.stepz / img.elem_size);
                        
                if (HasBit(P, 0) && plane_data[0 - img.stepy - 1]) {
                    SetBit(conn_bitmask, 0);
                }

                if ((HasBit(P, 1) && plane_data[0 - img.stepy]) || (HasBit(P, 2) && plane_data[0 - img.stepy + 1])) {
                    SetBit(conn_bitmask, 1);
                }

                if (HasBit(P, 3) && plane_data[0 - img.stepy + 2]) {
                    SetBit(conn_bitmask, 2);
                }

                if ((HasBit(P, 4) && plane_data[- 1]) || (HasBit(P, 8) && plane_data[img.stepy - 1])) {
                    SetBit(conn_bitmask, 3);
                }

                if ((HasBit(P, 5) && plane_data[0]) || (HasBit(P, 6) && plane_data[1]) || (HasBit(P, 9) && plane_data[img.stepy]) || (HasBit(P, 10) && plane_data[img.stepy + 1])) {
                    SetBit(conn_bitmask, 4);
                }

                if ((HasBit(P, 7) && plane_data[2]) || (HasBit(P, 11) && plane_data[img.stepy + 2])) {
                    SetBit(conn_bitmask, 5);
                }

                if (HasBit(P, 12) && plane_data[2 * img.stepy - 1]) {
                    SetBit(conn_bitmask, 6);
                }

                if ((HasBit(P, 13) && plane_data[2 * img.stepy]) || (HasBit(P, 14) && plane_data[2 * img.stepy + 1])) {
                    SetBit(conn_bitmask, 7);
                }

                if (HasBit(P, 15) && plane_data[2 * img.stepy + 2]) {
                    SetBit(conn_bitmask, 8);
                }

                // Current planes
                plane_data += img.stepz / img.elem_size;

                if ((HasBit(P, 16) && plane_data[0 - img.stepy - 1]) || (HasBit(P, 32) && plane_data[img.stepz - img.stepy - 1])) {
                    SetBit(conn_bitmask, 9);
                }

                if ((HasBit(P, 17) && plane_data[0 - img.stepy]) || (HasBit(P, 18) && plane_data[0 - img.stepy + 1]) || (HasBit(P, 33) && plane_data[img.stepz - img.stepy]) || (HasBit(P, 34) && plane_data[img.stepz - img.stepy + 1])) {
                    SetBit(conn_bitmask, 10);
                }

                if ((HasBit(P, 19) && plane_data[0 - img.stepy + 2]) || (HasBit(P, 35) && plane_data[img.stepz - img.stepy + 2])) {
                    SetBit(conn_bitmask, 11);
                }

                if ((HasBit(P, 20) && plane_data[-1]) || (HasBit(P, 24) && plane_data[img.stepy - 1]) || (HasBit(P, 36) && plane_data[img.stepz - 1]) || (HasBit(P, 40) && plane_data[img.stepz + img.stepy - 1])) {
                    SetBit(conn_bitmask, 12);
                }

                if ((HasBit(P, 23) && plane_data[2]) || (HasBit(P, 27) && plane_data[img.stepy + 2]) || (HasBit(P, 39) && plane_data[img.stepz + 2]) || (HasBit(P, 43) && plane_data[img.stepz + img.stepy + 2])) {
                    SetBit(conn_bitmask, 14);
                }

                if ((HasBit(P, 28) && plane_data[2 * img.stepy - 1]) || (HasBit(P, 44) && plane_data[img.stepz + 2 * img.stepy - 1])) {
                    SetBit(conn_bitmask, 15);
                }

                if ((HasBit(P, 29) && plane_data[2 * img.stepy]) || (HasBit(P, 30) && plane_data[2 * img.stepy + 1]) || (HasBit(P, 45) && plane_data[img.stepz + 2 * img.stepy]) || (HasBit(P, 46) && plane_data[img.stepz + 2 * img.stepy + 1])) {
                    SetBit(conn_bitmask, 16);
                }

                if ((HasBit(P, 31) && plane_data[2 * img.stepy + 2]) || (HasBit(P, 47) && plane_data[img.stepz + 2 * img.stepy + 2])) {
                    SetBit(conn_bitmask, 17);
                }

                // Upper plane
                plane_data += 2 * (img.stepz / img.elem_size);

                if (HasBit(P, 48) && plane_data[0 - img.stepy - 1]) {
                    SetBit(conn_bitmask, 18);
                }

                if ((HasBit(P, 49) && plane_data[0 - img.stepy]) || (HasBit(P, 50) && plane_data[0 - img.stepy + 1])) {
                    SetBit(conn_bitmask, 19);
                }

                if (HasBit(P, 51) && plane_data[0 - img.stepy + 2]) {
                    SetBit(conn_bitmask, 20);
                }

                if ((HasBit(P, 52) && plane_data[-1]) || (HasBit(P, 56) && plane_data[img.stepy - 1])) {
                    SetBit(conn_bitmask, 21);
                }

                if ((HasBit(P, 53) && plane_data[0]) || (HasBit(P, 54) && plane_data[1]) || (HasBit(P, 57) && plane_data[img.stepy]) || (HasBit(P, 58) && plane_data[img.stepy + 1])) {
                    SetBit(conn_bitmask, 22);
                }

                if ((HasBit(P, 55) && plane_data[2]) || (HasBit(P, 59) && plane_data[img.stepy + 2])) {
                    SetBit(conn_bitmask, 23);
                }

                if (HasBit(P, 60) && plane_data[2 * img.stepy - 1]) {
                    SetBit(conn_bitmask, 24);
                }

                if ((HasBit(P, 61) && plane_data[2 * img.stepy]) || (HasBit(P, 62) && plane_data[2 * img.stepy + 1])) {
                    SetBit(conn_bitmask, 25);
                }

                if (HasBit(P, 63) && plane_data[2 * img.stepy + 2]) {
                    SetBit(conn_bitmask, 26);
                }
            }

            else {
                block_labels[labels_index] = 0;
            }

            block_conn[conn_index] = conn_bitmask;
        }
    }


    //__global__ void ExpandConnections(const cuda::PtrStepSzb connections, cuda::PtrStepSzb expansion) {
    //    unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    //    unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    //    unsigned conn_index = row * (connections.step / connections.elem_size) + col;
    //    unsigned exp_index = 3 * row * (expansion.step / expansion.elem_size) + 3 * col;
    //    if (row < connections.rows && col < connections.cols) {
    //        expansion[exp_index + (expansion.step / expansion.elem_size) + 1] = 2;
    //        unsigned char neighbours = connections[conn_index];
    //        if (HasBit(neighbours, 0)) {
    //            expansion[exp_index] = 1;
    //        }
    //        else {
    //            expansion[exp_index] = 0;
    //        }
    //        if (HasBit(neighbours, 1)) {
    //            expansion[exp_index + 1] = 1;
    //        }
    //        else {
    //            expansion[exp_index + 1] = 0;
    //        }
    //        if (HasBit(neighbours, 2)) {
    //            expansion[exp_index + 2] = 1;
    //        }
    //        else {
    //            expansion[exp_index + 2] = 0;
    //        }
    //        if (HasBit(neighbours, 3)) {
    //            expansion[exp_index + (expansion.step / expansion.elem_size)] = 1;
    //        }
    //        else {
    //            expansion[exp_index + (expansion.step / expansion.elem_size)] = 0;
    //        }
    //        if (HasBit(neighbours, 4)) {
    //            expansion[exp_index + (expansion.step / expansion.elem_size) + 2] = 1;
    //        }
    //        else {
    //            expansion[exp_index + (expansion.step / expansion.elem_size) + 2] = 0;
    //        }
    //        if (HasBit(neighbours, 5)) {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size)] = 1;
    //        }
    //        else {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size)] = 0;
    //        }
    //        if (HasBit(neighbours, 6)) {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 1] = 1;
    //        }
    //        else {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 1] = 0;
    //        }
    //        if (HasBit(neighbours, 7)) {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 2] = 1;
    //        }
    //        else {
    //            expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 2] = 0;
    //        }
    //    }
    //}


    __device__ unsigned int MinLabel(unsigned l1, unsigned l2) {
        if (l1 && l2)
            return min(l1, l2);
        else
            return l1;
    }


    __device__ unsigned int FindMinLabel(cuda::PtrStepSz3i labels, unsigned int neighbours, unsigned label, unsigned labels_index) {

        unsigned int min = label;

        for (char plane = -1; plane <= 1; plane++) {
            int * plane_data = labels.data + labels_index + plane * (labels.stepz / labels.elem_size);

            if (HasBit(neighbours, 0)) {
                min = MinLabel(min, plane_data[0 - (labels.stepy / labels.elem_size) - 1]);
            }

            if (HasBit(neighbours, 1)) {
                min = MinLabel(min, plane_data[0 - (labels.stepy / labels.elem_size)]);
            }

            if (HasBit(neighbours, 2)) {
                min = MinLabel(min, plane_data[0 - (labels.stepy / labels.elem_size) + 1]);
            }

            if (HasBit(neighbours, 3)) {
                min = MinLabel(min, plane_data[-1]);
            }

            if (plane && HasBit(neighbours, 4)) {
                min = MinLabel(min, plane_data[0]);
            }

            if (HasBit(neighbours, 5)) {
                min = MinLabel(min, plane_data[1]);
            }

            if (HasBit(neighbours, 6)) {
                min = MinLabel(min, plane_data[(labels.stepy / labels.elem_size) - 1]);
            }

            if (HasBit(neighbours, 7)) {
                min = MinLabel(min, plane_data[(labels.stepy / labels.elem_size)]);
            }

            if (HasBit(neighbours, 8)) {
                min = MinLabel(min, plane_data[(labels.stepy / labels.elem_size) + 1]);
            }

            neighbours >>= 9;
        }

        return min;
    }


    // Scan phase.
    // The pixel associated with current thread is given the minimum label of the neighbours.
    __global__ void Scan(cuda::PtrStepSz3i labels, cuda::PtrStepSz3i connections, char *changes) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned conn_index = z * (connections.stepz / connections.elem_size) + y * (connections.stepy / connections.elem_size) + x;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {

            unsigned int neighbours = connections[conn_index];

            unsigned label = labels[labels_index];

            if (label) {
                unsigned min_label = FindMinLabel(labels, neighbours, label, labels_index);
                if (min_label < label) {
                    labels[label - 1] = min(static_cast<unsigned int>(labels[label - 1]), min_label);
                    *changes = 1;
                }
            }
        }
    }


    __global__ void Analyze(cuda::PtrStepSz3i labels) {

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

    // Final Labeling phase
    // Assigns every pixel of 2x2x2 blocks the block label
    __global__ void FinalLabeling(cuda::PtrStepSz3i block_labels, cuda::PtrStepSz3i labels, const cuda::PtrStepSz3b img) {

        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        unsigned z = blockIdx.z * BLOCK_Z + threadIdx.z;
        unsigned blocks_index = z * (block_labels.stepz / block_labels.elem_size) + y * (block_labels.stepy / block_labels.elem_size) + x;
        unsigned labels_index = 2 * z * (labels.stepz / labels.elem_size) + 2 * y * (labels.stepy / labels.elem_size) + 2 * x;
        unsigned img_index = 2 * z * (img.stepz / img.elem_size) + 2 * y * (img.stepy / img.elem_size) + 2 * x;

        if (x < block_labels.x && y < block_labels.y && z < block_labels.z) {

            unsigned int label = block_labels[blocks_index];

            // Current plane
            if (img[img_index]) {
                labels[labels_index] = label;
            }
            else {
                labels[labels_index] = 0;
            }

            if (2 * x + 1 < labels.x) {
                if (img[img_index + 1])
                    labels[labels_index + 1] = label;
                else {
                    labels[labels_index + 1] = 0;
                }

                if (2 * y + 1 < labels.y) {
                    if (img[img_index + img.stepy + 1])
                        labels[labels_index + (labels.stepy / labels.elem_size) + 1] = label;
                    else {
                        labels[labels_index + (labels.stepy / labels.elem_size) + 1] = 0;
                    }
                }
            }

            if (2 * y + 1 < labels.y) {
                if (img[img_index + img.stepy])
                    labels[labels_index + (labels.stepy / labels.elem_size)] = label;
                else {
                    labels[labels_index + (labels.stepy / labels.elem_size)] = 0;
                }
            }

            // Upper plane
            if (2 * z + 1 < labels.z) {

                if (img[img_index + img.stepz / img.elem_size])
                    labels[labels_index + labels.stepz / labels.elem_size] = label;
                else {
                    labels[labels_index + labels.stepz / labels.elem_size] = 0;
                }

                if (2 * x + 1 < labels.x) {
                    if (img[img_index + img.stepz / img.elem_size + 1])
                        labels[labels_index + labels.stepz / labels.elem_size + 1] = label;
                    else {
                        labels[labels_index + labels.stepz / labels.elem_size + 1] = 0;
                    }

                    if (2 * y + 1 < labels.y) {
                        if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size + 1])
                            labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size) + 1] = label;
                        else {
                            labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size) + 1] = 0;
                        }
                    }
                }

                if (2 * y + 1 < labels.y) {
                    if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size])
                        labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)] = label;
                    else {
                        labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)] = 0;
                    }
                }

            }

        }

    }

}


class BE_3D : public GpuLabeling3D<Connectivity3D::CONN_26> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    char changes;
    char *d_changes;

    cuda::GpuMat3 d_connections_;
    cuda::GpuMat3 d_block_labels_;

public:
    BE_3D() {}

    void PerformLabeling() {

        d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);

        // Extra structures that I would gladly do without
        d_connections_.create((d_img_.x + 1) / 2, (d_img_.y + 1) / 2, (d_img_.z + 1) / 2, CV_32SC1);
        d_block_labels_.create((d_img_.x + 1) / 2, (d_img_.y + 1) / 2, (d_img_.z + 1) / 2, CV_32SC1);

        grid_size_ = dim3((d_block_labels_.x + BLOCK_X - 1) / BLOCK_X, (d_block_labels_.y + BLOCK_Y - 1) / BLOCK_Y, (d_block_labels_.z + BLOCK_Z - 1) / BLOCK_Z);
        block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

        cudaMalloc(&d_changes, sizeof(char));

        Init << <grid_size_, block_size_ >> > (d_img_, d_connections_, d_block_labels_);

        //Mat init_labels;
        //d_block_labels_.download(init_labels);
        //::NormalizeLabels(init_labels);
        //Mat img_out;
        //ColorLabels(init_labels, img_out);
        //volwrite("C:\\Users\\Stefano\\Desktop\\debug\\init_labels", img_out);


        while (true) {
            changes = 0;
            cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

            Scan << <grid_size_, block_size_ >> > (d_block_labels_, d_connections_, d_changes);

            cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

            if (!changes)
                break;

            Analyze << <grid_size_, block_size_ >> > (d_block_labels_);

        }

        //Mat block_labels;
        //d_block_labels_.download(block_labels);
        //::NormalizeLabels(block_labels);
        //ColorLabels(block_labels, img_out);
        //volwrite("C:\\Users\\Stefano\\Desktop\\debug\\block_labels", img_out);

        FinalLabeling << <grid_size_, block_size_ >> > (d_block_labels_, d_img_labels_, d_img_);


        //d_img_labels_.download(img_labels_);

        cudaFree(d_changes);
        d_connections_.release();
        d_block_labels_.release();

        cudaDeviceSynchronize();

        //d_img_labels_.download(img_labels_);
        //Mat errors;
        //bool correct = CheckLabeledVolume(img_, img_labels_, errors);
        //volwrite("C:\\Users\\Stefano\\Desktop\\debug\\BE_errors", errors);
    }


private:
    double Alloc() {
        perf_.start();
        d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);
        d_connections_.create((d_img_.x + 1) / 2, (d_img_.y + 1) / 2, (d_img_.z + 1) / 2, CV_32SC1);
        d_block_labels_.create((d_img_.x + 1) / 2, (d_img_.y + 1) / 2, (d_img_.z + 1) / 2, CV_32SC1);
        cudaMalloc(&d_changes, sizeof(char));
        cudaMemset(d_img_labels_.data, 0, d_img_labels_.stepz * d_img_labels_.z);
        cudaMemset(d_connections_.data, 0, d_connections_.stepz * d_connections_.z);
        cudaMemset(d_block_labels_.data, 0, d_block_labels_.stepz * d_block_labels_.z);
        cudaMemset(d_changes, 0, 1);
        cudaDeviceSynchronize();
        double t = perf_.stop();
        perf_.start();
        cudaMemset(d_img_labels_.data, 0, d_img_labels_.stepz * d_img_labels_.z);
        cudaMemset(d_connections_.data, 0, d_connections_.stepz * d_connections_.z);
        cudaMemset(d_block_labels_.data, 0, d_block_labels_.stepz * d_block_labels_.z);
        cudaMemset(d_changes, 0, 1);
        cudaDeviceSynchronize();
        t -= perf_.stop();
        return t;
    }

    double Dealloc() {
        perf_.start();
        cudaFree(d_changes);
        d_connections_.release();
        d_block_labels_.release();
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
		grid_size_ = dim3((d_block_labels_.x + BLOCK_X - 1) / BLOCK_X, (d_block_labels_.y + BLOCK_Y - 1) / BLOCK_Y, (d_block_labels_.z + BLOCK_Z - 1) / BLOCK_Z);
		block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

        Init << <grid_size_, block_size_ >> > (d_img_, d_connections_, d_block_labels_);
        // La Init esplode
        // Controlla che cosa contiene connections
        //cuda::GpuMat d_expanded_connections;
        //d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
        //ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
        //Mat1b expanded_connections;
        //d_expanded_connections.download(expanded_connections);
        //d_expanded_connections.release();

        //assert(cudaDeviceSynchronize() == cudaSuccess);

        // Immagine di debug della inizializzazione
        //Mat1i init_labels;
        //d_block_labels_.download(init_labels);

        while (true) {
            changes = 0;
            cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

            Scan << <grid_size_, block_size_ >> > (d_block_labels_, d_connections_, d_changes);

            cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

            if (!changes)
                break;

            Analyze << <grid_size_, block_size_ >> > (d_block_labels_);
        }

        // Immagine di debug delle label dei blocchi
        //Mat1i block_labels;
        //d_block_labels_.download(block_labels);

        FinalLabeling << <grid_size_, block_size_ >> > (d_block_labels_, d_img_labels_, d_img_);
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

REGISTER_LABELING(BE_3D);

