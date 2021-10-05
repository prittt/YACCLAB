// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional Authors:
// P. Chen, H.L. Zhao, C. Tao and H.S. Sang
// National Key Laboratory of Science and Technology on Multi-spectral Information Processing
// Institute for Pattern Recognition and Artificial Intelligence, Huazhong
// University of Science and Technology, Luoyu Road 1037, Wuhan, China
//
// This algorithm is described in "Block-run-based connected component labelling algorithm for GPGPU using shared memory", Chen et al., 2011
// The algorithm imposes limits on the image size; therefore, it can only be performed on certain datasets, listed below.
// This is the original implementation provided by the authors, and it is not optimized for YACCLAB tasks.
//
// WARNING: datasets available restricted to the following:
// ["3dpes", "fingerprints", "hamlet", "medical", "mirflickr"], except for medical/00094.png


#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

void chen_label_512(uchar* cbpt, uchar* cbpt2, uchar* cbps, uchar* cbb, uchar* cbb2, uchar* cbglabel, uint h, uint bn, uchar* cbeb);
void chen_label_1024(uchar* cbpt, uchar* cbpt2, uchar* cbps, uchar* cbb, uchar* cbb2, uchar* cbglabel, uint h, uint bn, uchar* cbeb);
void chen_label_2048(uchar* cbpt, uchar* cbpt2, uchar* cbps, uchar* cbb, uchar* cbb2, uchar* cbglabel, uint h, uint bn, uchar* cbeb);

__global__ static void FinalLabeling(uint* labels, int labels_step, ushort* blocks, int blocks_step, uchar* img, int img_step, int width, int height) {

    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned img_index = row * img_step + col;
    unsigned labels_index = row * (labels_step / 4) + col;
    unsigned blocks_index = row / 2 * (blocks_step / 2) + col / 2;

    if (row < height && col < width) {

        ushort label = blocks[blocks_index] + 1;

        if (img[img_index])
            labels[labels_index] = label;
        else {
            labels[labels_index] = 0;
        }

        if (col + 1 < width) {
            if (img[img_index + 1])
                labels[labels_index + 1] = label;
            else {
                labels[labels_index + 1] = 0;
            }

            if (row + 1 < height) {
                if (img[img_index + img_step + 1])
                    labels[labels_index + (labels_step / 4) + 1] = label;
                else {
                    labels[labels_index + (labels_step / 4) + 1] = 0;
                }
            }
        }

        if (row + 1 < height) {
            if (img[img_index + img_step])
                labels[labels_index + (labels_step / 4)] = label;
            else {
                labels[labels_index + (labels_step / 4)] = 0;
            }
        }

    }

}

class BRB : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    int smCnt = 0;
    int bn; //how many partitions
    int nwidth;
    int nheight;

    unsigned char* srcBuf, * dstBuf, * dstBuf2, * glabel, * errBuf, * bBuf, * b2Buf;
    size_t srcBufPitch, dstBufPitch, dstBuf2Pitch, glabelPitch, errBufPitch, bBufPitch, b2BufPitch;

public:

    BRB() {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        smCnt = devProp.multiProcessorCount;
    }

    void PerformLabeling() {

        //choosing the parameters for our ccl
        int bn = 8; //how many partitions
        int nwidth = 512;
        if (d_img_.cols == 1 || d_img_.rows == 1) {
            return;
        }
        if (d_img_.cols > 512) {
            nwidth = 1024;
            bn = 6;
        }
        if (d_img_.cols > 1024) {
            nwidth = 2048;
            bn = 3;
        }
        if (d_img_.cols > 2048) {
            throw std::runtime_error("Chen::PerformLabeling: Image too wide, max width is 2048");
            return;
        }

        if (smCnt == 0) {
            throw std::runtime_error("Chen::PerformLabeling: No device with cap 2.x found");
            return;
        }

        if (smCnt != 0) {
            bn = bn * smCnt;
        }

        int nheight = (d_img_.rows - 2) / (2 * bn);
        if ((nheight * 2 * bn + 2) < d_img_.rows)
            nheight++;
        nheight = nheight * 2 * bn + 2;

        unsigned char* srcBuf, * dstBuf, * dstBuf2, * glabel, * errBuf, * bBuf, * b2Buf;
        size_t srcBufPitch, dstBufPitch, dstBuf2Pitch, glabelPitch, errBufPitch, bBufPitch, b2BufPitch;
        cudaMallocPitch(&srcBuf, &srcBufPitch, nwidth, nheight);
        cudaMallocPitch(&dstBuf, &dstBufPitch, nwidth, (nheight - 2) / 2);
        cudaMallocPitch(&dstBuf2, &dstBuf2Pitch, nwidth, (nheight - 2) / 2);
        cudaMallocPitch(&glabel, &glabelPitch, 4, 1);
        cudaMallocPitch(&errBuf, &errBufPitch, nwidth, 9 * bn);
        cudaMallocPitch(&bBuf, &bBufPitch, nwidth, 2 * bn);
        cudaMallocPitch(&b2Buf, &b2BufPitch, nwidth, 2 * bn);

        cudaMemset2D(srcBuf, srcBufPitch, 0, nwidth, nheight);
        cudaMemcpy2D(srcBuf, srcBufPitch, d_img_.data, d_img_.step, d_img_.cols, d_img_.rows, ::cudaMemcpyDeviceToDevice);

        // Input image debug
        unsigned char* srcBufHost;
        srcBufHost = (unsigned char*)malloc(nwidth * nheight);
        cudaMemcpy2D(srcBufHost, nwidth, srcBuf, srcBufPitch, nwidth, nheight, ::cudaMemcpyDeviceToHost);

        if (nwidth == 512)
            chen_label_512(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);
        else if (nwidth == 1024)
            chen_label_1024(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);
        else if (nwidth == 2048)
            chen_label_2048(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);

        // Block image debug
        unsigned char* dstBufHost;
        dstBufHost = (unsigned char*)malloc(nwidth * ((nheight - 2) / 2));
        cudaMemcpy2D(dstBufHost, nwidth, dstBuf, dstBufPitch, nwidth, (nheight - 2) / 2, ::cudaMemcpyDeviceToHost);

        d_img_labels_.create(d_img_.size(), CV_32SC1);

        grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        FinalLabeling << <grid_size_, block_size_ >> > ((uint*)d_img_labels_.data, (int)d_img_labels_.step, (ushort*)dstBuf, dstBufPitch, (uchar*)srcBuf, srcBufPitch, (int)d_img_.cols, (int)d_img_.rows);

        // Output image debug
        d_img_labels_.download(img_labels_);

        cudaFree(srcBuf);
        cudaFree(dstBuf);
        cudaFree(dstBuf2);
        cudaFree(glabel);
        cudaFree(errBuf);
        cudaFree(bBuf);
        cudaFree(b2Buf);

        cudaDeviceSynchronize();
    }


private:
    void Alloc() {
        //choosing the parameters for our ccl
        bn = 8; //how many partitions
        nwidth = 512;
        if (d_img_.cols == 1 || d_img_.rows == 1) {
            return;
        }
        if (d_img_.cols > 512) {
            nwidth = 1024;
            bn = 6;
        }
        if (d_img_.cols > 1024) {
            nwidth = 2048;
            bn = 3;
        }
        if (d_img_.cols > 2048) {
            throw std::runtime_error("Chen::PerformLabeling: Image too wide, max width is 2048");
            return;
        }

        if (smCnt == 0) {
            throw std::runtime_error("Chen::PerformLabeling: No device with cap 2.x found");
            return;
        }

        if (smCnt != 0) {
            bn = bn * smCnt;
        }

        nheight = (d_img_.rows - 2) / (2 * bn);
        if ((nheight * 2 * bn + 2) < d_img_.rows)
            nheight++;
        nheight = nheight * 2 * bn + 2;

        cudaMallocPitch(&srcBuf, &srcBufPitch, nwidth, nheight);
        cudaMallocPitch(&dstBuf, &dstBufPitch, nwidth, (nheight - 2) / 2);
        cudaMallocPitch(&dstBuf2, &dstBuf2Pitch, nwidth, (nheight - 2) / 2);
        cudaMallocPitch(&glabel, &glabelPitch, 4, 1);
        cudaMallocPitch(&errBuf, &errBufPitch, nwidth, 9 * bn);
        cudaMallocPitch(&bBuf, &bBufPitch, nwidth, 2 * bn);
        cudaMallocPitch(&b2Buf, &b2BufPitch, nwidth, 2 * bn);
        d_img_labels_.create(d_img_.size(), CV_32SC1);
    }

    void Dealloc() {
        cudaFree(srcBuf);
        cudaFree(dstBuf);
        cudaFree(dstBuf2);
        cudaFree(glabel);
        cudaFree(errBuf);
        cudaFree(bBuf);
        cudaFree(b2Buf);
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

    void FirstScan() {
        cudaMemset2D(srcBuf, srcBufPitch, 0, nwidth, nheight);
        cudaMemcpy2D(srcBuf, srcBufPitch, d_img_.data, d_img_.step, d_img_.cols, d_img_.rows, ::cudaMemcpyDeviceToDevice);

        if (nwidth == 512)
            chen_label_512(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);
        else if (nwidth == 1024)
            chen_label_1024(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);
        else if (nwidth == 2048)
            chen_label_2048(dstBuf, dstBuf2, srcBuf, bBuf, b2Buf, glabel, nheight, bn, errBuf);

        cudaDeviceSynchronize();
    }

    void SecondScan() {
        grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
        block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

        FinalLabeling << <grid_size_, block_size_ >> > ((uint*)d_img_labels_.data, (int)d_img_labels_.step, (ushort*)dstBuf, dstBufPitch, (uchar*)srcBuf, srcBufPitch, (int)d_img_.cols, (int)d_img_.rows);

        cudaDeviceSynchronize();
    }


public:
    void PerformLabelingWithSteps()
    {
        if (d_img_.cols == 1 || d_img_.rows == 1) {
            return;
        }

        perf_.start();
        Alloc();
        perf_.stop();
        double alloc_timing = perf_.last();

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        double dealloc_timing = perf_.last();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
    }

};

REGISTER_LABELING(BRB);

