// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional Authors:
// Fanny Nina Paravecino and David Kaeli
// Northeastern University, Boston MA 02115, USA

#include <opencv2/cudafeatures2d.hpp>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"


#define THREADSX 16
#define THREADSY 16
#define THREADS 512
#define COLS 512
#define COLSHALF 256
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BUF_SIZE 256
#define MAX_LABELS 262144

typedef unsigned int uint;

class errorHandler { };
using namespace std;

typedef unsigned char uchar;

using namespace cv;

namespace {

    static void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    // CUDA kernels
    __global__ void findSpansKernel(int* out, int* components, const uchar* in,
        const int rows, const int cols, const int step)
    {
        uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint colsSpans = ((cols + 2 - 1) / 2) * 2;
        int current;
        int colsComponents = colsSpans / 2;
        bool flagFirst = true;
        int indexOut = 0;
        int indexComp = 0;
        int comp = i * colsComponents;
        if (i < rows)
        {
            for (int j = 0; j < cols; j++)
            {
                if (flagFirst && in[i * step + j] > 0)
                {
                    current = in[i * step + j];
                    out[i * colsSpans + indexOut] = j;
                    indexOut++;
                    flagFirst = false;
                }
                if (!flagFirst && in[i * step + j] != current)
                {
                    out[i * colsSpans + indexOut] = j - 1;
                    indexOut++;
                    flagFirst = true;
                    /*add the respective label*/
                    components[i * colsComponents + indexComp] = comp;
                    indexComp++;
                    comp++;
                }
            }
            if (!flagFirst)
            {
                out[i * colsSpans + indexOut] = cols - 1;
                /*add the respective label*/
                components[i * colsComponents + indexComp] = comp;
            }
        }
    }

    __global__ void relabelUnrollKernel(int* components, int previousLabel, int newLabel, const int colsComponents, const int idx, const int rows, const int factor)
    {
        uint id_i_child = (blockIdx.x * blockDim.x) + threadIdx.x;
        id_i_child = id_i_child + (rows * idx);
        uint id_j_child = (blockIdx.y * blockDim.y) + threadIdx.y;
        id_j_child = (colsComponents / factor) * id_j_child;
        uint i = id_i_child;
        //for (int j = id_j_child; j < (colsComponents / factor); j++)    // SA shouldn't it be j < (colsComponents / factor) * (id_j_child + 1)?
        if (i < rows) {
            for (int j = id_j_child; (j < (colsComponents / factor) * (id_j_child + 1)) && (j < colsComponents); j++)
            {
                if (components[i * colsComponents + j] == previousLabel)
                {
                    components[i * colsComponents + j] = newLabel;
                }
            }
        }
    }

    __global__ void mergeSpansKernel(int* components, int* spans, const int rows, const int cols)
    {
        //uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint idx = 0;
        uint colsSpans = ((cols + 2 - 1) / 2) * 2;
        uint colsComponents = colsSpans / 2;
        /*Merge Spans*/
        int startX, endX, newStartX, newEndX;
        int label = -1;
        /*threads and blocks need to relabel the components labels*/
        int threads = 16;
        const int factor = 4;

        /*--------For 256, 512--------*/
        dim3 threadsPerBlockUnrollRelabel(threads * threads);
        dim3 numBlocksUnrollRelabel((rows + threads * threads - 1) / (threads * threads), factor);
        /*-----------------*/
        for (int i = 0; i < rows - 1; i++) //compute until penultimate row, since we need the below row to compare
        {
            for (int j = 0; j < colsSpans - 1 && spans[i * colsSpans + j] >= 0; j = j + 2) //verify if there is a Span available
            {
                startX = spans[i * colsSpans + j];
                endX = spans[i * colsSpans + j + 1];
                int newI = i + 1; //line below
                for (int k = 0; k < colsSpans - 1 && spans[newI * colsSpans + k] >= 0; k = k + 2) //verify if there is a New Span available
                {
                    newStartX = spans[newI * colsSpans + k];
                    newEndX = spans[newI * colsSpans + k + 1];
                    if (startX <= newEndX + 1 && endX >= newStartX - 1)//Merge components
                    {
                        label = components[i * (colsSpans / 2) + (j / 2)];          //choose the startSpan label

                        relabelUnrollKernel << <numBlocksUnrollRelabel, threadsPerBlockUnrollRelabel >> > (components, components[newI * (colsSpans / 2) + (k / 2)], label, colsComponents, idx, rows, factor);

                        cudaDeviceSynchronize();
                        //cudaError_t err = cudaGetLastError();
                        //if (err != cudaSuccess)
                        //    printf("\tError:%s \n", (char)err);
                    }
                    //__syncthreads();
                }
            }
        }
    }

    __global__ void makeOutput(const int* components, const int* spans, const int rows, const int cols, int* labels, const int labelsStep) {

        const int r = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;

        const uint colsSpans = ((cols + 2 - 1) / 2) * 2;
        const uint colsComponents = colsSpans / 2;

        if (r < rows && j < colsComponents) {

            const int label = components[r * colsComponents + j];
            if (components[r * colsComponents + j] >= 0) {

                const int cStart = spans[r * colsSpans + 2 * j];
                const int cEnd = spans[r * colsSpans + 2 * j + 1];
                for (int c = cStart; c <= cEnd; c++) {
                    labels[r * labelsStep + c] = label + 1;
                }
            }
        }
    }
}

class ACCL : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    int* devComponents_;
    int* devOut_;

public:
    ACCL() {}

    void PerformLabeling() {

        const int cols = img_.cols;
        const int rows = img_.rows;

        const int colsSpans = ((cols + 1) / 2) * 2; /*ceil(cols/2)*2*/

        d_img_labels_.create(d_img_.size(), CV_32SC1);
        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        //int* devIn = 0;
        int* devComponents = 0;
        int* devOut = 0;

        const int colsComponents = colsSpans / 2;

        /*compute sizes of matrices*/
        const int sizeComponents = colsComponents * rows;
        const int sizeOut = colsSpans * rows;

        /*Block and Grid size*/
        int blockSize;
        //int minGridSize;
        int gridSize;

        /* Allocate GPU buffers for three vectors (two input, one output)*/
        cudaErrChk(cudaMalloc((void**)&devOut, sizeOut * sizeof(int)));
        cudaErrChk(cudaMalloc((void**)&devComponents, sizeComponents * sizeof(int)));

        cudaErrChk(cudaMemset(devOut, -1, sizeOut * sizeof(int)));
        cudaErrChk(cudaMemset(devComponents, -1, sizeComponents * sizeof(int)));

        /* Round up according to array size */
        blockSize = 512;
        gridSize = (rows + blockSize - 1) / blockSize;
        //gridSize = rows/blockSize;

        const int magicSize = 16;
        const dim3 makeOutputBlockSize(magicSize, magicSize);
        const dim3 makeOutputGridSize((colsComponents + magicSize - 1) / magicSize, (rows + magicSize - 1) / magicSize);

        findSpansKernel << <gridSize, blockSize >> > (devOut,
            devComponents, d_img_.data,
            rows, cols, (const int) d_img_.step);

        mergeSpansKernel << <1, 1 >> > (devComponents, devOut, rows, cols);

        makeOutput << < makeOutputGridSize, makeOutputBlockSize >> > (
            devComponents, devOut, rows, cols, 
            (int*) d_img_labels_.data, (const int) d_img_labels_.step / 4);

        /*Free*/
        cudaFree(devOut);
        cudaFree(devComponents);

        cudaDeviceSynchronize();
    }

private:
    double Alloc() {
        perf_.start();
        const int cols = img_.cols;
        const int rows = img_.rows;

        const int colsSpans = ((cols + 1) / 2) * 2; /*ceil(cols/2)*2*/
        const int colsComponents = colsSpans / 2;

        /*compute sizes of matrices*/
        const int sizeComponents = colsComponents * rows;
        const int sizeOut = colsSpans * rows;

        /* Allocate GPU buffers for three vectors (two input, one output)*/
        cudaErrChk(cudaMalloc((void**)&devOut_, sizeOut * sizeof(int)));
        cudaErrChk(cudaMalloc((void**)&devComponents_, sizeComponents * sizeof(int)));
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        perf_.stop();
        return perf_.last();
    }

    double Dealloc() {
        perf_.start();
        cudaFree(devOut_);
        cudaFree(devComponents_);
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

    void FirstScan() {
        const int cols = img_.cols;
        const int rows = img_.rows;

        const int colsSpans = ((cols + 1) / 2) * 2; /*ceil(cols/2)*2*/

        cudaMemset2D(d_img_labels_.data, d_img_labels_.step, 0, d_img_labels_.cols * 4, d_img_labels_.rows);

        const int colsComponents = colsSpans / 2;

        /*compute sizes of matrices*/
        const int sizeComponents = colsComponents * rows;
        const int sizeOut = colsSpans * rows;

        /*Block and Grid size*/
        int blockSize;
        //int minGridSize;
        int gridSize;

        cudaErrChk(cudaMemset(devOut_, -1, sizeOut * sizeof(int)));
        cudaErrChk(cudaMemset(devComponents_, -1, sizeComponents * sizeof(int)));

        /* Round up according to array size */
        blockSize = 512;
        gridSize = (rows + blockSize - 1) / blockSize;
        
        //int* out = new int[sizeOut];
        //int* components = new int[sizeComponents];

        findSpansKernel << <gridSize, blockSize >> > (devOut_,
            devComponents_, d_img_.data,
            rows, cols, (const int)d_img_.step);

        //cudaErrChk(cudaMemcpy(out, devOut_, sizeOut * sizeof(int), cudaMemcpyDeviceToHost));
        //cudaErrChk(cudaMemcpy(components, devComponents_, sizeComponents * sizeof(int), cudaMemcpyDeviceToHost));

        mergeSpansKernel << <1, 1 >> > (devComponents_, devOut_, rows, cols);

        //cudaErrChk(cudaMemcpy(out, devOut_, sizeOut * sizeof(int), cudaMemcpyDeviceToHost));
        //cudaErrChk(cudaMemcpy(components, devComponents_, sizeComponents * sizeof(int), cudaMemcpyDeviceToHost));

        //delete[] out;
        //delete[] components;

        cudaDeviceSynchronize();
    }

    void SecondScan() {
        const int cols = img_.cols;
        const int rows = img_.rows;

        const int colsSpans = ((cols + 1) / 2) * 2; /*ceil(cols/2)*2*/
        const int colsComponents = colsSpans / 2;

        const int magicSize = 16;
        const dim3 makeOutputBlockSize(magicSize, magicSize);
        const dim3 makeOutputGridSize((colsComponents + magicSize - 1) / magicSize, (rows + magicSize - 1) / magicSize);

        makeOutput << < makeOutputGridSize, makeOutputBlockSize >> > (
            devComponents_, devOut_, rows, cols,
            (int*)d_img_labels_.data, (const int)d_img_labels_.step / 4);

        cudaDeviceSynchronize();
    }

public:
    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        double dealloc_timing = Dealloc();

        perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);

    }

};

REGISTER_LABELING(ACCL);