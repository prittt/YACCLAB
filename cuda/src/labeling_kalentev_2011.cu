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

// Label Equivalence with optimization introduced by Kalentev (OLE stands for Optimized Label Equivalence)

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


	__device__ unsigned int MinLabel(unsigned l1, unsigned l2) {
		if (l1 && l2)
			return min(l1, l2);
		else
			return l1;
	}


	__device__ unsigned int FindMinLabel(cuda::PtrStepSzi labels, unsigned row, unsigned col, unsigned label, unsigned labels_index) {

		unsigned int min = label;

		if (row > 0) {
			min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size)]);
			if (col > 0)
				min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size) - 1]);
			if (col < labels.cols - 1)
				min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size) + 1]);
		}
		if (row < labels.rows - 1) {
			min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size)]);
			if (col > 0)
				min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size) - 1]);
			if (col < labels.cols - 1)
				min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size) + 1]);
		}
		if (col > 0)
			min = MinLabel(min, labels.data[labels_index - 1]);
		if (col < labels.cols - 1)
			min = MinLabel(min, labels.data[labels_index + 1]);
		
		return min;
	}


	// Scan phase.
	// The pixel associated with current thread is given the minimum label of the neighbours.
	__global__ void Scan(cuda::PtrStepSzi labels, char *changes) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;
		
		if (row < labels.rows && col < labels.cols) {

			unsigned label = labels[labels_index];
			
			if (label) {
				unsigned min_label = FindMinLabel(labels, row, col, label, labels_index);
				if (min_label < label) {
					labels[label - 1] = min(static_cast<unsigned int>(labels[label - 1]), min_label);
					*changes = 1;
				}
			}
		}
	}


	// Analysis phase.
	// The pixel associated with current thread is given the minimum label of the neighbours.
	__global__ void Analyze(cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned label = labels[labels_index];

			if (label) {								

				unsigned index = labels_index;

				while (label - 1 != index) {
					index = label - 1;
					label = labels[index];
				}				

				labels[labels_index] = label;
			}
		}
	}

}

class OLE : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;
	char changes;
	char *d_changes;

public:
	OLE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		cudaMalloc(&d_changes, sizeof(char));

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);		

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		while (true) {
			changes = 0;
			cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

			Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes);

			cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

			if (!changes)
				break;

			Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		}
		cudaFree(d_changes);
		cudaDeviceSynchronize();		
	}

	void PerformLabelingBlocksize(int x, int y, int z) override {

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		cudaMalloc(&d_changes, sizeof(char));

		grid_size_ = dim3((d_img_.cols + x - 1) / x, (d_img_.rows + y - 1) / y, 1);
		block_size_ = dim3(x, y, 1);

		BLOCKSIZE_KERNEL(Init, grid_size_, block_size_, 0, d_img_, d_img_labels_)

		while (true) {
			changes = 0;
			cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

			BLOCKSIZE_KERNEL(Scan, grid_size_, block_size_, 0, d_img_labels_, d_changes)

			cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

			if (!changes)
				break;

			BLOCKSIZE_KERNEL(Analyze, grid_size_, block_size_, 0, d_img_labels_)
		}
		cudaFree(d_changes);
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

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		while (true) {
			changes = 0;
			cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

			Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes);

			cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);
			if (!changes)
				break;

			Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		}

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

REGISTER_LABELING(OLE);

REGISTER_KERNELS(OLE, Init, Scan, Analyze)