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

// Oliveira2010

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

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


	__global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_row = threadIdx.y;
		unsigned local_col = threadIdx.x;
		unsigned local_index = local_row * blockDim.x + local_col;

		unsigned global_row = blockIdx.y * BLOCK_ROWS + local_row;
		unsigned global_col = blockIdx.x * blockDim.x + local_col;
		unsigned img_index = global_row * img.step + global_col;

		__shared__ int s_buf[BLOCK_ROWS * BLOCK_COLS];
		__shared__ unsigned char s_img[BLOCK_ROWS * BLOCK_COLS];

		bool in_limits = (global_row < img.rows && global_col < img.cols);

		s_buf[local_index] = local_index + 1;
		s_img[local_index] = in_limits ? img[img_index] : 0xFF;

		__syncthreads();

		unsigned char v = s_img[local_index];

		if (in_limits) {

			if (v) {

				if (local_col > 0 && s_img[local_index - 1]) {
					Union(s_buf, local_index, local_index - 1);
				}


				if (local_row > 0 && s_img[local_index - blockDim.x]) {
					Union(s_buf, local_index, local_index - blockDim.x);
				}

			}

			else {
				if (local_row > 0 && s_img[local_index - blockDim.x]) {

					if (local_col > 0 && s_img[local_index - 1]) {
						Union(s_buf, local_index - 1, local_index - blockDim.x);
					}


					if (local_col < blockDim.x - 1 && s_img[local_index + 1]) {
						Union(s_buf, local_index + 1, local_index - blockDim.x);
					}
				}

			}

		}

		__syncthreads();

		if (in_limits) {

			if (v) {
				unsigned f = Find(s_buf, local_index);
				unsigned f_row = f / blockDim.x;
				unsigned f_col = f % blockDim.x;
				unsigned global_f = (blockIdx.y * BLOCK_ROWS + f_row) * (labels.step / labels.elem_size) + (blockIdx.x * blockDim.x + f_col);
				labels.data[global_row * labels.step / sizeof(int) + global_col] = global_f + 1;		// C'� distinzione tra background e foreground
			}

			else {
				labels.data[global_row * labels.step / sizeof(int) + global_col] = 0;
			}

		}
	}


	__global__ void LocalMergeSize(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_row = threadIdx.y;
		unsigned local_col = threadIdx.x;
		unsigned local_index = local_row * blockDim.x + local_col;

		unsigned global_row = blockIdx.y * blockDim.y + local_row;
		unsigned global_col = blockIdx.x * blockDim.x + local_col;
		unsigned img_index = global_row * img.step + global_col;

		extern __shared__ unsigned char shared_memory[];

		int* s_buf = reinterpret_cast<int*>(shared_memory);
		unsigned char* s_img = reinterpret_cast<unsigned char*>(shared_memory + blockDim.y * blockDim.x * sizeof(int));

		bool in_limits = (global_row < img.rows&& global_col < img.cols);

		s_buf[local_index] = local_index + 1;
		s_img[local_index] = in_limits ? img[img_index] : 0xFF;

		__syncthreads();

		unsigned char v = s_img[local_index];

		if (in_limits) {

			if (v) {

				if (local_col > 0 && s_img[local_index - 1]) {
					Union(s_buf, local_index, local_index - 1);
				}


				if (local_row > 0 && s_img[local_index - blockDim.x]) {
					Union(s_buf, local_index, local_index - blockDim.x);
				}

			}

			else {
				if (local_row > 0 && s_img[local_index - blockDim.x]) {

					if (local_col > 0 && s_img[local_index - 1]) {
						Union(s_buf, local_index - 1, local_index - blockDim.x);
					}


					if (local_col < blockDim.x - 1 && s_img[local_index + 1]) {
						Union(s_buf, local_index + 1, local_index - blockDim.x);
					}
				}

			}

		}

		__syncthreads();

		if (in_limits) {

			if (v) {
				unsigned f = Find(s_buf, local_index);
				unsigned f_row = f / blockDim.x;
				unsigned f_col = f % blockDim.x;
				unsigned global_f = (blockIdx.y * blockDim.y + f_row) * (labels.step / labels.elem_size) + (blockIdx.x * blockDim.x + f_col);
				labels.data[global_row * labels.step / sizeof(int) + global_col] = global_f + 1;		// C'è distinzione tra background e foreground
			}

			else {
				labels.data[global_row * labels.step / sizeof(int) + global_col] = 0;
			}

		}
	}



	__global__ void GlobalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_row = threadIdx.y;
		unsigned local_col = threadIdx.x;

		unsigned global_row = blockIdx.y * blockDim.y + local_row;
		unsigned global_col = blockIdx.x * blockDim.x + local_col;
		unsigned img_index = global_row * img.step + global_col;
		unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

		bool in_limits = (global_row < img.rows && global_col < img.cols);

		if (in_limits) {

			unsigned char v = img[img_index];

			if (v) {

				if (global_col > 0 && local_col == 0 && img[img_index - 1]) {
					Union(labels.data, labels_index, labels_index - 1);
				}

				if (global_row > 0 && local_row == 0 && img[img_index - img.step]) {
					Union(labels.data, labels_index, labels_index - labels.step / sizeof(int));
				}

			}

			else {

				if (global_row > 0 && img[img_index - img.step]) {

					if (global_col > 0 && (local_row == 0 || local_col == 0) && img[img_index - 1]) {
						Union(labels.data, labels_index - 1, labels_index - labels.step / sizeof(int));
					}

					if ((global_col < img.cols - 1) && (local_row == 0 || local_col == blockDim.x - 1) && img[img_index + 1]) {
						Union(labels.data, labels_index + 1, labels_index - labels.step / sizeof(int));
					}
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

class UF : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	UF() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		// Phase 1
		// CCL on tiles		
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		// Immagine di debug della prima fase
		//cuda::GpuMat d_local_labels;
		//d_img_labels_.copyTo(d_local_labels);
		//PathCompression << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
		//// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_local_labels);
		//Mat1i local_labels(img_.size());
		//d_local_labels.download(local_labels);

		// Phase 2
		// Merges UFTrees of different tiles
		GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		// Immagine di debug della seconda fase
		//cuda::GpuMat d_global_labels;
		//d_img_labels_.copyTo(d_global_labels);
		//PathCompression << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
		//// ZeroBackground << <grid_size_, block_size_ >> > (d_img_, d_global_labels);
		//Mat1i global_labels(img_.size());
		//d_global_labels.download(global_labels);

		// Phase 3
		// Collapse UFTrees
		PathCompression << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		cudaDeviceSynchronize();
	}


	void PerformLabelingBlocksize(int x, int y, int z) override {

		const int block_cols = x;
		const int block_rows = y;

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		grid_size_ = dim3((d_img_.cols + block_cols - 1) / block_cols, (d_img_.rows + block_rows - 1) / block_rows, 1);
		block_size_ = dim3(block_cols, block_rows, 1);
		int shared_size = block_rows * block_cols * (sizeof(int) + sizeof(unsigned char));

		BLOCKSIZE_KERNEL(LocalMergeSize, grid_size_, block_size_, shared_size, d_img_, d_img_labels_)

		BLOCKSIZE_KERNEL(GlobalMerge, grid_size_, block_size_, 0, d_img_, d_img_labels_)

		BLOCKSIZE_KERNEL(PathCompression, grid_size_, block_size_, 0, d_img_, d_img_labels_)
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
		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);
		cudaDeviceSynchronize();
	}

	void GlobalScan() {
		GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);
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

REGISTER_LABELING(UF);

REGISTER_KERNELS(UF, LocalMergeSize, GlobalMerge, PathCompression)