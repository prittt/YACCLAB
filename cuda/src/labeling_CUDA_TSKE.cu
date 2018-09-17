#include <opencv2/core.hpp>
#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#include <opencv2\core.hpp>
#include <opencv2\cudafeatures2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <map>

// Il minimo per entrambi è 4
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace CUDA_TSKE_namespace {


	// Risale alla radice dell'albero a partire da un suo nodo n
	__device__ unsigned Find(const int *s_buf, unsigned n) {
		// Attenzione: non invocare la find su un pixel di background

		unsigned label = s_buf[n];

		assert(label > 0);

		while (label - 1 != n) {
			n = label - 1;
			label = s_buf[n];

			assert(label > 0);
		}

		return n;

	}


	// Unisce gli alberi contenenti i nodi a e b, collegandone le radici
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



	// Init phase.
	// Labels start at value 1, to differentiate them from background, that has value 0.
	__global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		__shared__ int local_labels[BLOCK_ROWS * BLOCK_COLS];
		unsigned local_index = threadIdx.y * BLOCK_COLS + threadIdx.x;

		bool foreground = false;

		// Init
		if (row < img.rows && col < img.cols) {

			if (img.data[img_index]) {

				foreground = true;

				if (threadIdx.y > 0 && img.data[img_index - img.step]) {
					local_labels[local_index] = local_index - BLOCK_COLS + 1;
				}

				else if (threadIdx.y > 0 && threadIdx.x > 0 && img.data[img_index - img.step - 1]) {
					local_labels[local_index] = local_index - BLOCK_COLS;

				}

				else if (threadIdx.y > 0 && threadIdx.x < BLOCK_COLS - 1 && col < img.cols - 1 && img.data[img_index - img.step + 1]) {
					local_labels[local_index] = local_index - BLOCK_COLS + 2;
				}

				else if (threadIdx.x > 0 && img.data[img_index - 1]) {
					local_labels[local_index] = local_index;

				}

				else {
					local_labels[local_index] = local_index + 1;
				}
			}
		}

		__syncthreads();

		// LocalMerge
		if (foreground) {
			local_labels[local_index] = Find(local_labels, local_index) + 1;
		}

		__syncthreads();
		
		// LocalReduction
		if (foreground) {
			if (threadIdx.x > 0 && img.data[img_index - 1]) {
				Union(local_labels, local_index, local_index - 1);
			}
			if (threadIdx.y > 0 && threadIdx.x < BLOCK_COLS - 1 && col < labels.cols - 1 && img.data[img_index - img.step + 1]) {
				Union(local_labels, local_index, local_index - BLOCK_COLS + 1);
			}
		}

		__syncthreads();

		if (foreground) {
			unsigned f = Find(local_labels, local_index);
			unsigned f_row = f / BLOCK_COLS;
			unsigned f_col = f % BLOCK_COLS;
			unsigned global_f = (blockIdx.y * BLOCK_ROWS + f_row) * (labels.step / labels.elem_size) + (blockIdx.x * BLOCK_COLS + f_col);
			labels.data[labels_index] = global_f + 1;
		}
		else if (row < img.rows && col < img.cols) {
			labels.data[labels_index] = 0;
		}
	}


	__global__ void GlobalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < img.rows && col < img.cols) {

			if (img[img_index]) {

				if (threadIdx.x == 0 && col > 0 && img[img_index - 1]) {
					Union(labels.data, labels_index, labels_index - 1);
				}

				if (threadIdx.y == 0 && row > 0 && img[img_index - img.step]) {
					Union(labels.data, labels_index, labels_index - labels.step / labels.elem_size);
				}

				if (row > 0 && col > 0 && (threadIdx.y == 0 || threadIdx.x == 0) && img[img_index - img.step - 1]) {
					Union(labels.data, labels_index, labels_index - labels.step / sizeof(int) - 1);
				}

				if (row > 0 && (col < img.cols - 1) && (threadIdx.y == 0 || threadIdx.x == BLOCK_COLS - 1) && img[img_index - img.step + 1]) {
					Union(labels.data, labels_index, labels_index - labels.step / sizeof(int) + 1);
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

			if (label) {								// Performances are the same as the paper variant

				unsigned index = labels_index;

				while (label - 1 != index) {
					index = label - 1;
					label = labels[index];
				}

				labels[labels_index] = label;
			}
		}
	}


	__global__ void PathCompression(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned global_row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned global_col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

		if (global_row < labels.rows && global_col < labels.cols) {
			if (img[global_row * img.step + global_col]) {
				labels[labels_index] = Find(labels.data, labels_index) + 1;
			}
		}
	}


	__global__ void Reduce(const cuda::PtrStepb img, cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			if (img.data[img_index]) {

				if (col > 0 && img.data[img_index - 1]) {
					Union(labels.data, labels_index, labels_index - 1);
				}

				if (row > 0 && col < labels.cols - 1 && img.data[img_index - img.step + 1]) {
					Union(labels.data, labels_index, labels_index - (labels.step / labels.elem_size) + 1);
				}

			}

		}

	}

}

using namespace CUDA_TSKE_namespace;

class CUDA_TSKE : public GpuLabeling {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	CUDA_TSKE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		Mat1i local_labels;
		d_img_labels_.download(local_labels);
		// init_labels.release();

		GlobalMerge<<<grid_size_, block_size_>>>(d_img_, d_img_labels_);
		Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

		Mat1i final_labels;
		d_img_labels_.download(final_labels);

		assert(cudaDeviceSynchronize() == cudaSuccess);
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

	void AllScans() {
		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		//Mat1i local_labels;
		//d_img_labels_.download(local_labels);
		// init_labels.release();

		GlobalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);
		Analyze << <grid_size_, block_size_ >> > (d_img_labels_);

		//Mat1i final_labels;
		//d_img_labels_.download(final_labels);

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

REGISTER_LABELING(CUDA_TSKE);

