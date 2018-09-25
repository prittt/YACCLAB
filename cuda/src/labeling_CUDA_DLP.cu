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
#include <cassert>

#include <opencv2\core.hpp>
#include <opencv2\cudafeatures2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <map>

// Il minimo per entrambi è 4
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

#define SOUTH_BORDER 32
#define EAST_BORDER 32

#define RELABELING_ROWS 16
#define RELABELING_COLS 16

using namespace cv;

namespace CUDA_DLP_namespace {


	// Risale alla radice dell'albero a partire da un suo nodo n
	__device__ unsigned Find(const int *s_buf, unsigned n) {			// n is an index but return value is a label
		// You can now call Find on a background pixel

		unsigned label = s_buf[n];
		if (label) {

			while (label - 1 != n) {
				n = label - 1;
				label = s_buf[n];
			}

			return n + 1;
		}
		else {
			return 0;
		}
	}

	// SetRoot procedure
	__device__ void SetRoot(int *labels, unsigned label, unsigned eps) {			// label and eps are both labels (not indexes; labels are shifted by one to the right wrt indexes)

		int father = labels[label - 1];

		if (father > eps) {
			labels[label - 1] = eps;
		}
	}

	// atomicRUF procedure
	__device__ void atomicRUF(int *labels, unsigned label, unsigned eps) {

		if (label > eps) {
			unsigned minResult = atomicMin(labels + label - 1, eps);
			if (eps > minResult) {
				atomicRUF(labels, eps, minResult);
			}
			else {
				if (label > minResult) {
					atomicRUF(labels, minResult, eps);
				}
			}
		}
	}


	//Effettuo il controllo sui 4 vicini della maschera
	//Prova a sincronizzare dopo ogni vicino
	__global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_index = threadIdx.y * BLOCK_COLS + threadIdx.x;

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned img_index = row * img.step + col;

		__shared__ int s_buf[BLOCK_ROWS * BLOCK_COLS];
		__shared__ unsigned char s_img[BLOCK_ROWS * BLOCK_COLS];

		bool in_limits = (row < img.rows && col < img.cols);		// borders aren't processed

		// DLP-I
		s_img[local_index] = in_limits ? img[img_index] : 0xFF;
		unsigned char v = s_img[local_index];
		s_buf[local_index] = v ? local_index + 1 : 0;

		__syncthreads();

		// DLP-SR (optional)
		if (threadIdx.y < BLOCK_ROWS - 1 && threadIdx.x < BLOCK_COLS - 1 && row < labels.rows - 1 && col < labels.cols - 1) {
			
			int min = INT32_MAX;
			int a[4];
			a[0] = s_buf[local_index];
			if (a[0] != 0 && a[0] < min)
				min = a[0];
			a[1] = s_buf[local_index + 1];
			if (a[1] != 0 && a[1] < min)
				min = a[1];
			a[2] = s_buf[local_index + BLOCK_COLS];
			if (a[2] != 0 && a[2] < min)
				min = a[2];
			a[3] = s_buf[local_index + BLOCK_COLS + 1];
			if (a[3] != 0 && a[3] < min)
				min = a[3];

			for (unsigned i = 0; i < 4; i++) {
				unsigned int label = a[i];
				if (label != 0 && label != min) {
					SetRoot(s_buf, label, min);
				}
			}
		}

		__syncthreads();

		// DLP-R (optional)
		if (v && in_limits) {
			s_buf[local_index] = Find(s_buf, local_index);
		}

		__syncthreads();

		// DLP-RUF
		if (threadIdx.y < BLOCK_ROWS - 1 && threadIdx.x < BLOCK_COLS - 1 && row < labels.rows - 1 && col < labels.cols - 1) {

			int min = INT32_MAX;
			int a[4];
			a[0] = s_buf[local_index];
			if (a[0] != 0 && a[0] < min)
				min = a[0];
			a[1] = s_buf[local_index + 1];
			if (a[1] != 0 && a[1] < min)
				min = a[1];
			a[2] = s_buf[local_index + BLOCK_COLS];
			if (a[2] != 0 && a[2] < min)
				min = a[2];
			a[3] = s_buf[local_index + BLOCK_COLS + 1];
			if (a[3] != 0 && a[3] < min)
				min = a[3];

			for (unsigned i = 0; i < 4; i++) {
				unsigned int label = a[i];
				if (label != 0 && label != min) {
					atomicRUF(s_buf, label, min);
				}
			}
		}

		__syncthreads();

		// DLP-R
		if (v && in_limits) {
			s_buf[local_index] = Find(s_buf, local_index);
		}

		__syncthreads();

		// Labeltranslation			
		if (in_limits) {

			if (v) {
				unsigned f = Find(s_buf, local_index) - 1;
				unsigned f_row = f / BLOCK_COLS;
				unsigned f_col = f % BLOCK_COLS;
				unsigned global_f = (blockIdx.y * BLOCK_ROWS + f_row) * (labels.step / labels.elem_size) + (blockIdx.x * BLOCK_COLS + f_col);
				labels.data[row * labels.step / sizeof(int) + col] = global_f + 1;
			}
			else {
				labels.data[row * labels.step / sizeof(int) + col] = 0;
			}
		}
	}

	__global__ void SouthBorderMerge(cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y + 1) * BLOCK_ROWS - 1;
		unsigned col = blockIdx.x * SOUTH_BORDER + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		bool in_limits = (col < labels.cols - 1);

		if (in_limits) {

			int min = INT32_MAX;
			int a[4];
			a[0] = labels[labels_index];
			if (a[0] != 0 && a[0] < min)
				min = a[0];
			a[1] = labels[labels_index + 1];
			if (a[1] != 0 && a[1] < min)
				min = a[1];
			a[2] = labels[labels_index + labels.step / labels.elem_size];
			if (a[2] != 0 && a[2] < min)
				min = a[2];
			a[3] = labels[labels_index + labels.step / labels.elem_size + 1];
			if (a[3] != 0 && a[3] < min)
				min = a[3];

			for (unsigned i = 0; i < 4; i++) {
				unsigned int label = a[i];
				if (label != 0 && label != min) {
					atomicRUF(labels, label, min);
				}
			}
		}
	}

	__global__ void EastBorderMerge(cuda::PtrStepSzi labels) {

		unsigned col = (blockIdx.x + 1) * BLOCK_COLS - 1;
		unsigned row = blockIdx.y * EAST_BORDER + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		bool in_limits = (row < labels.rows - 1);

		if (in_limits) {

			int min = INT32_MAX;
			int a[4];
			a[0] = labels[labels_index];
			if (a[0] != 0 && a[0] < min)
				min = a[0];
			a[1] = labels[labels_index + 1];
			if (a[1] != 0 && a[1] < min)
				min = a[1];
			a[2] = labels[labels_index + labels.step / labels.elem_size];
			if (a[2] != 0 && a[2] < min)
				min = a[2];
			a[3] = labels[labels_index + labels.step / labels.elem_size + 1];
			if (a[3] != 0 && a[3] < min)
				min = a[3];

			for (unsigned i = 0; i < 4; i++) {
				unsigned int label = a[i];
				if (label != 0 && label != min) {
					atomicRUF(labels, label, min);
				}
			}
		}
	}

	__global__ void Relabeling(cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * RELABELING_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * RELABELING_COLS + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			labels[labels_index] = Find(labels.data, labels_index);
		}
	}

}

using namespace CUDA_DLP_namespace;

class CUDA_DLP : public GpuLabeling {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	CUDA_DLP() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		// Phase 1
		// Etichetta i pixel localmente al blocco		
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		// Immagine di debug della prima fase
		//Mat1i local_labels;
		//d_img_labels_.download(local_labels);

		// Phase 2
		// Collega tra loro gli alberi union-find dei diversi blocchi
		SouthBorderMerge << <dim3((d_img_.cols + SOUTH_BORDER - 1) / SOUTH_BORDER, grid_size_.y - 1, 1), SOUTH_BORDER >> > (d_img_labels_);
		EastBorderMerge << <dim3(grid_size_.x - 1, (d_img_.rows + EAST_BORDER - 1) / EAST_BORDER, 1), EAST_BORDER >> > (d_img_labels_);

		//Mat1i border_labels;
		//d_img_labels_.download(border_labels);

		// Phase 3
		// Collassa gli alberi union-find sulle radici
		grid_size_ = dim3((d_img_.cols + RELABELING_COLS - 1) / RELABELING_COLS, (d_img_.rows + RELABELING_ROWS - 1) / RELABELING_ROWS, 1);
		block_size_ = dim3(RELABELING_COLS, RELABELING_ROWS, 1);
		Relabeling << <grid_size_, block_size_ >> > (d_img_labels_);

		//Mat1i final_labels;
		//d_img_labels_.download(final_labels);

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
		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);
		
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);
		cudaDeviceSynchronize();
	}

	void GlobalScan() {
		SouthBorderMerge << <dim3((d_img_.cols + SOUTH_BORDER - 1) / SOUTH_BORDER, grid_size_.y - 1, 1), SOUTH_BORDER >> > (d_img_labels_);
		EastBorderMerge << <dim3(grid_size_.x - 1, (d_img_.rows + EAST_BORDER - 1) / EAST_BORDER, 1), EAST_BORDER >> > (d_img_labels_);
		grid_size_ = dim3((d_img_.cols + RELABELING_COLS - 1) / RELABELING_COLS, (d_img_.rows + RELABELING_ROWS - 1) / RELABELING_ROWS, 1);
		block_size_ = dim3(RELABELING_COLS, RELABELING_ROWS, 1);
		Relabeling << <grid_size_, block_size_ >> > (d_img_labels_);
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

REGISTER_LABELING(CUDA_DLP);

