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

namespace CUDA_BKE_namespace {

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

	// Risale alla radice dell'albero a partire da un suo nodo n
	__device__ unsigned Find_label(const int *s_buf, unsigned n, unsigned label) {
		// Attenzione: non invocare la find su un pixel di background

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


	__global__ void InitLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {
		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

#define CONDITION_B (col>0 && row>1 && img.data[img_index - 2 * img.step - 1])
#define CONDITION_C (row>1 && img.data[img_index - 2 * img.step])
#define CONDITION_D (col+1<img.cols && row>1 && img.data[img_index - 2 * img.step + 1])
#define CONDITION_E (col+2<img.cols && row>1 && img.data[img_index - 2 * img.step + 2])

#define CONDITION_G (col>1 && row>0 && img.data[img_index - img.step - 2])
#define CONDITION_H (col>0 && row>0 && img.data[img_index - img.step - 1])
#define CONDITION_I (row>0 && img.data[img_index - img.step])
#define CONDITION_J (col+1<img.step && row>0 && img.data[img_index - img.step + 1])
#define CONDITION_K (col+2<img.step && row>0 && img.data[img_index - img.step + 2])

#define CONDITION_M (col>1 && img.data[img_index - 2])
#define CONDITION_N (col>0 && img.data[img_index - 1])
#define CONDITION_O (img.data[img_index])
#define CONDITION_P (col+1<img.step && img.data[img_index + 1])

#define CONDITION_R (col>0 && row+1<img.rows && img.data[img_index + img.step - 1])
#define CONDITION_S (row+1<img.rows && img.data[img_index + img.step])
#define CONDITION_T (col+1<img.cols && row+1<img.rows && img.data[img_index + img.step + 1])


#define ACTION_NW labels[labels_index] = labels_index - 2 * (labels.step / labels.elem_size) - 1;
#define ACTION_N labels[labels_index] = labels_index - 2 * (labels.step / labels.elem_size) + 1;
#define ACTION_NE labels[labels_index] = labels_index - 2 * (labels.step / labels.elem_size) + 3;
#define ACTION_W labels[labels_index] = labels_index - 1;
#define ACTION_C labels[labels_index] = labels_index + 1;
#define ACTION_0 labels[labels_index] = 0;

		if (row < labels.rows && col < labels.cols) {

#include "labeling_CUDA_BKE_init_tree.inc"

		}

#undef ACTION_NW
#undef ACTION_N
#undef ACTION_NE
#undef ACTION_W
#undef ACTION_C
#undef ACTION_0
	}

	__global__ void Merge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			// Action 1: No action
#define ACTION_NULL 
//			// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_P Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) - 2); 
			// Action Q: Merge with block Q
#define ACTION_Q Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));	
			// Action R: Merge with block R
#define ACTION_R Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2); 
			// Action S: Merge with block S
#define ACTION_S Union(labels.data, labels_index, labels_index - 2);  

#include "labeling_CUDA_BKE_merge_tree.inc"

			if (labels[labels_index]) {

				//Placeholder procedure
				if ((CONDITION_O || CONDITION_P) && (CONDITION_I || CONDITION_J)) {
					ACTION_Q
				}
				if ((CONDITION_O || CONDITION_S) && (CONDITION_N || CONDITION_R)) {
					ACTION_S
				}
				if (CONDITION_P && CONDITION_K) {
					ACTION_R
				}
			}


#undef ACTION_NULL
#undef ACTION_P
#undef ACTION_Q
#undef ACTION_R
#undef ACTION_S

#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T

		}
	}

	__global__ void Compression(cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			unsigned label = labels[labels_index];
			if (label) {
				labels[labels_index] = Find_label(labels.data, labels_index, label) + 1;
			}
		}
	}


	__global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;
		unsigned img_index = row * (img.step / img.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned int label = labels[labels_index];

			if (label) {

				if (img.data[img_index]) {
					// labels[labels_index] = label;
				}
				else {
					labels[labels_index] = 0;
				}

				if (col + 1 < labels.cols) {
					if (img.data[img_index + 1])
						labels[labels_index + 1] = label;
					else {
						labels[labels_index + 1] = 0;
					}

					if (row + 1 < labels.rows) {
						if (img.data[img_index + img.step + 1])
							labels[labels_index + (labels.step / labels.elem_size) + 1] = label;
						else {
							labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
						}
					}
				}

				if (row + 1 < labels.rows) {
					if (img.data[img_index + img.step])
						labels[labels_index + (labels.step / labels.elem_size)] = label;
					else {
						labels[labels_index + (labels.step / labels.elem_size)] = 0;
					}
				}
			}
			else {
				labels[labels_index + 1] = 0;

				if (col + 1 < labels.cols) {
					labels[labels_index + 1] = 0;				

					if (row + 1 < labels.rows) {
						labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
					}
				}

				if (row + 1 < labels.rows) {
					labels[labels_index + (labels.step / labels.elem_size)] = 0;
				}

			}

		}

	}

}

using namespace CUDA_BKE_namespace;

class CUDA_BKE : public GpuLabeling {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	CUDA_BKE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		//Mat1i init_labels;
		//d_img_labels_.download(init_labels);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();


		Merge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		//Mat1i block_info_final;
		//d_img_labels_.download(block_info_final);		

		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		//d_img_labels_.download(img_labels_);
		cudaDeviceSynchronize();
	}


private:
	void Alloc() {
		d_img_labels_.create(d_img_.size(), CV_32SC1);
	}

	void Dealloc() {
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
		grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//Mat1i init_labels;
		//d_block_labels_.download(init_labels);

		Merge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		//Mat1i block_info_final;
		//d_img_labels_.download(block_info_final);		

		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		cudaDeviceSynchronize();
	}

public:
	void PerformLabelingWithSteps()
	{
		perf_.start();
		Alloc();
		perf_.stop();
		double alloc_timing = perf_.last();

		perf_.start();
		AllScans();
		perf_.stop();
		perf_.store(Step(StepType::ALL_SCANS), perf_.last());

		perf_.start();
		Dealloc();
		perf_.stop();
		double dealloc_timing = perf_.last();

		perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
	}

};

REGISTER_LABELING(CUDA_BKE);