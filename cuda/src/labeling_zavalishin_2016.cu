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


#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;


// Algorithm itself has good performances, but memory allocation is a problem.
namespace {

	// Only use it with unsigned numeric types
	template <typename T>
	__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
		return (bitmap >> pos) & 1;
	}

	__device__ __forceinline__ void SetBit(unsigned char &bitmap, unsigned char pos) {
		bitmap |= (1 << pos);
	}

	// Init phase.
	// Labels start at value 1.
	__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzb block_conn, cuda::PtrStepSzi block_labels) {

		unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned img_index = 2 * row * img.step + 2 * col;
		unsigned conn_index = row * (block_conn.step / block_conn.elem_size) + col;
		unsigned labels_index = row * (block_labels.step / block_labels.elem_size) + col;

		if (row < block_conn.rows && col < block_conn.cols) {

			unsigned P0 = 0x777;
			unsigned P = 0;

			if (img[img_index]) {
				P |= P0;
			}

			if (2 * col + 1 < img.cols) {

				if (img[img_index + 1]) {
					P |= (P0 << 1);
				}

				if (2 * row + 1 < img.rows && img[img_index + img.step + 1]) {
					P |= (P0 << 5);
				}

			}

			if (2 * row + 1 < img.rows) {

				if (img[img_index + img.step]) {
					P |= (P0 << 4);
				}

			}

			if (col == 0) {
				P &= 0xEEEE;
			}
			if (2 * col + 1 >= img.cols) {
				P &= 0x3333;
			}
			else if (2 * col + 2 >= img.cols) {
				P &= 0x7777;
			}

			if (row == 0) {
				P &= 0xFFF0;
			}
			if (2 * row + 1 >= img.rows) {
				P &= 0xFF;
			}
			else if (2 * row + 2 >= img.rows) {
				P &= 0xFFF;
			}
			
			// P is now ready to be used to find neighbour blocks (or it should be)
			// P value avoids range errors

			unsigned char conn_bitmask = 0;

			if (P > 0) {

				block_labels[labels_index] = labels_index + 1;

				if (HasBit(P, 0) && img[img_index - img.step - 1]) {
					SetBit(conn_bitmask, 0);
				}

				if ((HasBit(P, 1) && img[img_index - img.step]) || (HasBit(P, 2) && img[img_index + 1 - img.step])) {
					SetBit(conn_bitmask, 1);
				}

				if (HasBit(P, 3) && img[img_index + 2 - img.step]) {
					SetBit(conn_bitmask, 2);
				}

				if ((HasBit(P, 4) && img[img_index - 1]) || (HasBit(P, 8) && img[img_index + img.step - 1])) {
					SetBit(conn_bitmask, 3);
				}

				if ((HasBit(P, 7) && img[img_index + 2]) || (HasBit(P, 11) && img[img_index + img.step + 2])) {
					SetBit(conn_bitmask, 4);
				}

				if (HasBit(P, 12) && img[img_index + 2 * img.step - 1]) {
					SetBit(conn_bitmask, 5);
				}

				if ((HasBit(P, 13) && img[img_index + 2 * img.step]) || (HasBit(P, 14) && img[img_index + 2 * img.step + 1])) {
					SetBit(conn_bitmask, 6);
				}

				if (HasBit(P, 15) && img[img_index + 2 * img.step + 2]) {
					SetBit(conn_bitmask, 7);
				}
 			}

			else {
				block_labels[labels_index] = 0;
			}

			block_conn[conn_index] = conn_bitmask;

		}
	}


	__global__ void ExpandConnections(const cuda::PtrStepSzb connections, cuda::PtrStepSzb expansion) {

		unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned conn_index = row * (connections.step / connections.elem_size) + col;
		unsigned exp_index = 3 * row * (expansion.step / expansion.elem_size) + 3 * col;

		if (row < connections.rows && col < connections.cols) {

			expansion[exp_index + (expansion.step / expansion.elem_size) + 1] = 2;

			unsigned char neighbours = connections[conn_index];

			if (HasBit(neighbours, 0)) {
				expansion[exp_index] = 1;
			}
			else {
				expansion[exp_index] = 0;
			}

			if (HasBit(neighbours, 1)) {
				expansion[exp_index + 1] = 1;
			}
			else {
				expansion[exp_index + 1] = 0;
			}

			if (HasBit(neighbours, 2)) {
				expansion[exp_index + 2] = 1;
			}
			else {
				expansion[exp_index + 2] = 0;
			}

			if (HasBit(neighbours, 3)) {
				expansion[exp_index + (expansion.step / expansion.elem_size)] = 1;
			}
			else {
				expansion[exp_index + (expansion.step / expansion.elem_size)] = 0;
			}

			if (HasBit(neighbours, 4)) {
				expansion[exp_index + (expansion.step / expansion.elem_size) + 2] = 1;
			}
			else {
				expansion[exp_index + (expansion.step / expansion.elem_size) + 2] = 0;
			}

			if (HasBit(neighbours, 5)) {
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size)] = 1;
			}
			else {
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size)] = 0;
			}

			if (HasBit(neighbours, 6)) {
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 1] = 1;
			}																 
			else {															 
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 1] = 0;
			}

			if (HasBit(neighbours, 7)) {
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 2] = 1;
			}
			else {
				expansion[exp_index + 2 * (expansion.step / expansion.elem_size) + 2] = 0;
			}
		}
	}


	__device__ unsigned int MinLabel(unsigned l1, unsigned l2) {
		if (l1 && l2)
			return min(l1, l2);
		else
			return l1;
	}


	__device__ unsigned int FindMinLabel(cuda::PtrStepSzi labels, unsigned char neighbours, unsigned label, unsigned labels_index) {

		unsigned int min = label;

		if (HasBit(neighbours, 0)) {
			min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size) - 1]);
		}

		if (HasBit(neighbours, 1)) {
			min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size)]);
		}

		if (HasBit(neighbours, 2)) {
			min = MinLabel(min, labels.data[labels_index - (labels.step / labels.elem_size) + 1]);
		}

		if (HasBit(neighbours, 3)) {
			min = MinLabel(min, labels.data[labels_index - 1]);
		}

		if (HasBit(neighbours, 4)) {
			min = MinLabel(min, labels.data[labels_index + 1]);
		}

		if (HasBit(neighbours, 5)) {
			min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size) - 1]);
		}

		if (HasBit(neighbours, 6)) {
			min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size)]);
		}

		if (HasBit(neighbours, 7)) {
			min = MinLabel(min, labels.data[labels_index + (labels.step / labels.elem_size) + 1]);
		}

		return min;
	}


	// Scan phase.
	// The pixel associated with current thread is given the minimum label of the neighbours.
	__global__ void Scan(cuda::PtrStepSzi labels, cuda::PtrStepSzb connections, char *changes) {

		unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;
		unsigned connections_index = row * (connections.step / connections.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			
			unsigned char neighbours = connections[connections_index];

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


	// Analysis phase.
	// The pixel associated with current thread is given the minimum label of the neighbours.
	__global__ void Analyze(cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
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

	// Final Labeling phase
	// Assigns every pixel of 2x2 blocks the block label
	__global__ void FinalLabeling(cuda::PtrStepSzi block_labels, cuda::PtrStepSzi labels, const cuda::PtrStepSzb img) {

		unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned blocks_index = row * (block_labels.step / block_labels.elem_size) + col;
		unsigned labels_index = 2 * row * (labels.step / labels.elem_size) + 2 * col;
		unsigned img_index = 2 * row * (img.step / img.elem_size) + 2 * col;

		if (row < block_labels.rows && col < block_labels.cols) {

			unsigned int label = block_labels[blocks_index];

			if (img[img_index])
				labels[labels_index] = label;
			else {
				labels[labels_index] = 0;
			}

			if (2 * col + 1 < labels.cols) {
				if (img[img_index + 1])
					labels[labels_index + 1] = label;
				else {
					labels[labels_index + 1] = 0;
				}

				if (2 * row + 1 < labels.rows) {
					if (img[img_index + img.step + 1]) 
						labels[labels_index + (labels.step / labels.elem_size) + 1] = label;
					else {
						labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
					}
				}
			}

			if (2 * row + 1 < labels.rows) {
				if (img[img_index + img.step])
					labels[labels_index + (labels.step / labels.elem_size)] = label;
				else {
					labels[labels_index + (labels.step / labels.elem_size)] = 0;
				}
			}

		}

	}

}

class BE : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;
	char changes;
	char *d_changes;

	cuda::GpuMat d_connections_;
	cuda::GpuMat d_block_labels_;

public:
	BE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);
		
		// Extra structures that I would gladly do without
		d_connections_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_8UC1);
		d_block_labels_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_32SC1);

		cudaMalloc(&d_changes, sizeof(char));

		grid_size_ = dim3((d_connections_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_connections_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_connections_, d_block_labels_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

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

		//Mat1i block_labels;
		//d_block_labels_.download(block_labels);

		FinalLabeling << <grid_size_, block_size_ >> >(d_block_labels_, d_img_labels_, d_img_);

		//d_img_labels_.download(img_labels_);

        cudaDeviceSynchronize();

		cudaFree(d_changes);
		d_connections_.release();
		d_block_labels_.release();
	}

	void PerformLabelingBlocksize(int x, int y, int z) override {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		// Extra structures that I would gladly do without
		d_connections_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_8UC1);
		d_block_labels_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_32SC1);

		cudaMalloc(&d_changes, sizeof(char));

		grid_size_ = dim3((d_connections_.cols + x - 1) / x, (d_connections_.rows + y - 1) / y, 1);
		block_size_ = dim3(x, y, 1);

		BLOCKSIZE_KERNEL(Init, grid_size_, block_size_, 0, d_img_, d_connections_, d_block_labels_)

		while (true) {
			changes = 0;
			cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

			BLOCKSIZE_KERNEL(Scan, grid_size_, block_size_, 0, d_block_labels_, d_connections_, d_changes)

			cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

			if (!changes)
				break;

			BLOCKSIZE_KERNEL(Analyze, grid_size_, block_size_, 0, d_block_labels_)
		}

		BLOCKSIZE_KERNEL(FinalLabeling, grid_size_, block_size_, 0, d_block_labels_, d_img_labels_, d_img_)

		cudaFree(d_changes);
		d_connections_.release();
		d_block_labels_.release();
	}


private:
	double Alloc() {
		perf_.start();
		d_img_labels_.create(d_img_.size(), CV_32SC1);
		d_connections_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_8UC1);
		d_block_labels_.create((d_img_.rows + 1) / 2, (d_img_.cols + 1) / 2, CV_32SC1);
		cudaMalloc(&d_changes, sizeof(char));
		perf_.stop();
		return perf_.last();
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
		grid_size_ = dim3((d_connections_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_connections_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_connections_, d_block_labels_);
		// La Init esplode
		// Controlla che cosa contiene connections
		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//assert(cudaDeviceSynchronize() == cudaSuccess);

		//Immagine di debug della inizializzazione
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

		FinalLabeling << <grid_size_, block_size_ >> >(d_block_labels_, d_img_labels_, d_img_);

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

REGISTER_LABELING(BE);

REGISTER_KERNELS(BE, Init, Scan, Analyze, FinalLabeling)