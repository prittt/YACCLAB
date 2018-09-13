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


// Algorithm itself has good performances, but memory allocation is a problem.
// I will try to reduce it.
namespace CUDA_BE_namespace {

	// Only use it with unsigned numeric types
	template <typename T>
	__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
		return (bitmap >> pos) & 1;
	}

	__device__ __forceinline__ void SetBit(unsigned char &bitmap, unsigned char pos) {
		bitmap |= (1 << pos);
	}

	__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, unsigned char *last_pixel) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned P0 = 0x777;
			unsigned P = 0;

			if (img[img_index]) {
				P |= P0;
			}

			if (col + 1 < img.cols) {

				if (img[img_index + 1]) {
					P |= (P0 << 1);
				}

				if (row + 1 < img.rows && img[img_index + img.step + 1]) {
					P |= (P0 << 5);
				}

			}

			if (row + 1 < img.rows) {

				if (img[img_index + img.step]) {
					P |= (P0 << 4);
				}

			}

			if (col == 0) {
				P &= 0xEEEE;
			}
			if (col + 1 >= img.cols) {
				P &= 0x3333;
			}
			else if (col + 2 >= img.cols) {
				P &= 0x7777;
			}

			if (row == 0) {
				P &= 0xFFF0;
			}
			if (row + 1 >= img.rows) {
				P &= 0xFF;
			}
			else if (row + 2 >= img.rows) {
				P &= 0xFFF;
			}

			// P is now ready to be used to find neighbour blocks (or it should be)
			// P value avoids range errors

			unsigned char conn_bitmask = 0;

			if (P > 0) {

				labels[labels_index] = labels_index + 1;

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
				labels[labels_index] = 0;
			}

			// Connection bitmask is stored in the north-east int of every block
			// If columns are odd, in the last column, it's stored in the south-west of every block instead
			// If columns are odd and rows are odd, it's stored in *last_pixel
			if (col + 1 < labels.cols)
				labels[labels_index + 1] = conn_bitmask;
			else if (row + 1 < labels.rows)
				labels[labels_index + (labels.step / labels.elem_size)] = conn_bitmask;
			else
				*last_pixel = conn_bitmask;

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
			min = MinLabel(min, labels.data[labels_index - 2 * (labels.step / labels.elem_size) - 2]);
		}

		if (HasBit(neighbours, 1)) {
			min = MinLabel(min, labels.data[labels_index - 2 * (labels.step / labels.elem_size)]);
		}

		if (HasBit(neighbours, 2)) {
			min = MinLabel(min, labels.data[labels_index - 2 * (labels.step / labels.elem_size) + 2]);
		}

		if (HasBit(neighbours, 3)) {
			min = MinLabel(min, labels.data[labels_index - 2]);
		}

		if (HasBit(neighbours, 4)) {
			min = MinLabel(min, labels.data[labels_index + 2]);
		}

		if (HasBit(neighbours, 5)) {
			min = MinLabel(min, labels.data[labels_index + 2 * (labels.step / labels.elem_size) - 2]);
		}

		if (HasBit(neighbours, 6)) {
			min = MinLabel(min, labels.data[labels_index + 2 * (labels.step / labels.elem_size)]);
		}

		if (HasBit(neighbours, 7)) {
			min = MinLabel(min, labels.data[labels_index + 2 * (labels.step / labels.elem_size) + 2]);
		}

		return min;
	}


	__global__ void Scan(cuda::PtrStepSzi labels, unsigned char *changes, unsigned char *last_pixel) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned char neighbours;

			if (col + 1 < labels.cols)
				neighbours = labels[labels_index + 1];
			else if (row + 1 < labels.rows)
				neighbours = labels[labels_index + (labels.step / labels.elem_size)];
			else
				neighbours = *last_pixel;

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


	__global__ void Analyze(cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
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


	__global__ void FinalLabeling(cuda::PtrStepSzi labels, const cuda::PtrStepSzb img) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;
		unsigned img_index = row * (img.step / img.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned int label = labels[labels_index];

			if (img[img_index]) {}
			// labels[labels_index] = label;
			else {
				labels[labels_index] = 0;
			}

			if (col + 1 < labels.cols) {
				if (img[img_index + 1])
					labels[labels_index + 1] = label;
				else {
					labels[labels_index + 1] = 0;
				}

				if (row + 1 < labels.rows) {
					if (img[img_index + img.step + 1])
						labels[labels_index + (labels.step / labels.elem_size) + 1] = label;
					else {
						labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
					}
				}
			}

			if (row + 1 < labels.rows) {
				if (img[img_index + img.step])
					labels[labels_index + (labels.step / labels.elem_size)] = label;
				else {
					labels[labels_index + (labels.step / labels.elem_size)] = 0;
				}
			}

		}

	}

}

using namespace CUDA_BE_namespace;

class CUDA_BE : public GpuLabeling {
private:
	dim3 grid_size_;
	dim3 block_size_;
	char changes_;
	unsigned char *d_changes_;
	bool d_changed_alloc_ = false;
	unsigned char *last_pixel_;

public:
	CUDA_BE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		if (d_img_.rows == 1) {
			if (d_img_.cols == 1) {
				d_img_.convertTo(d_img_labels_, CV_32SC1);
				return;
			}
			else if (d_img_.cols % 2) {
				cudaMalloc(&d_changes_, sizeof(unsigned char) * 2);
				d_changed_alloc_ = true;
				last_pixel_ = d_changes_ + 1;
			}
			else {
				cudaMalloc(&d_changes_, sizeof(unsigned char));
				d_changed_alloc_ = true;
			}
		}
		else if (d_img_.cols == 1) {
			if (d_img_.rows % 2) {
				cudaMalloc(&d_changes_, sizeof(unsigned char) * 2);
				d_changed_alloc_ = true;
				last_pixel_ = d_changes_ + 1;
			}
			else {
				cudaMalloc(&d_changes_, sizeof(unsigned char));
				d_changed_alloc_ = true;
			}
		}
		else {
			d_changes_ = d_img_labels_.data + d_img_labels_.step;
			last_pixel_ = d_img_labels_.data + d_img_labels_.step + sizeof(unsigned int);
		}

		grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_, last_pixel_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//Mat1i init_labels;
		//d_block_labels_.download(init_labels);

		while (true) {
			changes_ = 0;
			cudaMemcpy(d_changes_, &changes_, sizeof(unsigned char), cudaMemcpyHostToDevice);

			Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes_, last_pixel_);

			cudaMemcpy(&changes_, d_changes_, sizeof(unsigned char), cudaMemcpyDeviceToHost);

			if (!changes_)
				break;

			Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		}

		//Mat1i block_info_final;
		//d_img_labels_.download(block_info_final);		

		/*if ((img_.rows % 2) && (img_.cols % 2))
		LastPixel << <1, 1 >> > (d_img_labels_, last_pixel_);
		*/
		FinalLabeling << <grid_size_, block_size_ >> >(d_img_labels_, d_img_);

		// d_img_labels_.download(img_labels_);

		if (d_changed_alloc_)
			cudaFree(d_changes_);
	}


private:
	bool Alloc() {
		d_img_labels_.create(d_img_.size(), CV_32SC1);
		if (d_img_.rows == 1) {
			if (d_img_.cols == 1) {
				d_img_.convertTo(d_img_labels_, CV_32SC1);
				return true;
			}
			else if (d_img_.cols % 2) {
				cudaMalloc(&d_changes_, sizeof(unsigned char) * 2);
				last_pixel_ = d_changes_ + 1;
			}
			else {
				cudaMalloc(&d_changes_, sizeof(unsigned char));
			}
		}
		else if (d_img_.cols == 1) {
			if (d_img_.rows % 2) {
				cudaMalloc(&d_changes_, sizeof(unsigned char) * 2);
				last_pixel_ = d_changes_ + 1;
			}
			else {
				cudaMalloc(&d_changes_, sizeof(unsigned char));
			}
		}
		else {
			d_changes_ = d_img_labels_.data + d_img_labels_.step;
			last_pixel_ = d_img_labels_.data + d_img_labels_.step + sizeof(unsigned int);
		}
		return false;
	}

	void Dealloc() {
		if (d_changed_alloc_)
			cudaFree(d_changes_);
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
		last_pixel_ = d_changes_ + 1;

		grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_, last_pixel_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//Mat1i init_labels;
		//d_block_labels_.download(init_labels);

		while (true) {
			changes_ = 0;

			cudaMemcpy(d_changes_, &changes_, sizeof(unsigned char), cudaMemcpyHostToDevice);

			Scan << <grid_size_, block_size_ >> > (d_img_labels_, d_changes_, last_pixel_);

			cudaMemcpy(&changes_, d_changes_, sizeof(unsigned char), cudaMemcpyDeviceToHost);

			if (!changes_)
				break;

			Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		}

		//Mat1i block_labels;
		//d_block_labels_.download(block_labels);

		FinalLabeling << <grid_size_, block_size_ >> >(d_img_labels_, d_img_);

		cudaDeviceSynchronize();
	}

public:
	void PerformLabelingWithSteps()
	{
		perf_.start();
		bool done = Alloc();
		perf_.stop();
		double alloc_timing = perf_.last();

		if (!done) {
			perf_.start();
			AllScans();
			perf_.stop();
			perf_.store(Step(StepType::ALL_SCANS), perf_.last());
		}

		perf_.start();
		Dealloc();
		perf_.stop();
		double dealloc_timing = perf_.last();

		perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing);
	}

};

REGISTER_LABELING(CUDA_BE);

