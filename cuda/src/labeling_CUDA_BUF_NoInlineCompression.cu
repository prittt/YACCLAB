#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

// Il minimo per entrambi è 4
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {

	// Only use it with unsigned numeric types
	template <typename T>
	__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
		return (bitmap >> pos) & 1;
	}

	//__device__ __forceinline__ void SetBit(unsigned char &bitmap, unsigned char pos) {
	//	bitmap |= (1 << pos);
	//}

	// Risale alla radice dell'albero a partire da un suo nodo n
	__device__ unsigned Find(const int *s_buf, unsigned n) {
		while (s_buf[n] != n) {
			n = s_buf[n];
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
				int old = atomicMin(s_buf + b, a);
				done = (old == b);
				b = old;
			}
			else if (b < a) {
				int old = atomicMin(s_buf + a, b);
				done = (old == a);
				a = old;
			}
			else {
				done = true;
			}

		} while (!done);

	}


	__global__ void InitLabeling(cuda::PtrStepSzi labels) {
		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			labels[labels_index] = labels_index;
		}
	}

	__global__ void Merge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned P = 0;

			char buffer[4];
			*(reinterpret_cast<int*>(buffer)) = 0;

			if (col + 1 < img.cols) {
				// This does not depend on endianness
				*(reinterpret_cast<int16_t*>(buffer)) = *(reinterpret_cast<int16_t*>(img.data + img_index));

				if (row + 1 < img.rows) {
					*(reinterpret_cast<int16_t*>(buffer + 2)) = *(reinterpret_cast<int16_t*>(img.data + img_index + img.step));
				}
			}
			else {
				buffer[0] = img.data[img_index];

				if (row + 1 < img.rows) {
					buffer[2] = img.data[img_index + img.step];
				}
			}

			if (buffer[0]) {
				P |= 0x777;
			}
			if (buffer[1]) {
				P |= (0x777 << 1);
			}
			if (buffer[2]) {
				P |= (0x777 << 4);
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
			//else if (row + 2 >= img.rows) {                                           // controlla che vada anche senza
			//	P &= 0xFFF;
			//}

			// P is now ready to be used to find neighbour blocks (or it should be)
			// P value avoids range errors

			if (P > 0) {

				if (HasBit(P, 0) && img.data[img_index - img.step - 1]) {
					Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) - 2);
				}

				if ((HasBit(P, 1) && img.data[img_index - img.step]) || (HasBit(P, 2) && img.data[img_index + 1 - img.step])) {
					Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));
				}

				if (HasBit(P, 3) && img.data[img_index + 2 - img.step]) {
					Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2);
				}

				if ((HasBit(P, 4) && img.data[img_index - 1]) || (HasBit(P, 8) && img.data[img_index + img.step - 1])) {
					Union(labels.data, labels_index, labels_index - 2);
				}
			}
		}
	}

	__global__ void Compression(cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			//unsigned label = labels.data[labels_index];
			//if (label < labels_index) {
				labels[labels_index] = Find(labels.data, labels_index);
			//}
		}
	}


	__global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
		unsigned col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;
		unsigned img_index = row * (img.step / img.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned int label = labels[labels_index] + 1;

			if (img.data[img_index])
				labels[labels_index] = label;
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

	}

}

class CUDA_BUF_NoInlineCompression : public GpuLabeling2D<CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	CUDA_BUF_NoInlineCompression() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((((d_img_.cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((d_img_.rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		InitLabeling << <grid_size_, block_size_ >> > (d_img_labels_);

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

		// d_img_labels_.download(img_labels_);
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


		InitLabeling << <grid_size_, block_size_ >> > (d_img_labels_);

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

REGISTER_LABELING(CUDA_BUF_NoInlineCompression);
