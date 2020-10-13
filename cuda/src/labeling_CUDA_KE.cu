#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"


#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {


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



	// Init phase.
	// Labels start at value 1, to differentiate them from background, that has value 0.
	__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned img_index = row * img.step + col;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < img.rows && col < img.cols) {
			
			if (img.data[img_index]) {

				if (row > 0 && img.data[img_index - img.step]) {
					labels.data[labels_index] = labels_index - (labels.step / labels.elem_size) + 1;
				}

				else if (row > 0 && col > 0 && img.data[img_index - img.step - 1]) {
					labels.data[labels_index] = labels_index - (labels.step / labels.elem_size);
				}

				else if (row > 0 && col < img.cols - 1 && img.data[img_index - img.step + 1]) {
					labels.data[labels_index] = labels_index - (labels.step / labels.elem_size) + 2;
				}

				else if (col > 0 && img.data[img_index - 1]) {
					labels.data[labels_index] = labels_index;
				}

				else {
					labels.data[labels_index] = labels_index + 1;
				}
			}
			else {
				labels.data[labels_index] = 0;
			}
		}
	}


	//// Analysis phase.
	//// The pixel associated with current thread is given the minimum label of the neighbours.
	//__global__ void Analyze(cuda::PtrStepSzi labels) {

	//	unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
	//	unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
	//	unsigned labels_index = row * (labels.step / labels.elem_size) + col;

	//	if (row < labels.rows && col < labels.cols) {

	//		unsigned label = labels[labels_index];

	//		if (label) {								// Performances are the same as the paper variant

	//			unsigned index = labels_index;

	//			while (label - 1 != index) {
	//				index = label - 1;
	//				label = labels[index];
	//			}

	//			labels[labels_index] = label;
	//		}
	//	}
	//}

	__global__ void Compression(cuda::PtrStepSzi labels) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {
			unsigned label = labels.data[labels_index];
			if (label) {
				labels.data[labels_index] = Find_label(labels.data, labels_index, label) + 1;
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

class KE : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;

public:
	KE() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		/*Mat1i init_labels;
		d_img_labels_.download(init_labels);*/
		//init_labels.release();

		//Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		//Mat1i analyze_labels;
		//d_img_labels_.download(analyze_labels);
		//analyze_labels.release();

		Reduce << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);
		//Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		Compression << <grid_size_, block_size_ >> > (d_img_labels_);


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

	void AllScans() {
		d_img_labels_.create(d_img_.size(), CV_32SC1);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		//Mat1i init_labels;
		//d_img_labels_.download(init_labels);
		// init_labels.release();

		//Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

		//Mat1i analyze_labels;
		//d_img_labels_.download(analyze_labels);
		// analyze_labels.release();

		Reduce << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);
		//Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
		Compression << <grid_size_, block_size_ >> > (d_img_labels_);

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

REGISTER_LABELING(KE);

