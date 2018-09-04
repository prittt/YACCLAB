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

namespace CUDA_UF_white_namespace {

	// Risale alla radice dell'albero a partire da un suo nodo n
	// Si potrebbe aggiungere la path compression
	__device__ unsigned Find(const int *s_buf, unsigned n) {

		while (s_buf[n] != n) {
			n = s_buf[n];
		}

		return n;
	}


	// Unisce gli alberi contenenti i nodi a e b, collegandone le radici
	// Al momento manca il controllo sulla concorrenza
	// Senza il controllo sulla concorrenza si possono verificare errori
	// --> Aggiungiamo il controllo
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

		// Senza controllo sulla concorrenza
		//unsigned ra = Find(s_buf, a);
		//unsigned rb = Find(s_buf, b);

		//if (ra < rb) {
		//	s_buf[rb] = ra;
		//}
		//else if (rb < ra) {
		//	s_buf[ra] = rb;
		//}

	}


	//Effettuo il controllo sui 4 vicini della maschera
	//Prova a sincronizzare dopo ogni vicino
	__global__ void LocalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_row = threadIdx.y;
		unsigned local_col = threadIdx.x;
		unsigned local_index = local_row * BLOCK_COLS + local_col;

		unsigned global_row = blockIdx.y * BLOCK_ROWS + local_row;
		unsigned global_col = blockIdx.x * BLOCK_COLS + local_col;
		unsigned img_index = global_row * img.step + global_col;

		__shared__ int s_buf[BLOCK_ROWS * BLOCK_COLS];
		__shared__ unsigned char s_img[BLOCK_ROWS * BLOCK_COLS];

		bool in_limits = (global_row < img.rows && global_col < img.cols);

		s_buf[local_index] = local_index;
		s_img[local_index] = in_limits ? img[img_index] : 0xFF;

		__syncthreads();

		unsigned char v = s_img[local_index];

		//unsigned char v = img[img_index];

		if (in_limits) {

			if (v == 1) {

				if (local_col > 0 && s_img[local_index - 1]) {
					Union(s_buf, local_index, local_index - 1);
				}


				if (local_row > 0 && s_img[local_index - BLOCK_COLS]) {
					Union(s_buf, local_index, local_index - BLOCK_COLS);
				}

			}

			else {
				if (local_row > 0 && s_img[local_index - BLOCK_COLS]) {
					//s_buf[local_index] = -1;
					//unsigned char upper = s_img[local_index - BLOCK_COLS];

					if (local_col > 0 && s_img[local_index - 1]) {
						Union(s_buf, local_index - 1, local_index - BLOCK_COLS);
					}


					if (local_col < BLOCK_COLS - 1 && s_img[local_index + 1]) {
						Union(s_buf, local_index + 1, local_index - BLOCK_COLS);
					}
				}

			}

			//	if (v == 1) {

			//		if (local_col > 0 && img[img_index - 1] == 1) {
			//			Union(s_buf, local_index, local_index - 1);
			//		}


			//		if (local_row > 0 && img[img_index - img.step] == 1) {
			//			Union(s_buf, local_index, local_index - BLOCK_COLS);
			//		}

			//	}

			//	else {
			//		if (local_row > 0) {
			//			//s_buf[local_index] = -1;
			//			unsigned char upper = img[img_index - img.step];

			//			if (local_col > 0 && img[img_index - 1] == upper) {
			//				Union(s_buf, local_index - 1, local_index - BLOCK_COLS);
			//			}


			//			if (local_col < BLOCK_COLS - 1 && img[img_index + 1] == upper) {
			//				Union(s_buf, local_index + 1, local_index - BLOCK_COLS);
			//			}
			//		}

			//	}

		}

		__syncthreads();

		if (in_limits) {
			unsigned f = Find(s_buf, local_index);
			unsigned f_row = f / BLOCK_COLS;
			unsigned f_col = f % BLOCK_COLS;
			unsigned global_f = (blockIdx.y * BLOCK_ROWS + f_row) * (labels.step / sizeof(int)) + (blockIdx.x * BLOCK_COLS + f_col);
			labels.data[global_row * labels.step / sizeof(int) + global_col] = global_f;		// Non c'è distinzione tra background e foreground
		}
	}


	__global__ void GlobalMerge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

		unsigned local_row = threadIdx.y;
		unsigned local_col = threadIdx.x;

		unsigned global_row = blockIdx.y * BLOCK_ROWS + local_row;
		unsigned global_col = blockIdx.x * BLOCK_COLS + local_col;
		unsigned img_index = global_row * img.step + global_col;
		unsigned labels_index = global_row * (labels.step / sizeof(int)) + global_col;

		bool in_limits = (global_row < img.rows && global_col < img.cols);

		unsigned char v = img[img_index];

		if (in_limits) {

			if (v == 1) {

				if (global_col > 0 && local_col == 0 && img[img_index - 1] == 1) {
					Union(labels.data, labels_index, labels_index - 1);
				}

				if (global_row > 0 && local_row == 0 && img[img_index - img.step] == 1) {
					Union(labels.data, labels_index, labels_index - labels.step / sizeof(int));
				}

			}

			else {

				if (global_row > 0 && img[img_index - img.step]) {
					// unsigned char upper = img[img_index - img.step];

					if (global_col > 0 && (local_row == 0 || local_col == 0) && img[img_index - 1]) {
						Union(labels.data, labels_index - 1, labels_index - labels.step / sizeof(int));
					}

					if ((global_col < img.cols - 1) && (local_row == 0 || local_col == BLOCK_COLS - 1) && img[img_index + 1]) {
						Union(labels.data, labels_index + 1, labels_index - labels.step / sizeof(int));
					}
				}
			}

		}
	}


	__global__ void PathCompression(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, cuda::PtrStepSzi labels_definitive) {

		unsigned global_row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned global_col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = global_row * (labels.step / sizeof(int)) + global_col;

		if (global_row < labels.rows && global_col < labels.cols) {
			unsigned char val = img[global_row * img.step + global_col];

			labels_definitive[labels_index] = (val == 0) ? 0 : Find(labels.data, labels_index) + 1;
		}
	}

}

using namespace CUDA_UF_white_namespace;

class CUDA_UF_white : public GpuLabeling {
private:
	cuda::GpuMat d_labels_;
	dim3 grid_size_;
	dim3 block_size_;

public:
	CUDA_UF_white() {}

	void PerformLabeling() {

		// Alloc 
		// img_labels_ = cv::Mat1i(img_.size());
		d_labels_.create(img_.size(), CV_32SC1);
		d_img_labels_.create(img_.size(), CV_32SC1);
		grid_size_ = dim3((img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_.x = BLOCK_COLS;
		block_size_.y = BLOCK_ROWS;
		block_size_.z = 1;

		// Memory transfer Host -> Device
		// d_img_.upload(img_);

		// Phase 1
		// Etichetta i pixel localmente al blocco		
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_labels_);

		// Immagine di debug della prima fase
		//cuda::GpuMat d_local_labels;
		//d_local_labels.create(img_.size(), CV_32SC1);
		//PathCompression << <grid_size_, block_size_ >> > (d_img_, d_labels_, d_local_labels);
		//Mat1i local_labels(img_.size());
		//d_local_labels.download(local_labels);

		// Phase 2
		// Collega tra loro gli alberi union-find dei diversi blocchi
		GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_labels_);

		// Immagine di debug della seconda fase
		//cuda::GpuMat d_global_labels;
		//d_global_labels.create(img_.size(), CV_32SC1);
		//PathCompression << <grid_size_, block_size_ >> > (d_img_, d_labels_, d_global_labels);
		//Mat1i global_labels(img_.size());
		//d_global_labels.download(global_labels);

		// Phase 3
		// Collassa gli alberi union-find sulle radici
		PathCompression << <grid_size_, block_size_ >> > (d_img_, d_labels_, d_img_labels_);

		cudaDeviceSynchronize();

		// Memory transfer Device -> Host
		// d_img_labels_.download(img_labels_);

		// GPU dealloc
		// d_img_.release();
		d_labels_.release();
		// d_img_labels_.release();
	}


private:
	// Provo a contare solamente l'allocazione delle strutture dati in Gpu ($)
	double Alloc() {
		// Memory allocation for the output image
		//perf_.start();
		// img_labels_ = cv::Mat1i(img_.size());
		//memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
		//perf_.stop();
		//double t = perf_.last();
		//perf_.start();
		//memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
		//perf_.stop();
		//double ma_t = t - perf_.last();
		//perf_.start();
		// Memory allocation for GPU data structures
		// d_img_.create(img_.size(), CV_8UC1);
		/*$*/ perf_.start();
		d_labels_.create(img_.size(), CV_32SC1);
		d_img_labels_.create(img_.size(), CV_32SC1);
		perf_.stop();
		// Initialization of class variables
		// Metti dei define al posto di questi
		grid_size_.x = (img_.cols + BLOCK_COLS - 1) / BLOCK_COLS;
		grid_size_.y = (img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
		grid_size_.z = 1;
		block_size_.x = BLOCK_COLS;
		block_size_.y = BLOCK_ROWS;
		block_size_.z = 1;
		// perf_.stop();
		// return perf_.last() + ma_t;
		/*$*/ return perf_.last();
	}

	// $
	double Dealloc() {
		// d_img_.release();
		perf_.start();
		d_labels_.release();
		perf_.stop();
		// d_img_labels_.release();
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
		LocalMerge << <grid_size_, block_size_ >> >(d_img_, d_labels_);
		cudaDeviceSynchronize();
	}

	void GlobalScan() {
		GlobalMerge << <grid_size_, block_size_ >> > (d_img_, d_labels_);
		PathCompression << <grid_size_, block_size_ >> > (d_img_, d_labels_, d_img_labels_);
		cudaDeviceSynchronize();
	}

public:
	void PerformLabelingWithSteps()
	{
		double alloc_timing = Alloc();

		// double transfer_timing = MemoryTransferHostToDevice();

		perf_.start();
		LocalScan();
		perf_.stop();
		perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

		perf_.start();
		GlobalScan();
		perf_.stop();
		perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

		//perf_.start();
		//MemoryTransferDeviceToHost();
		//perf_.stop();
		//transfer_timing += perf_.last();

		double dealloc_timing = Dealloc();

		perf_.store(Step(StepType::ALLOC_DEALLOC), alloc_timing + dealloc_timing /*+ transfer_timing */);

	}

};

REGISTER_LABELING(CUDA_UF_white);

