#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"


// Optimized Label Equivalence (OLE), enhanced by the use of texture memory as hinted in Asad2019


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


	__device__ unsigned int FindMinLabel(cudaTextureObject_t texObject, unsigned row, unsigned col, unsigned label) {

		unsigned int min = label;

		min = MinLabel(min, tex2D<unsigned int>(texObject, col - 1, row - 1));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col + 0, row - 1));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col + 1, row - 1));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col - 1, row + 0));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col + 1, row + 0));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col - 1, row + 1));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col + 0, row + 1));
		min = MinLabel(min, tex2D<unsigned int>(texObject, col + 1, row + 1));

		return min;
	}


	// Scan phase.
	// The pixel associated with current thread is given the minimum label of the neighbours.
	__global__ void Scan(cuda::PtrStepSzi labels, cudaTextureObject_t texObject, char *changes) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		// unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		unsigned label = tex2D<unsigned int>(texObject, col, row);

		if (label) {
			unsigned min_label = FindMinLabel(texObject, row, col, label);
			if (min_label < label) {
				labels[label - 1] = min(static_cast<unsigned int>(labels[label - 1]), min_label);
				*changes = 1;
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

	__device__ unsigned int FindMinLabelNotTex(cuda::PtrStepSzi labels, unsigned row, unsigned col, unsigned label, unsigned labels_index) {

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
	__global__ void ScanNotTex(cuda::PtrStepSzi labels, cudaTextureObject_t texObject, char* changes) {

		unsigned row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
		unsigned col = blockIdx.x * BLOCK_COLS + threadIdx.x;
		unsigned labels_index = row * (labels.step / labels.elem_size) + col;

		if (row < labels.rows && col < labels.cols) {

			unsigned label = labels[labels_index];

			if (label) {
				unsigned min_label = FindMinLabelNotTex(labels, row, col, label, labels_index);
				if (min_label < label) {
					labels[label - 1] = min(static_cast<unsigned int>(labels[label - 1]), min_label);
					*changes = 1;
				}
			}
		}
	}

}

class OLE_TEX : public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
	dim3 grid_size_;
	dim3 block_size_;
	char changes;
	char *d_changes;

public:
	OLE_TEX() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.size(), CV_32SC1);

		cudaMalloc(&d_changes, sizeof(char));

		// Workaround for 1D images, necessary for sm >= 70
		//void (*scan_kernel) (cuda::PtrStepSzi, cudaTextureObject_t, char*) = (d_img_.rows == 1 || d_img_.cols == 1) ? ScanNotTex : Scan;

		// Create Texture Object
		cudaChannelFormatDesc chFormatDesc = cudaCreateChannelDesc<unsigned int>();

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = d_img_labels_.data;
		resDesc.res.pitch2D.desc = chFormatDesc;
		resDesc.res.pitch2D.width = d_img_.cols;
		resDesc.res.pitch2D.height = d_img_.rows;
		resDesc.res.pitch2D.pitchInBytes = d_img_labels_.step;

		cudaTextureDesc texDesc = {
			{cudaAddressModeBorder, cudaAddressModeBorder},     // addressMode (fetches with out-of-range coordinates return 0)
			cudaFilterModePoint,                                // filterMode (do not interpolate and take the nearest value)
			cudaReadModeElementType,                            // readMode (do not convert to floating point, only for 8-bit and 16-bit integer formats)
			// other values are defaulted to 0
		};

		cudaTextureObject_t texObject;
		cudaCreateTextureObject(&texObject, &resDesc, &texDesc, nullptr);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		if (d_img_.rows == 1 || d_img_.cols == 1) {
			while (true) {
				changes = 0;
				cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

				ScanNotTex << <grid_size_, block_size_ >> > (d_img_labels_, texObject, d_changes);

				cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

				if (!changes)
					break;

				Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
			}
		}

		else {
			while (true) {
				changes = 0;
				cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

				Scan << <grid_size_, block_size_ >> > (d_img_labels_, texObject, d_changes);

				cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

				if (!changes)
					break;

				Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
			}
		}

		cudaDestroyTextureObject(texObject);

		cudaFree(d_changes);
		cudaDeviceSynchronize();
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
		// Create Texture Object
		cudaChannelFormatDesc chFormatDesc = cudaCreateChannelDesc<unsigned int>();

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = d_img_labels_.data;
		resDesc.res.pitch2D.desc = chFormatDesc;
		resDesc.res.pitch2D.width = d_img_.cols;
		resDesc.res.pitch2D.height = d_img_.rows;
		resDesc.res.pitch2D.pitchInBytes = d_img_labels_.step;

		cudaTextureDesc texDesc = {
			{cudaAddressModeBorder, cudaAddressModeBorder},     // addressMode (fetches with out-of-range coordinates return 0)
			cudaFilterModePoint,                                // filterMode (do not interpolate and take the nearest value)
			cudaReadModeElementType,                            // readMode (do not convert to floating point, only for 8-bit and 16-bit integer formats)
			// other values are defaulted to 0
		};

		cudaTextureObject_t texObject;
		cudaCreateTextureObject(&texObject, &resDesc, &texDesc, nullptr);

		grid_size_ = dim3((d_img_.cols + BLOCK_COLS - 1) / BLOCK_COLS, (d_img_.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
		block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

		Init << <grid_size_, block_size_ >> >(d_img_, d_img_labels_);

		if (d_img_.rows == 1 || d_img_.cols == 1) {
			while (true) {
				changes = 0;
				cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

				ScanNotTex << <grid_size_, block_size_ >> > (d_img_labels_, texObject, d_changes);

				cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

				if (!changes)
					break;

				Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
			}
		}

		else {
			while (true) {
				changes = 0;
				cudaMemcpy(d_changes, &changes, sizeof(char), cudaMemcpyHostToDevice);

				Scan << <grid_size_, block_size_ >> > (d_img_labels_, texObject, d_changes);

				cudaMemcpy(&changes, d_changes, sizeof(char), cudaMemcpyDeviceToHost);

				if (!changes)
					break;

				Analyze << <grid_size_, block_size_ >> > (d_img_labels_);
			}
		}

		cudaDestroyTextureObject(texObject);

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

REGISTER_LABELING(OLE_TEX);
