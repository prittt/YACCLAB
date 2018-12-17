#include <opencv2\cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "labeling_algorithms.h"
#include "register.h"

// Il minimo per entrambi è 4
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4


using namespace cv;

namespace {

	// Only use it with unsigned numeric types
	template <typename T>
	__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
		return (bitmap >> pos) & 1;
	}

	// Only use it with unsigned numeric types
	template <typename T>
	__device__ __forceinline__ void SetBit(T &bitmap, unsigned char pos) {
	    bitmap |= (1 << pos);
	}

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


	__global__ void InitLabeling(const cuda::PtrStepSz3b img, cuda::PtrStepSz3i labels, unsigned int *last_voxel_conn) {
		unsigned x = (blockIdx.x * BLOCK_X + threadIdx.x) * 2;
		unsigned y = (blockIdx.y * BLOCK_Y + threadIdx.y) * 2;
		unsigned z = (blockIdx.z * BLOCK_Z + threadIdx.z) * 2;
		unsigned img_index = z * (img.stepz / img.elem_size) + y * (img.stepy / img.elem_size) + x;
		unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

		if (x < labels.x && y < labels.y && z < labels.z) {

#define P0 0x77707770777

			unsigned long long P = 0L;

			if (img[img_index]) {
				P |= P0;
			}

			if (x + 1 < img.x) {

				if (img[img_index + 1]) {
					P |= (P0 << 1);
				}

				if (y + 1 < img.y && img[img_index + img.stepy / img.elem_size + 1]) {
					P |= (P0 << 5);
				}

			}

			if (y + 1 < img.y) {

				if (img[img_index + img.stepy / img.elem_size]) {
					P |= (P0 << 4);
				}

			}

			if (z + 1 < img.z) {
				if (img[img_index + img.stepz / img.elem_size]) {
					P |= (P0 << 16);
				}

				if (x + 1 < img.x) {

					if (img[img_index + img.stepz / img.elem_size + 1]) {
						P |= (P0 << 17);
					}

					//if (y + 1 < img.y && img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size + 1]) {
					//    P |= (P0 << 21);
					//}

				}

				if (y + 1 < img.y) {

					if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size]) {
						P |= (P0 << 20);
					}

				}
			}

#undef P0

			// checks on borders

			if (x == 0) {
				P &= 0xEEEEEEEEEEEEEEEE;
			}
			if (x + 1 >= img.x) {
				P &= 0x3333333333333333;
			}
			else if (x + 2 >= img.x) {
				P &= 0x7777777777777777;
			}

			if (y == 0) {
				P &= 0xFFF0FFF0FFF0FFF0;
			}
			if (y + 1 >= img.y) {
				P &= 0x00FF00FF00FF00FF;
			}
			else if (y + 2 >= img.y) {
				P &= 0x0FFF0FFF0FFF0FFF;
			}

			if (z == 0) {
				P &= 0xFFFFFFFFFFFF0000;
			}
			if (z + 1 >= img.z) {
				P &= 0x00000000FFFFFFFF;
			}
			//else if (z + 2 >= img.z) {
			//    P &= 0x0000FFFFFFFFFFFF;
			//}

			// P is now ready to be used to find neighbour blocks (or it should be)
			// P value avoids range errors

			unsigned int connections = 0;
			bool connected = false;

			if (P > 0) {

				// Lower plane
				unsigned char * plane_data = img.data + img_index - img.stepz;
				unsigned lower_plane_index = labels_index - 2 * (labels.stepz / labels.elem_size);

				if (HasBit(P, 0) && plane_data[0 - img.stepy - 1]) {
					if (connected) {
						SetBit(connections, 0);
					}
					else {
						labels[labels_index] = lower_plane_index - 2 * (labels.stepy / labels.elem_size) - 2;
						connected = true;
					}
				}

				if ((HasBit(P, 1) && plane_data[0 - img.stepy]) || (HasBit(P, 2) && plane_data[0 - img.stepy + 1])) {
					if (connected) {
						SetBit(connections, 1);
					}
					else {
						labels[labels_index] = lower_plane_index - 2 * (labels.stepy / labels.elem_size);
						connected = true;
					}
				}

				if (HasBit(P, 3) && plane_data[0 - img.stepy + 2]) {
					if (connected) {
						SetBit(connections, 2);
					}
					else {
						labels[labels_index] = lower_plane_index - 2 * (labels.stepy / labels.elem_size) + 2;
						connected = true;
					}
				}

				if ((HasBit(P, 4) && plane_data[-1]) || (HasBit(P, 8) && plane_data[img.stepy - 1])) {
					if (connected) {
						SetBit(connections, 3);
					}
					else {
						labels[labels_index] = lower_plane_index - 2;
						connected = true;
					}
				}

				if ((HasBit(P, 5) && plane_data[0]) || (HasBit(P, 6) && plane_data[1]) || (HasBit(P, 9) && plane_data[img.stepy]) || (HasBit(P, 10) && plane_data[img.stepy + 1])) {
					if (connected) {
						SetBit(connections, 4);
					}
					else {
						labels[labels_index] = lower_plane_index;
						connected = true;
					}
				}

				if ((HasBit(P, 7) && plane_data[2]) || (HasBit(P, 11) && plane_data[img.stepy + 2])) {
					if (connected) {
						SetBit(connections, 5);
					}
					else {
						labels[labels_index] = lower_plane_index + 2;
						connected = true;
					}
				}

				if (HasBit(P, 12) && plane_data[2 * img.stepy - 1]) {
					if (connected) {
						SetBit(connections, 6);
					}
					else {
						labels[labels_index] = lower_plane_index + 2 * (labels.stepy / labels.elem_size) - 2;
						connected = true;
					}
				}

				if ((HasBit(P, 13) && plane_data[2 * img.stepy]) || (HasBit(P, 14) && plane_data[2 * img.stepy + 1])) {
					if (connected) {
						SetBit(connections, 7);
					}
					else {
						labels[labels_index] = lower_plane_index + 2 * (labels.stepy / labels.elem_size);
						connected = true;
					}
				}

				if (HasBit(P, 15) && plane_data[2 * img.stepy + 2]) {
					if (connected) {
						SetBit(connections, 8);
					}
					else {
						labels[labels_index] = lower_plane_index + 2 * (labels.stepy / labels.elem_size) + 2;
						connected = true;
					}
				}

				// Current planes
				plane_data += img.stepz;

				if ((HasBit(P, 16) && plane_data[0 - img.stepy - 1]) || (HasBit(P, 32) && plane_data[img.stepz - img.stepy - 1])) {
					if (connected) {
						SetBit(connections, 9);
					}
					else {
						labels[labels_index] = labels_index - 2 * (labels.stepy / labels.elem_size) - 2;
						connected = true;
					}
				}

				if ((HasBit(P, 17) && plane_data[0 - img.stepy]) || (HasBit(P, 18) && plane_data[0 - img.stepy + 1]) || (HasBit(P, 33) && plane_data[img.stepz - img.stepy]) || (HasBit(P, 34) && plane_data[img.stepz - img.stepy + 1])) {
					if (connected) {
						SetBit(connections, 10);
					}
					else {
						labels[labels_index] = labels_index - 2 * (labels.stepy / labels.elem_size);
						connected = true;
					}
				}

				if ((HasBit(P, 19) && plane_data[0 - img.stepy + 2]) || (HasBit(P, 35) && plane_data[img.stepz - img.stepy + 2])) {
					if (connected) {
						SetBit(connections, 11);
					}
					else {
						labels[labels_index] = labels_index - 2 * (labels.stepy / labels.elem_size) + 2;
						connected = true;
					}
				}

				if ((HasBit(P, 20) && plane_data[-1]) || (HasBit(P, 24) && plane_data[img.stepy - 1]) || (HasBit(P, 36) && plane_data[img.stepz - 1]) || (HasBit(P, 40) && plane_data[img.stepz + img.stepy - 1])) {
					if (connected) {
						SetBit(connections, 12);
					}
					else {
						labels[labels_index] = labels_index - 2;
						connected = true;
					}
				}

			}

			if (!connected) {
				labels[labels_index] = labels_index;
			}

			// Write connections
			if (x + 1 < labels.x) {
				labels[labels_index + 1] = connections;
			}
			else if (y + 1 < labels.y) {
				labels[labels_index + labels.stepy / labels.elem_size] = connections;
			}
			else if (z + 1 < labels.z) {
				labels[labels_index + labels.stepz / labels.elem_size] = connections;
			}
			else {
				*last_voxel_conn = connections;
			}
		}
	}

	__global__ void Merge(cuda::PtrStepSz3i labels, unsigned int *last_voxel_conn) {

		unsigned x = (blockIdx.x * BLOCK_X + threadIdx.x) * 2;
		unsigned y = (blockIdx.y * BLOCK_Y + threadIdx.y) * 2;
		unsigned z = (blockIdx.z * BLOCK_Z + threadIdx.z) * 2;
		unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

		if (x < labels.x && y < labels.y && z < labels.z) {

			unsigned int connections;

			if (x + 1 < labels.x) {
				connections = labels[labels_index + 1];
			}
			else if (y + 1 < labels.y) {
				connections = labels[labels_index + labels.stepy / labels.elem_size];
			}
			else if (z + 1 < labels.z) {
				connections = labels[labels_index + labels.stepz / labels.elem_size];
			}
			else {
				connections = *last_voxel_conn;
			}

			// Lower plane
			unsigned lower_plane_index = labels_index - 2 * (labels.stepz / labels.elem_size);

			if (HasBit(connections, 0)) {
				Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size + 1));
			}

			if (HasBit(connections, 1)) {
				Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size));
			}

			if (HasBit(connections, 2)) {
				Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size - 1));
			}

			if (HasBit(connections, 3)) {
				Union(labels.data, labels_index, lower_plane_index - 2);
			}

			if (HasBit(connections, 4)) {
				Union(labels.data, labels_index, lower_plane_index);
			}

			if (HasBit(connections, 5)) {
				Union(labels.data, labels_index, lower_plane_index + 2);
			}

			if (HasBit(connections, 6)) {
				Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size - 1));
			}

			if (HasBit(connections, 7)) {
				Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size));
			}

			if (HasBit(connections, 8)) {
				Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size + 1));
			}

			// Current planes

			if (HasBit(connections, 9)) {
				Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size + 1));
			}

			if (HasBit(connections, 10)) {
				Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size));
			}

			if (HasBit(connections, 11)) {
				Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size - 1));
			}

			if (HasBit(connections, 12)) {
				Union(labels.data, labels_index, labels_index - 2);
			}
		}
	}

	__global__ void PathCompression(cuda::PtrStepSz3i labels) {

		unsigned x = 2 * (blockIdx.x * BLOCK_X + threadIdx.x);
		unsigned y = 2 * (blockIdx.y * BLOCK_Y + threadIdx.y);
		unsigned z = 2 * (blockIdx.z * BLOCK_Z + threadIdx.z);
		unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

		if (x < labels.x && y < labels.y && z < labels.z) {
			int val = labels[labels_index];
			if (val < labels_index) {
				labels[labels_index] = Find(labels.data, val);
			}
		}
	}


	__global__ void FinalLabeling(const cuda::PtrStepSz3b img, cuda::PtrStepSz3i labels) {

		unsigned x = 2 * (blockIdx.x * BLOCK_X + threadIdx.x);
		unsigned y = 2 * (blockIdx.y * BLOCK_Y + threadIdx.y);
		unsigned z = 2 * (blockIdx.z * BLOCK_Z + threadIdx.z);
		unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;
		unsigned img_index = z * (img.stepz / img.elem_size) + y * (img.stepy / img.elem_size) + x;

		if (x < labels.x && y < labels.y && z < labels.z) {

			int label = labels[labels_index] + 1;

			// Current plane
			if (img[img_index]) {
				labels[labels_index] = label;
			}
			else {
				labels[labels_index] = 0;
			}

			if (x + 1 < labels.x) {
				if (img[img_index + 1])
					labels[labels_index + 1] = label;
				else {
					labels[labels_index + 1] = 0;
				}

				if (y + 1 < labels.y) {
					if (img[img_index + img.stepy + 1])
						labels[labels_index + (labels.stepy / labels.elem_size) + 1] = label;
					else {
						labels[labels_index + (labels.stepy / labels.elem_size) + 1] = 0;
					}
				}
			}

			if (y + 1 < labels.y) {
				if (img[img_index + img.stepy])
					labels[labels_index + (labels.stepy / labels.elem_size)] = label;
				else {
					labels[labels_index + (labels.stepy / labels.elem_size)] = 0;
				}
			}

			// Upper plane
			if (z + 1 < labels.z) {

				if (img[img_index + img.stepz / img.elem_size])
					labels[labels_index + labels.stepz / labels.elem_size] = label;
				else {
					labels[labels_index + labels.stepz / labels.elem_size] = 0;
				}

				if (x + 1 < labels.x) {
					if (img[img_index + img.stepz / img.elem_size + 1])
						labels[labels_index + labels.stepz / labels.elem_size + 1] = label;
					else {
						labels[labels_index + labels.stepz / labels.elem_size + 1] = 0;
					}

					if (y + 1 < labels.y) {
						if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size + 1])
							labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size) + 1] = label;
						else {
							labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size) + 1] = 0;
						}
					}
				}

				if (y + 1 < labels.y) {
					if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size])
						labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)] = label;
					else {
						labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)] = 0;
					}
				}

			}

		}

	}

}

class CUDA_BKE_3D : public GpuLabeling3D<CONN_26> {
private:
	dim3 grid_size_;
	dim3 block_size_;
	unsigned int* last_voxel_conn_;
	bool allocated_last_conn_;

public:
	CUDA_BKE_3D() {}

	void PerformLabeling() {

		d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);

		// Decide whether last_pixel_ needs specific allocation or not
		// It only needs it in the case that input volume has 2 or more dimensions equals to 1
		allocated_last_conn_ = false;
		if ((d_img_.x % 2 == 1) && (d_img_.y % 2 == 1) && (d_img_.z % 2 == 1)) {
			if (d_img_.x > 1 && d_img_.y > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 1) * d_img_labels_.stepz + (d_img_labels_.y - 2) * d_img_labels_.stepy) + d_img_labels_.x - 2;
			}
			else if (d_img_.x > 1 && d_img_.z > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 2) * d_img_labels_.stepz + (d_img_labels_.y - 1) * d_img_labels_.stepy) + d_img_labels_.x - 2;
			}
			else if (d_img_.y > 1 && d_img_.z > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 2) * d_img_labels_.stepz + (d_img_labels_.y - 2) * d_img_labels_.stepy) + d_img_labels_.x - 1;
			}
			else {
				cudaMalloc(&last_voxel_conn_, sizeof(unsigned int));
				allocated_last_conn_ = true;
			}
		}

		grid_size_ = dim3((d_img_.x + BLOCK_X - 1) / BLOCK_X, (d_img_.y + BLOCK_Y - 1) / BLOCK_Y, (d_img_.z + BLOCK_Z - 1) / BLOCK_Z);
		block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

		InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_voxel_conn_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//Mat1i init_labels;
		//d_block_labels_.download(init_labels);

		Merge << <grid_size_, block_size_ >> > (d_img_labels_, last_voxel_conn_);

		//Mat1i block_info_final;
		//d_img_labels_.download(block_info_final);		

		PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);

		FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

		// d_img_labels_.download(img_labels_);
		
		if (allocated_last_conn_) {
			cudaFree(last_voxel_conn_);
		}
		
		cudaDeviceSynchronize();
	}


private:
	void Alloc() {
		d_img_labels_.create(d_img_.x, d_img_.y, d_img_.z, CV_32SC1);
		allocated_last_conn_ = false;
		if ((d_img_.x % 2 == 1) && (d_img_.y % 2 == 1) && (d_img_.z % 2 == 1)) {
			if (d_img_.x > 1 && d_img_.y > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 1) * d_img_labels_.stepz + (d_img_labels_.y - 2) * d_img_labels_.stepy) + d_img_labels_.x - 2;
			}
			else if (d_img_.x > 1 && d_img_.z > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 2) * d_img_labels_.stepz + (d_img_labels_.y - 1) * d_img_labels_.stepy) + d_img_labels_.x - 2;
			}
			else if (d_img_.y > 1 && d_img_.z > 1) {
				last_voxel_conn_ = reinterpret_cast<unsigned int*>(d_img_labels_.data + (d_img_labels_.z - 2) * d_img_labels_.stepz + (d_img_labels_.y - 2) * d_img_labels_.stepy) + d_img_labels_.x - 1;
			}
			else {
				cudaMalloc(&last_voxel_conn_, sizeof(unsigned int));
				allocated_last_conn_ = true;
			}
		}
		cudaDeviceSynchronize();
	}

	void Dealloc() {
		if (allocated_last_conn_) {
			cudaFree(last_voxel_conn_);
		}
		cudaDeviceSynchronize();
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
		grid_size_ = dim3((d_img_.x + BLOCK_X - 1) / BLOCK_X, (d_img_.y + BLOCK_Y - 1) / BLOCK_Y, (d_img_.z + BLOCK_Z - 1) / BLOCK_Z);
		block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);


		InitLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_voxel_conn_);

		//cuda::GpuMat d_expanded_connections;
		//d_expanded_connections.create(d_connections_.rows * 3, d_connections_.cols * 3, CV_8UC1);
		//ExpandConnections << <grid_size_, block_size_ >> > (d_connections_, d_expanded_connections);
		//Mat1b expanded_connections;
		//d_expanded_connections.download(expanded_connections);
		//d_expanded_connections.release();

		//Mat1i init_labels;
		//d_block_labels_.download(init_labels);

		Merge << <grid_size_, block_size_ >> > (d_img_labels_, last_voxel_conn_);

		//Mat1i block_info_final;
		//d_img_labels_.download(block_info_final);		

		PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);

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

REGISTER_LABELING(CUDA_BKE_3D);