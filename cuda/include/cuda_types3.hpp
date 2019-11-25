#ifndef OPENCV_CORE_CUDA_TYPES3_HPP
#define OPENCV_CORE_CUDA_TYPES3_HPP

#ifndef __cplusplus
#  error cuda_types.hpp header must be compiled as C++
#endif

#if defined(__OPENCV_BUILD) && defined(__clang__)
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
#if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

/** @file
 * @deprecated Use @ref cudev instead.
 */

 //! @cond IGNORED

#ifdef __CUDACC__
#define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
#define __CV_CUDA_HOST_DEVICE__
#endif

#include "opencv2/core/cuda_types.hpp"

namespace cv
{
	namespace cuda
	{

		// Simple lightweight structures that encapsulates information about an image on device.
		// It is intended to pass to nvcc-compiled code. GpuMat3 depends on headers that nvcc can't compile

		template <typename T> struct PtrSz3 : public DevPtr<T>
		{
			__CV_CUDA_HOST_DEVICE__ PtrSz3() : size(0) {}
			__CV_CUDA_HOST_DEVICE__ PtrSz3(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

			size_t size;
		};

		template <typename T> struct PtrStep3 : public DevPtr<T>
		{
			__CV_CUDA_HOST_DEVICE__ PtrStep3() : stepy(0), stepz(0) {}
			__CV_CUDA_HOST_DEVICE__ PtrStep3(T* data_, size_t stepy_, size_t stepz_) : DevPtr<T>(data_), stepz(stepz_), stepy(stepy_) {}
			//__CV_CUDA_HOST_DEVICE__ PtrStep3(GpuMat3 &gpu_mat) : DevPtr<T>(reinterpret_cast<T*>(gpu_mat.data)), stepz(gpu_mat.stepz), stepy(gpu_mat.stepy) {}

			size_t stepy;
			size_t stepz;

			__CV_CUDA_HOST_DEVICE__       T* ptr(int z = 0, int y = 0) { return (T*)((char*)DevPtr<T>::data + z * stepz + y * stepy); }
			__CV_CUDA_HOST_DEVICE__ const T* ptr(int z = 0, int y = 0) const { return (const T*)((const char*)DevPtr<T>::data + z * stepz + y * stepy); }

			__CV_CUDA_HOST_DEVICE__       T& operator ()(int x, int y, int z) { return ptr(z, y)[x]; }
			__CV_CUDA_HOST_DEVICE__ const T& operator ()(int x, int y, int z) const { return ptr(z, y)[x]; }
		};

		template <typename T> struct PtrStepSz3 : public PtrStep3<T>
		{
			__CV_CUDA_HOST_DEVICE__ PtrStepSz3() : x(0), y(0), z(0) {}
			__CV_CUDA_HOST_DEVICE__ PtrStepSz3(int x_, int y_, int z_, T* data_, size_t stepy_, size_t stepz_)
				: PtrStep3<T>(data_, stepy_, stepz_), x(x_), y(y_), z(z_) {}

			template <typename U>
			explicit PtrStepSz3(const PtrStepSz3<U>& d) : PtrStep3<T>((T*)d.data, d.stepy, d.stepz), x(d.x), y(d.y), z(d.z) {}

			int x;
			int y;
			int z;
		};

		typedef PtrStepSz3<unsigned char> PtrStepSz3b;
		typedef PtrStepSz3<float> PtrStepSz3f;
		typedef PtrStepSz3<int> PtrStepSz3i;

		typedef PtrStep3<unsigned char> PtrStep3b;
		typedef PtrStep3<float> PtrStep3f;
		typedef PtrStep3<int> PtrStep3i;

	}
}

//! @endcond

#endif /* OPENCV_CORE_CUDA_TYPES3_HPP */
