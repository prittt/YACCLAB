// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef OPENCV_CORE_CUDA_MAT3_INL_HPP
#define OPENCV_CORE_CUDA_MAT3_INL_HPP

#include "cuda_mat3.hpp"


namespace cv {
	namespace cuda {

		inline GpuMat3::GpuMat3() : flags(0), stepy(0), stepz(0), data(0) {}

		inline
			int GpuMat3::type() const
		{
			return CV_MAT_TYPE(flags);
		}

		inline
			size_t GpuMat3::elemSize() const
		{
			return CV_ELEM_SIZE(flags);
		}

		inline
			bool GpuMat3::empty() const
		{
			return data == 0;
		}


		//template <class T> inline
		//	GpuMat3::operator PtrStepSz3<T>() const
		//{
		//	return PtrStepSz3<T>(x, y, z, (T*)data, stepy, stepz);
		//}

		//template <class T> inline
		//	GpuMat3::operator PtrStep3<T>() const
		//{
		//	return PtrStep3<T>((T*)data, stepy, stepz);
		//}

	}
}


#endif // OPENCV_CORE_CUDAINL_HPP
