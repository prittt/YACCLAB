// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "cuda_mat3.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/cuda/utility.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;



void cv::cuda::GpuMat3::release()
{
	cudaFree(data);
	data = 0;
	stepy = stepz = x = y = z = 0;
}

void cv::cuda::GpuMat3::create(int _x, int _y, int _z, int _type)
{
	CV_DbgAssert(_x >= 0 && _y >= 0 && _z >= 0);

	_type &= Mat::TYPE_MASK;

	if (x == _x && y == _y && z == _z && type() == _type && data)
		return;

	if (data)
		release();

	if (_x > 0 && _y > 0 && _z > 0)
	{
		flags = Mat::MAGIC_VAL + _type;
		x = _x;
		y = _y;
		z = _z;

		const size_t esz = elemSize();

		struct cudaPitchedPtr pitchedPtr;
		struct cudaExtent extent;

		extent.width = _x * esz;     
		extent.height = _y;
		extent.depth = _z;

		CV_CUDEV_SAFE_CALL(cudaMalloc3D(&pitchedPtr, extent));

		data = reinterpret_cast<uchar *>(pitchedPtr.ptr);
		stepy = pitchedPtr.pitch;
		stepz = pitchedPtr.pitch * y;

		if (esz * x == pitchedPtr.pitch)
			flags |= Mat::CONTINUOUS_FLAG;
	}
}

void cv::cuda::GpuMat3::upload(Mat &mat)
{
	CV_DbgAssert(!mat.empty());
	CV_DbgAssert(mat.dims == 3);

	create(mat.size[2], mat.size[1], mat.size[0], mat.type());

	cudaPitchedPtr srcPtr, dstPtr;
	srcPtr.pitch = mat.step[1];
	srcPtr.ptr = mat.data;
	srcPtr.xsize = mat.size[2] * mat.elemSize();
	srcPtr.ysize = mat.size[1];
	dstPtr.pitch = stepy;
	dstPtr.ptr = data;
	dstPtr.xsize = mat.size[2] * elemSize();
	dstPtr.ysize = mat.size[1];

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = srcPtr;
	params.dstPtr = dstPtr;
	params.extent = make_cudaExtent(x * elemSize(), y, z);
	params.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3DParms *params_ptr = &params;

	CV_CUDEV_SAFE_CALL(cudaMemcpy3D(params_ptr));
}

void cv::cuda::GpuMat3::download(Mat &mat) const
{
	CV_DbgAssert(!empty());

	int sz[] = { z, y, x };

	mat.create(3, sz, type());

	cudaPitchedPtr srcPtr, dstPtr;
	dstPtr.pitch = mat.step[1];
	dstPtr.ptr = mat.data;
	dstPtr.xsize = mat.size[2] * elemSize();
	dstPtr.ysize = mat.size[1];
	srcPtr.pitch = stepy;
	srcPtr.ptr = data;
	srcPtr.xsize = mat.size[2] * elemSize();
	srcPtr.ysize = mat.size[1];

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = srcPtr;
	params.dstPtr = dstPtr;
	params.extent = make_cudaExtent(x * elemSize(), y, z);
	params.kind = cudaMemcpyDeviceToHost;

	CV_CUDEV_SAFE_CALL(cudaMemcpy3D(&params));
}

#endif