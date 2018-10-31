// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
//
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
//
// * Neither the name of YACCLAB nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef YACCLAB_TENSOR_H_
#define YACCLAB_TENSOR_H_

#include <string>

#include "cuda_mat3.hpp"
#include "volume_util.h"


using namespace cv;

class YacclabTensor {
public:
    virtual void Release() = 0;

    // Possibly add some descriptive fields
};

class YacclabTensorInput : public YacclabTensor {
public:
    virtual void Create() = 0;
    virtual bool ReadBinary(const std::string &filename) = 0;
};

class YacclabTensorInput2D : public YacclabTensorInput {
protected:
    static Mat1b mat_;
public:
    virtual void Create() { mat_.create(1, 1); }
    virtual void Release() { mat_.release(); }
    virtual bool ReadBinary(const std::string &filename);
    virtual Mat& GetMat() {  return mat_;  }
    virtual Mat1b& Raw() {  return mat_;  }
};

class YacclabTensorInput3D : public YacclabTensorInput {
protected:
    static Mat mat_;
public:
    virtual void Create() {    int sz[] = { 1, 1, 1 };    mat_.create(3, sz, CV_8UC1);    }
    virtual void Release() {  mat_.release(); }
    virtual bool ReadBinary(const std::string &filename);
    virtual Mat& GetMat() {  return mat_;  }
    virtual Mat& Raw() {  return mat_;  }
};


#if defined USE_CUDA
class YacclabTensorInput2DCuda : public YacclabTensorInput2D {
protected:
    static cuda::GpuMat d_mat_;
public:
    virtual void Create() {   YacclabTensorInput2D::Create();   d_mat_.upload(mat_);   }
    virtual void Release() { YacclabTensorInput2D::Release();  d_mat_.release(); }
    virtual bool ReadBinary(const std::string &filename);
    virtual Mat& GetMat() {  return mat_;  }
    virtual cuda::GpuMat& GpuRaw() {  return d_mat_;  }
};

class YacclabTensorInput3DCuda : public YacclabTensorInput3D {
protected:
    static cuda::GpuMat3 d_mat_;
public:
    virtual void Create() { YacclabTensorInput3D::Create();   d_mat_.upload(mat_); }
    virtual void Release() { YacclabTensorInput3D::Release();  d_mat_.release(); }
    virtual bool ReadBinary(const std::string &filename);
    virtual Mat& GetMat() {  return mat_;  }
    virtual cuda::GpuMat3& GpuRaw() {  return d_mat_;  }
};

#endif

class YacclabTensorOutput : public YacclabTensor {
public:
    virtual void NormalizeLabels() = 0;
    virtual void WriteColored(const std::string &filename) const = 0;
    virtual void PrepareForCheck() = 0;

    virtual const Mat& GetMat() = 0;

    virtual bool Equals(YacclabTensorOutput *other);
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const = 0;
};

class YacclabTensorOutput2D : public YacclabTensorOutput {
protected:
    Mat1i mat_;
public:
    YacclabTensorOutput2D() {}
    YacclabTensorOutput2D(Mat1i mat) : mat_(std::move(mat)) {}

    virtual void NormalizeLabels();
    virtual void WriteColored(const std::string &filename) const;
    virtual void PrepareForCheck() {}
    virtual void Release() {  mat_.release();  }
    virtual const Mat& GetMat() {  return mat_;  }
    virtual Mat1i& Raw() {  return mat_;  }
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const {  return std::make_unique<YacclabTensorOutput2D>(mat_);  }
};

class YacclabTensorOutput3D : public YacclabTensorOutput {
protected:
    Mat mat_;
public:
    YacclabTensorOutput3D() {}
    YacclabTensorOutput3D(Mat mat) : mat_(std::move(mat)) {}

    virtual void NormalizeLabels();
    virtual void WriteColored(const std::string &filename) const;
    virtual void PrepareForCheck() {}
    virtual void Release() {  mat_.release();  }
    virtual const Mat& GetMat() {  return mat_;  }
    virtual Mat& Raw() {  return mat_;  }
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const {  return std::make_unique<YacclabTensorOutput3D>(mat_); }
};

#if defined USE_CUDA
class YacclabTensorOutput2DCuda : public YacclabTensorOutput2D {
protected:
    cuda::GpuMat d_mat_;
public:
    virtual void PrepareForCheck() {  d_mat_.download(YacclabTensorOutput2D::mat_);  }
    virtual void Release() {  YacclabTensorOutput2D::Release();  d_mat_.release();  }
    virtual const Mat& GetMat() {  return YacclabTensorOutput2D::mat_;  }
    virtual cuda::GpuMat& GpuRaw() {  return d_mat_;  }
};

class YacclabTensorOutput3DCuda : public YacclabTensorOutput3D {
protected:
    cuda::GpuMat3 d_mat_;
public:
    virtual void PrepareForCheck() {  d_mat_.download(YacclabTensorOutput3D::mat_);  }
    virtual void Release() {  YacclabTensorOutput3D::Release();  d_mat_.release();  }
    virtual const Mat& GetMat() {  return YacclabTensorOutput3D::mat_;  }
    virtual cuda::GpuMat3& GpuRaw() {  return d_mat_;  }
};
#endif

#endif //YACCLAB_TENSOR_H_