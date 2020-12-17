// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_TENSOR_H_
#define YACCLAB_TENSOR_H_

#include <string>
#include <memory>

#ifdef YACCLAB_WITH_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif

#include "cuda_mat3.hpp"
#include "volume_util.h"


class YacclabTensor {
public:
    virtual void Release() = 0;
    // Possibly add some descriptive fields

    //virtual ~YacclabTensor() { this->Release(); };
    virtual ~YacclabTensor() = default;
};

class YacclabTensorInput : public YacclabTensor {
public:
    virtual void Create() = 0;
    virtual bool ReadBinary(const std::string &filename) = 0;
    virtual unsigned int Dims() = 0;
};

class YacclabTensorInput2D : public YacclabTensorInput {
protected:
    static cv::Mat1b mat_;
public:
    virtual void Create() override { mat_.create(1, 1); }
    virtual void Release() override { mat_.release(); }
    virtual bool ReadBinary(const std::string &filename) override;
    virtual cv::Mat& GetMat() {  return mat_;  }
    virtual cv::Mat1b& Raw() {  return mat_;  }
	virtual unsigned int Dims() override { return 2; };
};

class YacclabTensorInput3D : public YacclabTensorInput {
protected:
    static cv::Mat mat_;
public:
    virtual void Create() override {    int sz[] = { 1, 1, 1 };    mat_.create(3, sz, CV_8UC1);    }
    virtual void Release() override {  mat_.release(); }
    virtual bool ReadBinary(const std::string &filename) override;
    virtual cv::Mat& GetMat() { return mat_;  }
    virtual cv::Mat& Raw() {  return mat_;  }
    virtual unsigned int Dims() override { return 3; };
};


#if defined YACCLAB_WITH_CUDA
class YacclabTensorInput2DCuda : public YacclabTensorInput2D {
protected:
    static cv::cuda::GpuMat d_mat_;
public:
    virtual void Create() override {   YacclabTensorInput2D::Create();   d_mat_.upload(mat_);   }
    virtual void Release() override { YacclabTensorInput2D::Release();  d_mat_.release(); }
    virtual bool ReadBinary(const std::string &filename) override;
    virtual cv::Mat& GetMat() override {  return mat_;  }
    virtual cv::cuda::GpuMat& GpuRaw() {  return d_mat_;  }
};

class YacclabTensorInput3DCuda : public YacclabTensorInput3D {
protected:
    static cv::cuda::GpuMat3 d_mat_;
public:
    virtual void Create() override { YacclabTensorInput3D::Create();   d_mat_.upload(mat_); }
    virtual void Release() override { YacclabTensorInput3D::Release();  d_mat_.release(); }
    virtual bool ReadBinary(const std::string &filename) override;
    virtual cv::Mat& GetMat() override {  return mat_;  }
    virtual cv::cuda::GpuMat3& GpuRaw() {  return d_mat_;  }
};

#endif

class YacclabTensorOutput : public YacclabTensor {
public:
    virtual void NormalizeLabels(bool label_background = false) = 0;
    virtual void WriteColored(const std::string &filename) const = 0;
    virtual void PrepareForCheck() = 0;

    virtual const cv::Mat& GetMat() = 0;

    virtual bool Equals(YacclabTensorOutput *other);
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const = 0;
};

class YacclabTensorOutput2D : public YacclabTensorOutput {
protected:
    cv::Mat1i mat_;
public:
    YacclabTensorOutput2D() {}
    YacclabTensorOutput2D(cv::Mat1i mat) : mat_(std::move(mat)) {}

    virtual void NormalizeLabels(bool label_background = false) override;
    virtual void WriteColored(const std::string &filename) const override;
    virtual void PrepareForCheck() override {}
    virtual void Release() override {  mat_.release();  }
    virtual const cv::Mat& GetMat() override {  return mat_;  }
    virtual cv::Mat1i& Raw() {  return mat_;  }
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const override {  return std::make_unique<YacclabTensorOutput2D>(mat_);  }
};

class YacclabTensorOutput3D : public YacclabTensorOutput {
protected:
    cv::Mat mat_;
public:
    YacclabTensorOutput3D() {}
    YacclabTensorOutput3D(cv::Mat mat) : mat_(std::move(mat)) {}

    virtual void NormalizeLabels(bool label_background = false) override;
    virtual void WriteColored(const std::string &filename) const override;
    virtual void PrepareForCheck() override {}
    virtual void Release() override {  mat_.release();  }
    virtual const cv::Mat& GetMat() override {  return mat_;  }
    virtual cv::Mat& Raw() {  return mat_;  }
    virtual std::unique_ptr<YacclabTensorOutput> Copy() const override {  return std::make_unique<YacclabTensorOutput3D>(mat_); }
};

#if defined YACCLAB_WITH_CUDA
class YacclabTensorOutput2DCuda : public YacclabTensorOutput2D {
protected:
    cv::cuda::GpuMat d_mat_;
public:
    virtual void PrepareForCheck() override {  d_mat_.download(YacclabTensorOutput2D::mat_);  }
    virtual void Release() override {  YacclabTensorOutput2D::Release();  d_mat_.release();  }
    virtual const cv::Mat& GetMat() override {  return YacclabTensorOutput2D::mat_;  }
    virtual cv::cuda::GpuMat& GpuRaw() {  return d_mat_;  }
};

class YacclabTensorOutput3DCuda : public YacclabTensorOutput3D {
protected:
    cv::cuda::GpuMat3 d_mat_;
public:
    virtual void PrepareForCheck() override {  d_mat_.download(YacclabTensorOutput3D::mat_);  }
    virtual void Release() override {  YacclabTensorOutput3D::Release();  d_mat_.release();  }
    virtual const cv::Mat& GetMat() override {  return YacclabTensorOutput3D::mat_;  }
    virtual cv::cuda::GpuMat3& GpuRaw() {  return d_mat_;  }
};
#endif

#endif //YACCLAB_TENSOR_H_
