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

#ifndef YACCLAB_LABELING_ALGORITHMS_H_
#define YACCLAB_LABELING_ALGORITHMS_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utilities.h"
#include "cuda_mat3.hpp"
#include "system_info.h"
#include "volume_util.h"
#include "yacclab_tensor.h"
#include "check_labeling.h"


// gpu include
#if defined YACCLAB_WITH_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif

#include "performance_evaluator.h"

#define UPPER_BOUND_4_CONNECTIVITY (((size_t)img_.rows * (size_t)img_.cols + 1) / 2 + 1)
#define UPPER_BOUND_8_CONNECTIVITY ((size_t)((img_.rows + 1) / 2) * (size_t)((img_.cols + 1) / 2) + 1)

#define UPPER_BOUND_6_CONNECTIVITY (((size_t)img_.size[0] * (size_t)img_.size[1] * (size_t)img_.size[2] + 1) / 2 + 1)
#define UPPER_BOUND_26_CONNECTIVITY ((size_t)((img_.size[0] + 1) / 2) * (size_t)((img_.size[1] + 1) / 2) * (size_t)((img_.size[2] + 1) / 2) + 1)




class Labeling {
public:
    PerformanceEvaluator perf_;
    std::unique_ptr<YacclabTensorInput> input_;
    std::unique_ptr<YacclabTensorOutput> output_;

    Labeling(std::unique_ptr<YacclabTensorInput> input, std::unique_ptr<YacclabTensorOutput> output) :
        input_(std::move(input)), output_(std::move(output)) {}

    virtual ~Labeling() = default;
	
    virtual void PerformLabeling() { throw std::runtime_error("'PerformLabeling()' not implemented"); }
    virtual void PerformLabelingWithSteps() { throw std::runtime_error("'PerformLabelingWithSteps()' not implemented"); }
    virtual void PerformLabelingMem(std::vector<uint64_t>& accesses) { throw std::runtime_error("'PerformLabelingMem(...)' not implemented"); }
    virtual void FreeLabelingData() { output_->Release(); }

    virtual std::string GetTitle() const { return GetGnuplotTitle(); }
    virtual std::string CheckAlg() const = 0;

    virtual YacclabTensorInput* GetInput() { return input_.get(); }
    virtual YacclabTensorOutput* GetOutput() { return output_.get(); }

};

template <Connectivity2D Conn>
class Labeling2D : public Labeling {
public:
    cv::Mat1b &img_;
    cv::Mat1i &img_labels_;
    unsigned int n_labels_;

    Labeling2D(std::unique_ptr<YacclabTensorInput2D> input, std::unique_ptr<YacclabTensorOutput2D> output) :
        Labeling(std::move(input), std::move(output)),
        img_((dynamic_cast<YacclabTensorInput2D*>(input_.get()))->Raw()),
        img_labels_((dynamic_cast<YacclabTensorOutput2D*>(output_.get()))->Raw()) {}

    Labeling2D() : Labeling2D(std::make_unique<YacclabTensorInput2D>(), std::make_unique<YacclabTensorOutput2D>()) {}

    virtual ~Labeling2D() = default;

    virtual std::string CheckAlg() const { return LabelingCheckSingleton2D::GetCheckAlg(Conn); }

};

template <Connectivity3D Conn>
class Labeling3D : public Labeling {
public:
    cv::Mat &img_;
    cv::Mat &img_labels_;

    Labeling3D(std::unique_ptr<YacclabTensorInput3D> input, std::unique_ptr<YacclabTensorOutput3D> output) :
        Labeling(std::move(input), std::move(output)),
        img_((dynamic_cast<YacclabTensorInput3D*>(input_.get()))->Raw()),
        img_labels_((dynamic_cast<YacclabTensorOutput3D*>(output_.get()))->Raw()) {}

    Labeling3D() : Labeling3D(std::make_unique<YacclabTensorInput3D>(), std::make_unique<YacclabTensorOutput3D>()) {}

    virtual ~Labeling3D() = default;

    virtual std::string CheckAlg() const { return LabelingCheckSingleton3D::GetCheckAlg(Conn); }

};


#if defined YACCLAB_WITH_CUDA
template <Connectivity2D Conn>
class GpuLabeling2D : public Labeling2D<Conn> {
public:
	using Labeling2D<Conn>::input_;
	using Labeling2D<Conn>::output_;

	cv::cuda::GpuMat &d_img_;
	cv::cuda::GpuMat &d_img_labels_;

    GpuLabeling2D() : 
	Labeling2D<Conn>(std::make_unique<YacclabTensorInput2DCuda>(), std::make_unique<YacclabTensorOutput2DCuda>()), 
	d_img_(dynamic_cast<YacclabTensorInput2DCuda*>(input_.get())->GpuRaw()), 
	d_img_labels_(dynamic_cast<YacclabTensorOutput2DCuda*>(output_.get())->GpuRaw()) {}

	virtual ~GpuLabeling2D() = default;
	
    virtual std::string GetTitle() const { return GetGnuplotTitleGpu(); }
};


template <Connectivity3D Conn>
class GpuLabeling3D : public Labeling3D<Conn> {
public:
	using Labeling3D<Conn>::input_;
	using Labeling3D<Conn>::output_;

	cv::cuda::GpuMat3 &d_img_;
	cv::cuda::GpuMat3 &d_img_labels_;

    GpuLabeling3D() :
        Labeling3D<Conn>(std::make_unique<YacclabTensorInput3DCuda>(), std::make_unique<YacclabTensorOutput3DCuda>()),
        d_img_((dynamic_cast<YacclabTensorInput3DCuda*>(input_.get()))->GpuRaw()),
        d_img_labels_((dynamic_cast<YacclabTensorOutput3DCuda*>(output_.get()))->GpuRaw()) {}

    virtual ~GpuLabeling3D() = default;

    virtual std::string GetTitle() const { return GetGnuplotTitleGpu(); }
};

#endif // YACCLAB_WITH_CUDA

class LabelingMapSingleton {
public:
    std::map<std::string, Labeling*> data_;

    static LabelingMapSingleton& GetInstance();
    static Labeling* GetLabeling(const std::string& s);			
    static bool Exists(const std::string& s);
    LabelingMapSingleton(LabelingMapSingleton const&) = delete;
    void operator=(LabelingMapSingleton const&) = delete;

private:
    LabelingMapSingleton() {}
    ~LabelingMapSingleton()
    {
        for (std::map<std::string, Labeling*>::iterator it = data_.begin(); it != data_.end(); ++it)
            delete it->second;
    }
};

enum StepType {
    ALLOC_DEALLOC = 0,
    FIRST_SCAN = 1,
    SECOND_SCAN = 2,
    ALL_SCANS = 3,

    ST_SIZE = 4,
};

std::string Step(StepType n_step);

#endif //YACCLAB_LABELING_ALGORITHMS_H_
