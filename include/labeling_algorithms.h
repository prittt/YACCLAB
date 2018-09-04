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

// gpu include
#include <opencv2\cudafeatures2d.hpp>

#include "performance_evaluator.h"

#define UPPER_BOUND_8_CONNECTIVITY ((size_t)((img_.rows + 1) / 2) * (size_t)((img_.cols + 1) / 2) + 1)
#define UPPER_BOUND_4_CONNECTIVITY (((size_t)img_.rows * (size_t)img_.cols + 1) / 2 + 1)

class Labeling {
public:
    static cv::Mat1b img_;
    cv::Mat1i img_labels_;
    unsigned n_labels_;
    PerformanceEvaluator perf_;

    Labeling() {}
    virtual ~Labeling() = default;

    virtual void PerformLabeling() { throw std::runtime_error("'PerformLabeling()' not implemented"); }
    virtual void PerformLabelingWithSteps() { throw std::runtime_error("'PerformLabelingWithSteps()' not implemented"); }
    virtual void PerformLabelingMem(std::vector<unsigned long>& accesses) { throw std::runtime_error("'PerformLabelingMem(...)' not implemented"); }

    virtual void FreeLabelingData() { img_labels_.release(); }
    //static void SetImg(const cv::Mat1b& img) { img_ = img.clone(); }

//private:
//    virtual void Alloc() {}
//    virtual void Dealloc() {}
};


// This could be a Labeling subclass
class GpuLabeling : public Labeling {
public:
	static cv::cuda::GpuMat d_img_;
	cv::cuda::GpuMat d_img_labels_;

	GpuLabeling() {}
	virtual ~GpuLabeling() = default;

	// virtual void PerformLabeling() { throw std::runtime_error("'PerformLabeling()' not implemented"); }
	// virtual void PerformLabelingWithSteps() { throw std::runtime_error("'PerformLabelingWithSteps()' not implemented"); }
	// virtual void PerformLabelingMem(std::vector<unsigned long>& accesses) { throw std::runtime_error("'PerformLabelingMem(...)' not implemented"); }

	virtual void FreeLabelingData() { d_img_labels_.release(); img_labels_.release(); }
	//static void SetImg(const cv::Mat1b& img) { img_ = img.clone(); }

	//private:
	//    virtual void Alloc() {}
	//    virtual void Dealloc() {}
};

// Maybe I should split this in two classes
class LabelingMapSingleton {
public:
    std::map<std::string, Labeling*> data_;

	// std::map<std::string, GpuLabeling*> gpu_data_;

    static LabelingMapSingleton& GetInstance();
    static Labeling* GetLabeling(const std::string& s);			///////
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