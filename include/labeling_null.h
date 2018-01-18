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

#ifndef YACCLAB_LABELING_NULL_H_
#define YACCLAB_LABELING_NULL_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class labeling_NULL : public Labeling {
public:

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size());

        for (int r = 0; r < img_labels_.rows; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);

            for (int c = 0; c < img_labels_.cols; ++c) {
                    img_labels_row[c] = img_row[c];
            }
        }
    }

    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        AllScans();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
    }

    void PerformLabelingMem(std::vector<unsigned long int>& accesses)
    {
        MemMat<uchar> img(img_);
        MemMat<int> img_labels(img_.size());

        for (int r = 0; r < img_labels.rows; ++r) {
            for (int c = 0; c < img_labels.cols; ++c) {
                img_labels(r, c) = img(r, c);
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
    }

private:
    double Alloc()
    {
        // Memory allocation for the output image
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double ma_t = t - perf_.last();
        // Return total time
        return ma_t;
    }
    void Dealloc()
    {
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void AllScans()
    {
        for (int r = 0; r < img_labels_.rows; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);

            for (int c = 0; c < img_labels_.cols; ++c) {
                img_labels_row[c] = img_row[c];
            }
        }
    }
};

#endif // !YACCLAB_LABELING_NULL_H_