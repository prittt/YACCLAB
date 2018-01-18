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

#ifndef YACCLAB_LABELING_GRANA_2016_H_
#define YACCLAB_LABELING_GRANA_2016_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class PRED : public Labeling {
public:
    PRED() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0); // Call to memset

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img_row[c]>0
#define CONDITION_P img_row_prev[c-1]>0
#define CONDITION_Q img_row_prev[c]>0
#define CONDITION_R img_row_prev[c+1]>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#define COLS w

        {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(0);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(0);

#include "labeling_grana_2016_forest_0.inc"

        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);

#include "labeling_grana_2016_forest.inc"  

        }//End rows's for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R

    // Second scan
        n_labels_ = LabelsSolver::Flatten();

        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *b = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *e = b + img_labels_.cols;
            for (; b != e; ++b) {
                *b = LabelsSolver::GetLabel(*b);
            }
        }

        LabelsSolver::Dealloc();
    }

    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);

    }

    void PerformLabelingMem(std::vector<unsigned long int>& accesses)
    {

        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0); 

        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::MemSetup();

        // First scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img(r,c)>0
#define CONDITION_P img(r-1,c-1)>0
#define CONDITION_Q img(r-1,c)>0
#define CONDITION_R img(r-1,c+1)>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels(r,c) = LabelsSolver::MemNewLabel(); // new label
#define ACTION_3 img_labels(r,c) = img_labels(r - 1, c - 1); // x <- p
#define ACTION_4 img_labels(r,c) = img_labels(r - 1, c); // x <- q
#define ACTION_5 img_labels(r,c) = img_labels(r - 1, c + 1); // x <- r
#define ACTION_6 img_labels(r,c) = img_labels(r, c - 1); // x <- s
#define ACTION_7 img_labels(r,c) = LabelsSolver::MemMerge(img_labels(r - 1, c - 1), img_labels(r - 1, c + 1)); // x <- p + r
#define ACTION_8 img_labels(r,c) = LabelsSolver::MemMerge(img_labels(r, c - 1), img_labels(r - 1, c + 1)); // x <- s + r

#define COLS w

        {
            int r = 0; 
#include "labeling_grana_2016_forest_0.inc"

        }

        for (int r = 1; r < h; ++r) {

#include "labeling_grana_2016_forest.inc"  

        }//End rows' for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R

        n_labels_ = LabelsSolver::MemFlatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels.rows; ++r_i) {
            for (int c_i = 0; c_i < img_labels.cols; ++c_i) {
                img_labels(r_i, c_i) = LabelsSolver::MemGetLabel(img_labels(r_i, c_i));
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage(); 

        LabelsSolver::MemDealloc();
    }

private:
    double Alloc()
    {
        // Memory allocation of the labels solver
        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);
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
        return ls_t + ma_t;
    }

    void Dealloc()
    {
        LabelsSolver::Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization
        LabelsSolver::Setup();

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img_row[c]>0
#define CONDITION_P img_row_prev[c-1]>0
#define CONDITION_Q img_row_prev[c]>0
#define CONDITION_R img_row_prev[c+1]>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#define COLS w

        {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(0);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(0);

#include "labeling_grana_2016_forest_0.inc"
                
        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);
        
#include "labeling_grana_2016_forest.inc"  

        }//End rows's for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
    }
    void SecondScan()
    {
        // Second scan
        n_labels_ = LabelsSolver::Flatten();

        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *b = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *e = b + img_labels_.cols;
            for (; b != e; ++b) {
                *b = LabelsSolver::GetLabel(*b);
            }
        }
    }
};

#endif // YACCLAB_LABELING_GRANA_2016_H_