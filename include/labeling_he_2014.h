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

#ifndef YACCLAB_LABELING_HE_2014_H_
#define YACCLAB_LABELING_HE_2014_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class CTB : public Labeling2D<CONN_8> {
public:
    CTB() {}

    // Possible configurations defined by the CTB algorithms
#define CA 1
#define CB 2
#define CC 3
#define CD 4
#define CE 5
#define CF 6
#define CG 7
#define CH 8
#define CI 9
#define BR -1 // State: Begin of a new row

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0); // Call to memset

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        // Scan Mask
        // +--+--+--+
        // |n1|n2|n3|
        // +--+--+--+
        // |n4|a |
        // +--+--+
        // |n5|b |
        // +--+--+

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);

            int prob_fol_state = BR;
            int prev_state = BR;

            for (int c = 0; c < w; c += 1) {

                // A bunch of defines used to check if the pixels are foreground, and current state of graph
                // without going outside the image limits.

#define CONDITION_A img_row[c]>0
#define CONDITION_B r+1<h && img_row_fol[c]>0
#define CONDITION_N1 c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_N2 r-1>=0 && img_row_prev[c]>0
#define CONDITION_N3 r-1>=0 && c+1<w && img_row_prev[c+1]>0
#define CONDITION_N4 c-1>=0 && img_row[c-1]>0
#define CONDITION_N5 c-1>=0 && r+1<h && img_row_fol[c-1]>0

#define ACTION_1 // nothing
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // a <- n1
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // a <- n2
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // a <- n3
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // a <- n4
#define ACTION_7 img_labels_row[c] = img_labels_row_fol[c - 1]; // a <- n5
#define ACTION_8 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c - 1]); // a + n1
#define ACTION_9 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c + 1]); // a + n3
#define ACTION_10 LabelsSolver::Merge(img_labels_row[c], img_labels_row_fol[c - 1]); // a + n5
#define ACTION_11 img_labels_row_fol[c] = LabelsSolver::NewLabel(); // b new label
#define ACTION_12 img_labels_row_fol[c] = img_labels_row[c - 1]; // b <- n4
#define ACTION_13 img_labels_row_fol[c] = img_labels_row_fol[c - 1]; // b <- n5
#define ACTION_14 img_labels_row_fol[c] = img_labels_row[c]; // b <- a

#include "labeling_he_2014_graph.inc"
            }//End columns's for
        }//End rows's for

#undef ACTION_1 
#undef ACTION_2 
#undef ACTION_3 
#undef ACTION_4 
#undef ACTION_5 
#undef ACTION_6 
#undef ACTION_7 
#undef ACTION_8 
#undef ACTION_9 
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14

#undef CONDITION_A 
#undef CONDITION_B 
#undef CONDITION_N1
#undef CONDITION_N2
#undef CONDITION_N3
#undef CONDITION_N4
#undef CONDITION_N5

        n_labels_ = LabelsSolver::Flatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *img_labels_row_end = img_labels_row_start + img_labels_.cols;
            unsigned int *img_labels_row = img_labels_row_start;
            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
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

        // Scan Mask
        // +--+--+--+
        // |n1|n2|n3|
        // +--+--+--+
        // |n4|a |
        // +--+--+
        // |n5|b |
        // +--+--+

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

        for (int r = 0; r < h; r += 2) {

            int prob_fol_state = BR;
            int prev_state = BR;

            for (int c = 0; c < w; c += 1) {

                // A bunch of defines used to check if the pixels are foreground, and current state of graph
                // without going outside the image limits.

#define CONDITION_A img(r,c)>0
#define CONDITION_B r+1<h && img(r + 1, c)>0
#define CONDITION_N1 c-1>=0 && r-1>=0 && img(r - 1, c - 1)>0
#define CONDITION_N2 r-1>=0 && img(r - 1, c)>0
#define CONDITION_N3 r-1>=0 && c+1<w && img(r - 1, c + 1)>0
#define CONDITION_N4 c-1>=0 && img(r, c - 1)>0
#define CONDITION_N5 c-1>=0 && r+1<h && img(r + 1, c - 1)>0

#define ACTION_1 // nothing
#define ACTION_2 img_labels(r, c) = LabelsSolver::MemNewLabel(); // new label
#define ACTION_3 img_labels(r, c) = img_labels(r - 1, c - 1); // a <- n1
#define ACTION_4 img_labels(r, c) = img_labels(r - 1, c); // a <- n2
#define ACTION_5 img_labels(r, c) = img_labels(r - 1, c + 1); // a <- n3
#define ACTION_6 img_labels(r, c) = img_labels(r, c - 1); // a <- n4
#define ACTION_7 img_labels(r, c) = img_labels(r + 1, c - 1); // a <- n5
#define ACTION_8 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r - 1, c - 1)); // a + n1
#define ACTION_9 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r - 1, c + 1)); // a + n3
#define ACTION_10 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r + 1, c - 1)); // a + n5
#define ACTION_11 img_labels(r + 1, c) = LabelsSolver::MemNewLabel(); // b new label
#define ACTION_12 img_labels(r + 1, c) = img_labels(r, c - 1); // b <- n4
#define ACTION_13 img_labels(r + 1, c) = img_labels(r + 1, c - 1); // b <- n5
#define ACTION_14 img_labels(r + 1, c) = img_labels(r, c); // b <- a

#include "labeling_he_2014_graph.inc"
            }//End columns's for
        }//End rows's for

#undef ACTION_1 
#undef ACTION_2 
#undef ACTION_3 
#undef ACTION_4 
#undef ACTION_5 
#undef ACTION_6 
#undef ACTION_7 
#undef ACTION_8 
#undef ACTION_9 
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14

#undef CONDITION_A 
#undef CONDITION_B 
#undef CONDITION_N1
#undef CONDITION_N2
#undef CONDITION_N3
#undef CONDITION_N4
#undef CONDITION_N5

        n_labels_ = LabelsSolver::MemFlatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels.rows; ++r_i) {
            for (int c_i = 0; c_i < img_labels.cols; ++c_i) {
                img_labels(r_i,c_i) = LabelsSolver::MemGetLabel(img_labels(r_i, c_i));
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

        // Scan Mask
        // +--+--+--+
        // |n1|n2|n3|
        // +--+--+--+
        // |n4|a |
        // +--+--+
        // |n5|b |
        // +--+--+

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);

            int prob_fol_state = BR;
            int prev_state = BR;

            for (int c = 0; c < w; c += 1) {

                // A bunch of defines used to check if the pixels are foreground, and current state of graph
                // without going outside the image limits.

#define CONDITION_A img_row[c]>0
#define CONDITION_B r+1<h && img_row_fol[c]>0
#define CONDITION_N1 c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_N2 r-1>=0 && img_row_prev[c]>0
#define CONDITION_N3 r-1>=0 && c+1<w && img_row_prev[c+1]>0
#define CONDITION_N4 c-1>=0 && img_row[c-1]>0
#define CONDITION_N5 c-1>=0 && r+1<h && img_row_fol[c-1]>0

#define ACTION_1 // nothing
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // a <- n1
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // a <- n2
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // a <- n3
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // a <- n4
#define ACTION_7 img_labels_row[c] = img_labels_row_fol[c - 1]; // a <- n5
#define ACTION_8 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c - 1]); // a + n1
#define ACTION_9 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c + 1]); // a + n3
#define ACTION_10 LabelsSolver::Merge(img_labels_row[c], img_labels_row_fol[c - 1]); // a + n5
#define ACTION_11 img_labels_row_fol[c] = LabelsSolver::NewLabel(); // b new label
#define ACTION_12 img_labels_row_fol[c] = img_labels_row[c - 1]; // b <- n4
#define ACTION_13 img_labels_row_fol[c] = img_labels_row_fol[c - 1]; // b <- n5
#define ACTION_14 img_labels_row_fol[c] = img_labels_row[c]; // b <- a

#include "labeling_he_2014_graph.inc"
            }//End columns's for
        }//End rows's for

#undef ACTION_1 
#undef ACTION_2 
#undef ACTION_3 
#undef ACTION_4 
#undef ACTION_5 
#undef ACTION_6 
#undef ACTION_7 
#undef ACTION_8 
#undef ACTION_9 
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14

#undef CONDITION_A 
#undef CONDITION_B 
#undef CONDITION_N1
#undef CONDITION_N2
#undef CONDITION_N3
#undef CONDITION_N4
#undef CONDITION_N5
    }
    void SecondScan() 
    {
        n_labels_ = LabelsSolver::Flatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *img_labels_row_end = img_labels_row_start + img_labels_.cols;
            unsigned int *img_labels_row = img_labels_row_start;
            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
            }
        }
    }
};

#endif // YACCLAB_LABELING_HE_2014_H_