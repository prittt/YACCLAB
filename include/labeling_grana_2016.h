// Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#pragma once

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
        img_labels_ = cv::Mat1i(img_.size(), 0);

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

        {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(0);
            uint* const imgLabels_row = img_labels_.ptr<uint>(0);

            int c = -1;
        tree_A0: if (++c >= w) goto break_A0;
            if (CONDITION_X) {
                // x = new label
                imgLabels_row[c] = LabelsSolver::NewLabel();
                goto tree_B0;
            }
            else {
                // nothing
                goto tree_A0;
            }
        tree_B0: if (++c >= w) goto break_B0;
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                goto tree_B0;
            }
            else {
                // nothing
                goto tree_A0;
            }
        break_A0:
        break_B0:;
        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            uint* const imgLabels_row = img_labels_.ptr<uint>(r);
            uint* const imgLabels_row_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0]);

            // First column
            int c = 0;

            // It is also the last column? If yes skip to the specific 
            // tree. This is necessary to handle one column vector images
            if (c == w - 1) goto one_col;

            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
                        goto tree_B;
                    }
                    else {
                        // x = new label
                        imgLabels_row[c] = LabelsSolver::NewLabel();
                        goto tree_C;
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }

        tree_A: if (++c >= w - 1) goto break_A;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
                        goto tree_B;
                    }
                    else {
                        imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                        goto tree_C;
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_B: if (++c >= w - 1) goto break_B;
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                goto tree_A;
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_C: if (++c >= w - 1) goto break_C;
            if (CONDITION_X) {
                if (CONDITION_R) {
                    imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
                    goto tree_B;
                }
                else {
                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                    goto tree_C;
                }
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_D: if (++c >= w - 1) goto break_D;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        if (CONDITION_P) {
                            imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]); // x = p + r
                            goto tree_B;
                        }
                        else {
                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
                            goto tree_B;
                        }
                    }
                    else {
                        if (CONDITION_P) {
                            imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
                            goto tree_C;
                        }
                        else {
                            // x = new label
                            imgLabels_row[c] = LabelsSolver::NewLabel();
                            goto tree_C;
                        }
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }


            // Last column
        break_A:
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                }
            }
            continue;
        break_B:
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
            }
            continue;
        break_C:
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
            }
            continue;
        break_D:
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    if (CONDITION_P) {
                        imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
                    }
                    else {
                        // x = new label
                        imgLabels_row[c] = LabelsSolver::NewLabel();
                    }
                }
            }
            continue;
        one_col:
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    // x = new label
                    imgLabels_row[c] = LabelsSolver::NewLabel();
                }
            }
        }//End rows's for

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

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
    }

    void PerformLabelingWithSteps()
    {
        perf_.start();
        Alloc();
        perf_.stop();
        double alloc_timing = perf_.last();

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
    }

private:
    void Alloc()
    {
        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        img_labels_ = cv::Mat1i(img_.size()); // Memory allocation of the output image
    }
    void Dealloc()
    {
        LabelsSolver::Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        img_labels_ = cv::Mat1i::zeros(img_.size()); // Initialization
        LabelsSolver::Setup();

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img_row[c]>0
#define CONDITION_P img_row_prev[c-1]>0
#define CONDITION_Q img_row_prev[c]>0
#define CONDITION_R img_row_prev[c+1]>0

#define ACTION_1 // nothing to do 
#define ACTION_2 imgLabels_row[c] = LabelsSolver::NewLabel(); // new label
#define ACTION_3 img_labels(r, c) = img_labels(r - 1, c - 1); // x <- p
#define ACTION_4 img_labels(r, c) = img_labels(r - 1, c); // x <- q
#define ACTION_5 img_labels(r, c) = img_labels(r - 1, c + 1); // x <- r
#define ACTION_6 imgLabels_row[c] = imgLabels_row[c - 1]; // x <- s
#define ACTION_7 img_labels(r, c) = LabelsSolver::MemMerge((unsigned)img_labels(r - 1, c - 1), (unsigned)img_labels(r - 1, c + 1)); // x <- p + r
#define ACTION_8 img_labels(r, c) = LabelsSolver::MemMerge((unsigned)img_labels(r, c - 1), (unsigned)img_labels(r - 1, c + 1)); // x <- s + r

        {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(0);
            uint* const imgLabels_row = img_labels_.ptr<uint>(0);

            int c = -1;
        tree_A0: if (++c >= w) goto break_A0;
            if (CONDITION_X) {
                // new label
                ACTION_2
                goto tree_B0;
            }
            else {
                // nothing
                ACTION_1
                goto tree_A0;
            }
        tree_B0: if (++c >= w) goto break_B0;
            if (CONDITION_X) {
                // x <- s
                ACTION_6
                goto tree_B0;
            }
            else {
                // nothing
                ACTION_1
                goto tree_A0;
            }
        break_A0:
        break_B0:;
        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            uint* const imgLabels_row = img_labels_.ptr<uint>(r);
            uint* const imgLabels_row_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0]);
        
            int c = 0; // First column

        /*tree_0:*/ if (c == w - 1) goto break_0;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
                        goto tree_B;
                    }
                    else {
                        // x = new label
                        imgLabels_row[c] = LabelsSolver::NewLabel();
                        goto tree_C;
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }

        tree_A: if (++c >= w - 1) goto break_A;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
                        goto tree_B;
                    }
                    else {
                        imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                        goto tree_C;
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_B: if (++c >= w - 1) goto break_B;
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                goto tree_A;
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_C: if (++c >= w - 1) goto break_C;
            if (CONDITION_X) {
                if (CONDITION_R) {
                    imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
                    goto tree_B;
                }
                else {
                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                    goto tree_C;
                }
            }
            else {
                // nothing
                goto tree_D;
            }
        tree_D: if (++c >= w - 1) goto break_D;
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                    goto tree_A;
                }
                else {
                    if (CONDITION_R) {
                        if (CONDITION_P) {
                            imgLabels_row[c] = LabelsSolver::Merge(imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]); // x = p + r
                            goto tree_B;
                        }
                        else {
                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
                            goto tree_B;
                        }
                    }
                    else {
                        if (CONDITION_P) {
                            imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
                            goto tree_C;
                        }
                        else {
                            // x = new label
                            imgLabels_row[c] = LabelsSolver::NewLabel();
                            goto tree_C;
                        }
                    }
                }
            }
            else {
                // nothing
                goto tree_D;
            }


            // Last column
        break_A:
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
                }
            }
            continue;
        break_B:
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
            }
            continue;
        break_C:
            if (CONDITION_X) {
                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
            }
            continue;
        break_D:
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    if (CONDITION_P) {
                        imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
                    }
                    else {
                        // x = new label
                        imgLabels_row[c] = LabelsSolver::NewLabel();
                    }
                }
            }
            continue;
        break_0:
            // This tree is necessary to handle (only) one column vector images
            if (CONDITION_X) {
                if (CONDITION_Q) {
                    imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
                }
                else {
                    // x = new label
                    imgLabels_row[c] = LabelsSolver::NewLabel();
                }
            }
        }//End rows's for
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

//void PRED::AllocateMemory()
//{
//    const size_t Plength = UPPER_BOUND_8_CONNECTIVITY;
//    P = (unsigned *)fastMalloc(sizeof(unsigned) * Plength); //array P for equivalences resolution
//    return;
//}
//
//void PRED::DeallocateMemory()
//{
//    fastFree(P);
//    return;
//}
//
//unsigned PRED::FirstScan()
//{
//    const int w(img_.cols), h(img_.rows);
//
//    img_labels_ = cv::Mat1i(img_.size(), 0);
//
//    //Background
//    P[0] = 0;
//    unsigned lunique = 1;
//
//#define condition_x img_row[c]>0
//#define condition_p img_row_prev[c-1]>0
//#define condition_q img_row_prev[c]>0
//#define condition_r img_row_prev[c+1]>0
//
//    {
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(0);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(0);
//
//        int c = -1;
//    tree_A0: if (++c >= w) goto break_A0;
//        if (condition_x)
//        {
//            // x = new label
//            imgLabels_row[c] = lunique;
//            P[lunique] = lunique;
//            lunique++;
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    tree_B0: if (++c >= w) goto break_B0;
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    break_A0:
//    break_B0:;
//    }
//
//    for (int r = 1; r < h; ++r)
//    {
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(r);
//        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(r);
//        unsigned* const imgLabels_row_prev = (unsigned *)(((char *)imgLabels_row) - img_labels_.step.p[0]);
//
//        // First column
//        int c = 0;
//
//        // It is also the last column? If yes skip to the specific tree (necessary to handle one column vector image)
//        if (c == w - 1) goto one_col;
//
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
//                    goto tree_B;
//                }
//                else
//                {
//                    // x = new label
//                    imgLabels_row[c] = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//    tree_A: if (++c >= w - 1) goto break_A;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
//                    goto tree_B;
//                }
//                else
//                {
//                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_B: if (++c >= w - 1) goto break_B;
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            goto tree_A;
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_C: if (++c >= w - 1) goto break_C;
//        if (condition_x)
//        {
//            if (condition_r)
//            {
//                imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
//                goto tree_B;
//            }
//            else
//            {
//                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//                goto tree_C;
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_D: if (++c >= w - 1) goto break_D;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    if (condition_p)
//                    {
//                        imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]); // x = p + r
//                        goto tree_B;
//                    }
//                    else
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
//                        goto tree_B;
//                    }
//                }
//                else
//                {
//                    if (condition_p)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
//                        goto tree_C;
//                    }
//                    else
//                    {
//                        // x = new label
//                        imgLabels_row[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                        goto tree_C;
//                    }
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//        // Last column
//    break_A:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//            }
//        }
//        continue;
//    break_B:
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//        }
//        continue;
//    break_C:
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//        }
//        continue;
//    break_D:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                if (condition_p)
//                {
//                    imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
//                }
//                else
//                {
//                    // x = new label
//                    imgLabels_row[c] = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                }
//            }
//        }
//        continue;
//    one_col:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                // x = new label
//                imgLabels_row[c] = lunique;
//                P[lunique] = lunique;
//                lunique++;
//            }
//        }
//    }//End rows's for
//
//#undef condition_x
//#undef condition_p
//#undef condition_q
//#undef condition_r
//
//    return lunique;
//}
//
//unsigned PRED::SecondScan(const unsigned& lunique)
//{
//    unsigned nLabels = flattenL(P, lunique);
//
//    // second scan
//    for (int r_i = 0; r_i < img_labels_.rows; ++r_i)
//    {
//        unsigned *b = img_labels_.ptr<unsigned>(r_i);
//        unsigned *e = b + img_labels_.cols;
//        for (; b != e; ++b)
//        {
//            *b = P[*b];
//        }
//    }
//
//    return nLabels;
//}
//
//unsigned PRED::PerformLabeling()
//{
//    const int h = img_.rows;
//    const int w = img_.cols;
//
//    img_labels_ = Mat1i(img_.size(), 0);
//
//    P[0] = 0;	//first label is for background pixels
//    unsigned lunique = 1;
//
//#define condition_x img_row[c]>0
//#define condition_p img_row_prev[c-1]>0
//#define condition_q img_row_prev[c]>0
//#define condition_r img_row_prev[c+1]>0
//
//    {
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(0);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(0);
//
//        int c = -1;
//    tree_A0: if (++c >= w) goto break_A0;
//        if (condition_x)
//        {
//            // x = new label
//            imgLabels_row[c] = lunique;
//            P[lunique] = lunique;
//            lunique++;
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    tree_B0: if (++c >= w) goto break_B0;
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    break_A0:
//    break_B0:;
//    }
//
//    for (int r = 1; r < h; ++r)
//    {
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(r);
//        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(r);
//        unsigned* const imgLabels_row_prev = (unsigned *)(((char *)imgLabels_row) - img_labels_.step.p[0]);
//
//        // First column
//        int c = 0;
//
//        // It is also the last column? If yes skip to the specific tree (necessary to handle one column vector image)
//        if (c == w - 1) goto one_col;
//
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
//                    goto tree_B;
//                }
//                else
//                {
//                    // x = new label
//                    imgLabels_row[c] = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//    tree_A: if (++c >= w - 1) goto break_A;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
//                    goto tree_B;
//                }
//                else
//                {
//                    imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_B: if (++c >= w - 1) goto break_B;
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            goto tree_A;
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_C: if (++c >= w - 1) goto break_C;
//        if (condition_x)
//        {
//            if (condition_r)
//            {
//                imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
//                goto tree_B;
//            }
//            else
//            {
//                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//                goto tree_C;
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_D: if (++c >= w - 1) goto break_D;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    if (condition_p)
//                    {
//                        imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]); // x = p + r
//                        goto tree_B;
//                    }
//                    else
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
//                        goto tree_B;
//                    }
//                }
//                else
//                {
//                    if (condition_p)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
//                        goto tree_C;
//                    }
//                    else
//                    {
//                        // x = new label
//                        imgLabels_row[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                        goto tree_C;
//                    }
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//        // Last column
//    break_A:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//            }
//        }
//        continue;
//    break_B:
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//        }
//        continue;
//    break_C:
//        if (condition_x)
//        {
//            imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
//        }
//        continue;
//    break_D:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                if (condition_p)
//                {
//                    imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
//                }
//                else
//                {
//                    // x = new label
//                    imgLabels_row[c] = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                }
//            }
//        }
//        continue;
//    one_col:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
//            }
//            else
//            {
//                // x = new label
//                imgLabels_row[c] = lunique;
//                P[lunique] = lunique;
//                lunique++;
//            }
//        }
//    }//End rows's for
//
//#undef condition_x
//#undef condition_p
//#undef condition_q
//#undef condition_r
//
//     //second scan
//    unsigned nLabel = flattenL(P, lunique);
//
//    // second scan
//    for (int r_i = 0; r_i < h; ++r_i)
//    {
//        unsigned *b = img_labels_.ptr<unsigned>(r_i);
//        unsigned *e = b + w;
//        for (; b != e; ++b)
//        {
//            *b = P[*b];
//        }
//    }
//
//    return nLabel;
//}
//
//unsigned PRED::PerformLabelingMem(vector<unsigned long>& accesses)
//{
//    int w(img_.cols), h(img_.rows);
//
//    MemMat<uchar> img(img_);
//    MemMat<int> img_labels_(img_.size(), 0); // memset is used
//
//                                             //A quick and dirty upper bound for the maximimum number of labels (only for 8-connectivity).
//                                             //const size_t Plength = (img_origin.rows + 1)*(img_origin.cols + 1) / 4 + 1; // Oversized in some cases
//    const size_t Plength = UPPER_BOUND_8_CONNECTIVITY;
//
//    //Tree of labels
//    MemVector<unsigned> P(Plength);
//    //Background
//    P[0] = 0;
//    unsigned lunique = 1;
//
//#define condition_x img_(r,c)>0
//#define condition_p img_(r-1,c-1)>0
//#define condition_q img_(r-1,c)>0
//#define condition_r img_(r-1,c+1)>0
//
//    {
//        int r = 0;
//        int c = -1;
//    tree_A0: if (++c >= w) goto break_A0;
//        if (condition_x)
//        {
//            // x = new label
//            img_labels_(r, c) = lunique;
//            P[lunique] = lunique;
//            lunique++;
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    tree_B0: if (++c >= w) goto break_B0;
//        if (condition_x)
//        {
//            img_labels_(r, c) = img_labels_(r, c - 1); // x = s
//            goto tree_B0;
//        }
//        else
//        {
//            // nothing
//            goto tree_A0;
//        }
//    break_A0:
//    break_B0:;
//    }
//
//    for (int r = 1; r < h; ++r)
//    {
//        // First column
//        int c = 0;
//
//        // It is also the last column? If yes skip to the specific tree (necessary to handle one column vector image)
//        if (c == w - 1) goto one_col;
//
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    img_labels_(r, c) = img_labels_(r - 1, c + 1); // x = r
//                    goto tree_B;
//                }
//                else
//                {
//                    // x = new label
//                    img_labels_(r, c) = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//    tree_A: if (++c >= w - 1) goto break_A;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    img_labels_(r, c) = set_union(P, (unsigned)img_labels_(r - 1, c + 1), (unsigned)img_labels_(r, c - 1)); // x = r + s
//                    goto tree_B;
//                }
//                else
//                {
//                    img_labels_(r, c) = img_labels_(r, c - 1); // x = s
//                    goto tree_C;
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_B: if (++c >= w - 1) goto break_B;
//        if (condition_x)
//        {
//            img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//            goto tree_A;
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_C: if (++c >= w - 1) goto break_C;
//        if (condition_x)
//        {
//            if (condition_r)
//            {
//                img_labels_(r, c) = set_union(P, (unsigned)img_labels_(r - 1, c + 1), (unsigned)img_labels_(r, c - 1)); // x = r + s
//                goto tree_B;
//            }
//            else
//            {
//                img_labels_(r, c) = img_labels_(r, c - 1); // x = s
//                goto tree_C;
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//    tree_D: if (++c >= w - 1) goto break_D;
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//                goto tree_A;
//            }
//            else
//            {
//                if (condition_r)
//                {
//                    if (condition_p)
//                    {
//                        img_labels_(r, c) = set_union(P, (unsigned)img_labels_(r - 1, c - 1), (unsigned)img_labels_(r - 1, c + 1)); // x = p + r
//                        goto tree_B;
//                    }
//                    else
//                    {
//                        img_labels_(r, c) = img_labels_(r - 1, c + 1); // x = r
//                        goto tree_B;
//                    }
//                }
//                else
//                {
//                    if (condition_p)
//                    {
//                        img_labels_(r, c) = img_labels_(r - 1, c - 1); // x = p
//                        goto tree_C;
//                    }
//                    else
//                    {
//                        // x = new label
//                        img_labels_(r, c) = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                        goto tree_C;
//                    }
//                }
//            }
//        }
//        else
//        {
//            // nothing
//            goto tree_D;
//        }
//
//        // Last column
//    break_A:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//            }
//            else
//            {
//                img_labels_(r, c) = img_labels_(r, c - 1); // x = s
//            }
//        }
//        continue;
//    break_B:
//        if (condition_x)
//        {
//            img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//        }
//        continue;
//    break_C:
//        if (condition_x)
//        {
//            img_labels_(r, c) = img_labels_(r, c - 1); // x = s
//        }
//        continue;
//    break_D:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//            }
//            else
//            {
//                if (condition_p)
//                {
//                    img_labels_(r, c) = img_labels_(r - 1, c - 1); // x = p
//                }
//                else
//                {
//                    // x = new label
//                    img_labels_(r, c) = lunique;
//                    P[lunique] = lunique;
//                    lunique++;
//                }
//            }
//        }
//        continue;
//    one_col:
//        if (condition_x)
//        {
//            if (condition_q)
//            {
//                img_labels_(r, c) = img_labels_(r - 1, c); // x = q
//            }
//            else
//            {
//                // x = new label
//                img_labels_(r, c) = lunique;
//                P[lunique] = lunique;
//                lunique++;
//            }
//        }
//    }//End rows's for
//
//#undef condition_x
//#undef condition_p
//#undef condition_q
//#undef condition_r
//
//    unsigned nLabel = flattenL(P, lunique);
//
//    // second scan
//    for (int r_i = 0; r_i < img_labels_.rows; ++r_i)
//    {
//        for (int c_i = 0; c_i < img_labels_.cols; ++c_i)
//        {
//            img_labels_(r_i, c_i) = P[img_labels_(r_i, c_i)];
//        }
//    }
//
//    // Store total accesses in the output vector 'accesses'
//    accesses = vector<unsigned long int>((int)MD_SIZE, 0);
//
//    accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAcesses();
//    accesses[MD_LABELED_MAT] = (unsigned long int)img_labels_.GetTotalAcesses();
//    accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)P.GetTotalAcesses();
//
//    //a = img_labels_.getImage();
//
//    return nLabel;
//}