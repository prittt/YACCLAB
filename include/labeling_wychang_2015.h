// Copyright(c) 2016 - 2019 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

// Copyright (C) 2015 - Wan-Yu Chang and Chung-Cheng Chiu
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
// 
// The "free" of library only licensed for the purposes of research and study. 
// For further information or business co-operation, please contact us at
// Wan-Yu Chang and Chung-Cheng Chiu - Chung Cheng Institute of Technology of National Defense University.
// No.75, Shiyuan Rd., Daxi Township, Taoyuan County 33551, Taiwan (R.O.C.)  - e-mail: david.cc.chiu@gmail.com 
//
// Specially thank for the help of Prof. Grana who provide his source code of the BBDT algorithm.

#ifndef YACCLAB_LABELING_WYCHANG_2015_H_
#define YACCLAB_LABELING_WYCHANG_2015_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class CCIT : public Labeling2D<CONN_8> {
public:
    CCIT() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0);

        int w = img_labels_.cols;
        int h = img_labels_.rows;

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        int lx, u, v, k;

#define CONDITION_B1 img_row[x] > 0 
#define CONDITION_B2 x+1<w && img_row[x+1] > 0              // WRONG in the original code -> add missing condition 
#define CONDITION_B3 y+1<h && img_row_fol[x] > 0            // WRONG in the original code -> add missing condition
#define CONDITION_B4 x+1<w && y+1<h && img_row_fol[x+1] > 0 // WRONG in the original code -> add missing condition
#define CONDITION_U1 x-1>0 && img_row_prev[x-1] > 0         // WRONG in the original code -> add missing condition
#define CONDITION_U2 img_row_prev[x] > 0
#define CONDITION_U3 x+1<w && img_row_prev[x+1] > 0         // WRONG in the original code -> add missing condition
#define CONDITION_U4 x+2<w && img_row_prev[x+2] > 0         // WRONG in the original code -> add missing condition
#define ASSIGN_S img_labels_row[x] = img_labels_row[x-2]
#define ASSIGN_P img_labels_row[x] = img_labels_row_prev_prev[x-2]
#define ASSIGN_Q img_labels_row[x] = img_labels_row_prev_prev[x]
#define ASSIGN_R img_labels_row[x] = img_labels_row_prev_prev[x+2]
#define ASSIGN_LX img_labels_row[x] = lx
#define LOAD_LX u = lx
#define LOAD_PU u = img_labels_row_prev_prev[x-2]
#define LOAD_PV v = img_labels_row_prev_prev[x-2]
#define LOAD_QU u = img_labels_row_prev_prev[x]
#define LOAD_QV v = img_labels_row_prev_prev[x]
#define LOAD_QK k = img_labels_row_prev_prev[x]
#define LOAD_RV v = img_labels_row_prev_prev[x+2]
#define LOAD_RK k = img_labels_row_prev_prev[x+2]
#define NEW_LABEL lx = img_labels_row[x] = LabelsSolver::NewLabel();
#define RESOLVE_2(u, v) LabelsSolver::Merge(u,v);
#define RESOLVE_3(u, v, k) LabelsSolver::Merge(u,LabelsSolver::Merge(v,k));

        bool nextprocedure2;

        int y = 0; // Extract from the first for
        const unsigned char* const img_row = img_.ptr<unsigned char>(y);
        const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
        unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);

        // Process first two rows
        for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree_0.inc.h"

        }

        for (int y = 2; y < h; y += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
            unsigned int* const img_labels_row_prev_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
            for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree.inc.h"

            }
        }

        // Second scan (changed with better performing strategy to handle odd rows and columns)
        n_labels_ = LabelsSolver::Flatten();

        int e_rows = img_labels_.rows & 0xfffffffe;
        bool o_rows = img_labels_.rows % 2 == 1;
        int e_cols = img_labels_.cols & 0xfffffffe;
        bool o_cols = img_labels_.cols % 2 == 1;

        int r = 0;
        for (; r < e_rows; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                    if (img_row_fol[c + 1] > 0)
                        img_labels_row_fol[c + 1] = iLabel;
                    else
                        img_labels_row_fol[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                    img_labels_row_fol[c] = 0;
                    img_labels_row_fol[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row_fol[c] = 0;
                }
            }
        }
        // Last row if the number of rows is odd
        if (o_rows) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                }
            }
        }

        LabelsSolver::Dealloc();

#undef CONDITION_B1 
#undef CONDITION_B2 
#undef CONDITION_B3 
#undef CONDITION_B4 
#undef CONDITION_U1 
#undef CONDITION_U2 
#undef CONDITION_U3 
#undef CONDITION_U4 
#undef ASSIGN_S
#undef ASSIGN_P
#undef ASSIGN_Q
#undef ASSIGN_R
#undef ASSIGN_LX 
#undef LOAD_LX 
#undef LOAD_PU 
#undef LOAD_PV 
#undef LOAD_QU 
#undef LOAD_QV 
#undef LOAD_QK 
#undef LOAD_RV 
#undef LOAD_RK 
#undef NEW_LABEL
#undef RESOLVE_2
#undef RESOLVE_3
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

    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    {

        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::MemSetup();

        // First scan
        int w(img_.cols);
        int h(img_.rows);

        int lx, u, v, k;

#define CONDITION_B1 img(y, x) > 0 
#define CONDITION_B2 x+1<w && img(y, x+1) > 0                // WRONG in the original code -> add missing condition 
#define CONDITION_B3 y+1<h && img(y + 1, x) > 0              // WRONG in the original code -> add missing condition
#define CONDITION_B4 x+1<w && y+1<h && img(y + 1, x + 1) > 0 // WRONG in the original code -> add missing condition
#define CONDITION_U1 x-1>0 && img(y - 1, x - 1) > 0          // WRONG in the original code -> add missing condition
#define CONDITION_U2 img(y - 1, x) > 0
#define CONDITION_U3 x+1<w && img(y - 1, x + 1) > 0          // WRONG in the original code -> add missing condition
#define CONDITION_U4 x+2<w && img(y - 1, x + 2) > 0          // WRONG in the original code -> add missing condition
#define ASSIGN_S  img_labels(y, x) = img_labels(y, x - 2)
#define ASSIGN_P  img_labels(y, x) = img_labels(y - 2, x - 2)
#define ASSIGN_Q  img_labels(y, x) = img_labels(y - 2, x)
#define ASSIGN_R  img_labels(y, x) = img_labels(y - 2, x + 2)
#define ASSIGN_LX img_labels(y, x) = lx
#define LOAD_LX u = lx
#define LOAD_PU u = img_labels(y - 2, x-2)
#define LOAD_PV v = img_labels(y - 2, x-2)
#define LOAD_QU u = img_labels(y - 2, x)
#define LOAD_QV v = img_labels(y - 2, x)
#define LOAD_QK k = img_labels(y - 2, x)
#define LOAD_RV v = img_labels(y - 2, x+2)
#define LOAD_RK k = img_labels(y - 2, x+2)
#define NEW_LABEL lx = img_labels(y, x) = LabelsSolver::MemNewLabel();
#define RESOLVE_2(u, v) LabelsSolver::MemMerge(u,v);
#define RESOLVE_3(u, v, k) LabelsSolver::MemMerge(u,LabelsSolver::MemMerge(v,k));

        bool nextprocedure2;

        int y = 0; // Extract from the first for
        // Process first two rows
        for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree_0.inc.h"

        }

        for (int y = 2; y < h; y += 2) {
            for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree.inc.h"

            }
        }

        n_labels_ = LabelsSolver::MemFlatten();

        // Second scan
        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {
                int iLabel = img_labels(r, c);
                if (iLabel > 0) {
                    iLabel = LabelsSolver::MemGetLabel(iLabel);
                    if (img(r, c) > 0)
                        img_labels(r, c) = iLabel;
                    else
                        img_labels(r, c) = 0;
                    if (c + 1 < w) {
                        if (img(r, c + 1) > 0)
                            img_labels(r, c + 1) = iLabel;
                        else
                            img_labels(r, c + 1) = 0;
                        if (r + 1 < h) {
                            if (img(r + 1, c) > 0)
                                img_labels(r + 1, c) = iLabel;
                            else
                                img_labels(r + 1, c) = 0;
                            if (img(r + 1, c + 1) > 0)
                                img_labels(r + 1, c + 1) = iLabel;
                            else
                                img_labels(r + 1, c + 1) = 0;
                        }
                    }
                    else if (r + 1 < h) {
                        if (img(r + 1, c) > 0)
                            img_labels(r + 1, c) = iLabel;
                        else
                            img_labels(r + 1, c) = 0;
                    }
                }
                else {
                    img_labels(r, c) = 0;
                    if (c + 1 < w) {
                        img_labels(r, c + 1) = 0;
                        if (r + 1 < h) {
                            img_labels(r + 1, c) = 0;
                            img_labels(r + 1, c + 1) = 0;
                        }
                    }
                    else if (r + 1 < h) {
                        img_labels(r + 1, c) = 0;
                    }
                }
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        LabelsSolver::MemDealloc();

#undef CONDITION_B1 
#undef CONDITION_B2 
#undef CONDITION_B3 
#undef CONDITION_B4 
#undef CONDITION_U1 
#undef CONDITION_U2 
#undef CONDITION_U3 
#undef CONDITION_U4 
#undef ASSIGN_S
#undef ASSIGN_P
#undef ASSIGN_Q
#undef ASSIGN_R
#undef ASSIGN_LX 
#undef LOAD_LX 
#undef LOAD_PU 
#undef LOAD_PV 
#undef LOAD_QU 
#undef LOAD_QV 
#undef LOAD_QK 
#undef LOAD_RV 
#undef LOAD_RK 
#undef NEW_LABEL
#undef RESOLVE_2
#undef RESOLVE_3
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
        int w = img_labels_.cols;
        int h = img_labels_.rows;

        int lx, u, v, k;

#define CONDITION_B1 img_row[x] > 0 
#define CONDITION_B2 x+1<w && img_row[x+1] > 0              // WRONG in the original code -> add missing condition 
#define CONDITION_B3 y+1<h && img_row_fol[x] > 0            // WRONG in the original code -> add missing condition
#define CONDITION_B4 x+1<w && y+1<h && img_row_fol[x+1] > 0 // WRONG in the original code -> add missing condition
#define CONDITION_U1 x-1>0 && img_row_prev[x-1] > 0         // WRONG in the original code -> add missing condition
#define CONDITION_U2 img_row_prev[x] > 0
#define CONDITION_U3 x+1<w && img_row_prev[x+1] > 0         // WRONG in the original code -> add missing condition
#define CONDITION_U4 x+2<w && img_row_prev[x+2] > 0         // WRONG in the original code -> add missing condition
#define ASSIGN_S img_labels_row[x] = img_labels_row[x-2]
#define ASSIGN_P img_labels_row[x] = img_labels_row_prev_prev[x-2]
#define ASSIGN_Q img_labels_row[x] = img_labels_row_prev_prev[x]
#define ASSIGN_R img_labels_row[x] = img_labels_row_prev_prev[x+2]
#define ASSIGN_LX img_labels_row[x] = lx
#define LOAD_LX u = lx
#define LOAD_PU u = img_labels_row_prev_prev[x-2]
#define LOAD_PV v = img_labels_row_prev_prev[x-2]
#define LOAD_QU u = img_labels_row_prev_prev[x]
#define LOAD_QV v = img_labels_row_prev_prev[x]
#define LOAD_QK k = img_labels_row_prev_prev[x]
#define LOAD_RV v = img_labels_row_prev_prev[x+2]
#define LOAD_RK k = img_labels_row_prev_prev[x+2]
#define NEW_LABEL lx = img_labels_row[x] = LabelsSolver::NewLabel();
#define RESOLVE_2(u, v) LabelsSolver::Merge(u,v);
#define RESOLVE_3(u, v, k) LabelsSolver::Merge(u,LabelsSolver::Merge(v,k));

        bool nextprocedure2;

        int y = 0; // Extract from the first for
        const unsigned char* const img_row = img_.ptr<unsigned char>(y);
        const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
        unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);

        // Process first two rows
        for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree_0.inc.h"

        }

        for (int y = 2; y < h; y += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
            unsigned int* const img_labels_row_prev_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
            for (int x = 0; x < w; x += 2) {

#include "labeling_wychang_2015_tree.inc.h"

            }
        }

#undef CONDITION_B1 
#undef CONDITION_B2 
#undef CONDITION_B3 
#undef CONDITION_B4 
#undef CONDITION_U1 
#undef CONDITION_U2 
#undef CONDITION_U3 
#undef CONDITION_U4 
#undef ASSIGN_S
#undef ASSIGN_P
#undef ASSIGN_Q
#undef ASSIGN_R
#undef ASSIGN_LX 
#undef LOAD_LX 
#undef LOAD_PU 
#undef LOAD_PV 
#undef LOAD_QU 
#undef LOAD_QV 
#undef LOAD_QK 
#undef LOAD_RV 
#undef LOAD_RK 
#undef NEW_LABEL
#undef RESOLVE_2
#undef RESOLVE_3
    }

    void SecondScan()
    {
        // Second scan (changed with better performing strategy to handle odd rows and columns)
        n_labels_ = LabelsSolver::Flatten();

        int e_rows = img_labels_.rows & 0xfffffffe;
        bool o_rows = img_labels_.rows % 2 == 1;
        int e_cols = img_labels_.cols & 0xfffffffe;
        bool o_cols = img_labels_.cols % 2 == 1;

        int r = 0;
        for (; r < e_rows; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                    if (img_row_fol[c + 1] > 0)
                        img_labels_row_fol[c + 1] = iLabel;
                    else
                        img_labels_row_fol[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                    img_labels_row_fol[c] = 0;
                    img_labels_row_fol[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row_fol[c] = 0;
                }
            }
        }
        // Last row if the number of rows is odd
        if (o_rows) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                }
            }
        }
    }
};

#endif //YACCLAB_LABELING_WYCHANG_2015_H_