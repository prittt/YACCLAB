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

#ifndef YACCLAB_LABELING_GRANA_2010_H_
#define YACCLAB_LABELING_GRANA_2010_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class BBDT : public Labeling {
public:
    BBDT() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size()); // Memory allocation for the output image

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        LabelsSolver::Setup(); // Labels solver initialization

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only matter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        // First scan
        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_prev_prev = (unsigned char *)(((char *)img_row_prev) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define CONDITION_C r-2>=0 && img_row_prev_prev[c]>0
#define CONDITION_D c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define CONDITION_E c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define CONDITION_G c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define CONDITION_H c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_I r-1>=0 && img_row_prev[c]>0
#define CONDITION_J c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define CONDITION_K c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define CONDITION_M c-2>=0 && img_row[c-2]>0
#define CONDITION_N c-1>=0 && img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P c+1<w && img_row[c+1]>0

#define CONDITION_R c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define CONDITION_S r+1<h && img_row_fol[c]>0
#define CONDITION_T c+1<w && r+1<h && img_row_fol[c+1]>0

                // Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; continue; 
// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); continue; 
//Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2]; continue;
// Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c]; continue;
// Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2]; continue;
// Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; continue; 
// Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]); continue;
//Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]); continue;
// Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]); continue;
// Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]); continue;
// Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]); continue;
// Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]); continue;
// Action 13: not used
#define ACTION_13 
// Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]); continue;
//Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;
//Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;

#include "labeling_grana_2010_tree.inc"
            }
        }

        // Second scan
        n_labels_ = LabelsSolver::Flatten();

        if (img_labels_.rows & 1) {
            if (img_labels_.cols & 1) {
                // Case 1: both rows and cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = LabelsSolver::GetLabel(iLabel);
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    if (img_row_fol[c] > 0)
                                        img_labels_row_fol[c] = iLabel;
                                    else
                                        img_labels_row_fol[c] = 0;
                                    if (img_row_fol[c + 1] > 0)
                                        img_labels_row_fol[c + 1] = iLabel;
                                    else
                                        img_labels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    img_labels_row_fol[c] = iLabel;
                                else
                                    img_labels_row_fol[c] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    img_labels_row_fol[c] = 0;
                                    img_labels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                img_labels_row_fol[c] = 0;
                            }
                        }
                    }
                }
            }// END Case 1
            else {
                // Case 2: only rows odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                            if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    img_labels_row_fol[c] = iLabel;
                                else
                                    img_labels_row_fol[c] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    img_labels_row_fol[c + 1] = iLabel;
                                else
                                    img_labels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            if (r + 1 < img_labels_.rows) {
                                img_labels_row_fol[c] = 0;
                                img_labels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END Case 2
        }
        else {
            if (img_labels_.cols & 1) {
                // Case 3: only cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    img_labels_row_fol[c + 1] = iLabel;
                                else
                                    img_labels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                img_labels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END case 3
            else {
                // Case 4: nothing odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                }
            }// END case 4
        }

        LabelsSolver::Dealloc();

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
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
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
        const int h = img_.rows;
        const int w = img_.cols;

        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::MemSetup();

        //Data structure for memory test
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only metter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img(r-2, c-1)>0
#define CONDITION_C r-2>=0 && img(r-2, c)>0
#define CONDITION_D c+1<w && r-2>=0 && img(r-2, c+1)>0
#define CONDITION_E c+2<w && r-2>=0 && img(r-2, c+2)>0

#define CONDITION_G c-2>=0 && r-1>=0 && img(r-1, c-2)>0
#define CONDITION_H c-1>=0 && r-1>=0 && img(r-1, c-1)>0
#define CONDITION_I r-1>=0 && img(r-1, c)>0
#define CONDITION_J c+1<w && r-1>=0 && img(r-1, c+1)>0
#define CONDITION_K c+2<w && r-1>=0 && img(r-1, c+2)>0

#define CONDITION_M c-2>=0 && img(r, c-2)>0
#define CONDITION_N c-1>=0 && img(r, c-1)>0
#define CONDITION_O img(r,c)>0
#define CONDITION_P c+1<w && img(r,c+1)>0

#define CONDITION_R c-1>=0 && r+1<h && img(r+1, c-1)>0
#define CONDITION_S r+1<h && img(r+1, c)>0
#define CONDITION_T c+1<w && r+1<h && img(r+1, c+1)>0

                // Action 1: No action
#define ACTION_1 img_labels(r, c) = 0; continue; 
// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels(r, c) = LabelsSolver::MemNewLabel(); continue; 
//Action 3: Assign label of block P
#define ACTION_3 img_labels(r, c) = img_labels(r - 2, c - 2); continue;
// Action 4: Assign label of block Q 
#define ACTION_4 img_labels(r, c) = img_labels(r - 2, c); continue;
// Action 5: Assign label of block R
#define ACTION_5 img_labels(r, c) = img_labels(r - 2, c + 2); continue;
// Action 6: Assign label of block S
#define ACTION_6 img_labels(r, c) = img_labels(r, c - 2); continue; 
// Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c)); continue;
//Action 8: Merge labels of block P and R
#define ACTION_8 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2)); continue;
// Action 9 Merge labels of block P and S
#define ACTION_9 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r, c - 2)); continue;
// Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2)); continue;
// Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r, c - 2)); continue;
// Action 12: Merge labels of block R and S
#define ACTION_12 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c + 2), img_labels(r, c - 2)); continue;
// Action 13: not used
#define ACTION_13 
// Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c)), img_labels(r, c - 2)); continue;
//Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2)), img_labels(r, c - 2)); continue;
//Action 16: labels of block Q, R and S
#define ACTION_16 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2)), img_labels(r, c - 2)); continue;

#include "labeling_grana_2010_tree.inc"
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
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        LabelsSolver::MemDealloc();

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
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
    }

private:

    void Alloc()
    {
        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        img_labels_ = cv::Mat1i(img_.size()); // Memory allocation for the output image
    }
    void Dealloc()
    {
        LabelsSolver::Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        const int h = img_.rows;
        const int w = img_.cols;
       
        LabelsSolver::Setup(); // Labels solver initialization

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only matter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        // First scan
        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_prev_prev = (unsigned char *)(((char *)img_row_prev) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define CONDITION_C r-2>=0 && img_row_prev_prev[c]>0
#define CONDITION_D c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define CONDITION_E c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define CONDITION_G c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define CONDITION_H c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_I r-1>=0 && img_row_prev[c]>0
#define CONDITION_J c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define CONDITION_K c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define CONDITION_M c-2>=0 && img_row[c-2]>0
#define CONDITION_N c-1>=0 && img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P c+1<w && img_row[c+1]>0

#define CONDITION_R c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define CONDITION_S r+1<h && img_row_fol[c]>0
#define CONDITION_T c+1<w && r+1<h && img_row_fol[c+1]>0

                // Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; continue; 
                // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); continue; 
                //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2]; continue;
                // Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c]; continue;
                // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2]; continue;
                // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; continue; 
                // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]); continue;
                //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]); continue;
                // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]); continue;
                // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]); continue;
                // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]); continue;
                // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]); continue;
                // Action 13: not used
#define ACTION_13 
                // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]); continue;
                //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;
                //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;

#include "labeling_grana_2010_tree.inc"
            }
        }

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
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
    }
    void SecondScan()
    {   // Second scan
        n_labels_ = LabelsSolver::Flatten();

        if (img_labels_.rows & 1) {
            if (img_labels_.cols & 1) {
                // Case 1: both rows and cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = LabelsSolver::GetLabel(iLabel);
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    if (img_row_fol[c] > 0)
                                        img_labels_row_fol[c] = iLabel;
                                    else
                                        img_labels_row_fol[c] = 0;
                                    if (img_row_fol[c + 1] > 0)
                                        img_labels_row_fol[c + 1] = iLabel;
                                    else
                                        img_labels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    img_labels_row_fol[c] = iLabel;
                                else
                                    img_labels_row_fol[c] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    img_labels_row_fol[c] = 0;
                                    img_labels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                img_labels_row_fol[c] = 0;
                            }
                        }
                    }
                }
            }// END Case 1
            else {
                // Case 2: only rows odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                            if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    img_labels_row_fol[c] = iLabel;
                                else
                                    img_labels_row_fol[c] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    img_labels_row_fol[c + 1] = iLabel;
                                else
                                    img_labels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            if (r + 1 < img_labels_.rows) {
                                img_labels_row_fol[c] = 0;
                                img_labels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END Case 2
        }
        else {
            if (img_labels_.cols & 1) {
                // Case 3: only cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    img_labels_row_fol[c + 1] = iLabel;
                                else
                                    img_labels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                img_labels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END case 3
            else {
                // Case 4: nothing odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const unsigned char* const img_row = img_.ptr<unsigned char>(r);
                    const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
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
                }
            }// END case 4
        }
    }
};

#endif // !YACCLAB_LABELING_GRANA_2010_H_