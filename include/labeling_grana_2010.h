// Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#include <vector>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class BBDT : public Labeling {
public:
    BBDT() {}

    unsigned PerformLabeling() override
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size());

        const size_t P_length = UPPER_BOUND_8_CONNECTIVITY;
        //array P_ for equivalences resolution
        P_ = (unsigned *)cv::fastMalloc(sizeof(unsigned) * P_length);

        P_[0] = 0;	//first label is for background pixels
        unsigned lunique = 1;

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

        // first scan
        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            const uchar* const img_row_prev_prev = (uchar *)(((char *)img_row_prev) - img_.step.p[0]);
            const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
            for (int c = 0; c < w; c += 2) {
                // We work with 2x2 blocks
                // +-+-+-+
                // |P_|Q|R|
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

#define condition_b c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define condition_c r-2>=0 && img_row_prev_prev[c]>0
#define condition_d c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define condition_e c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define condition_g c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define condition_h c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define condition_i r-1>=0 && img_row_prev[c]>0
#define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define condition_m c-2>=0 && img_row[c-2]>0
#define condition_n c-1>=0 && img_row[c-1]>0
#define condition_o img_row[c]>0
#define condition_p c+1<w && img_row[c+1]>0

#define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define condition_s r+1<h && img_row_fol[c]>0
#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0

                // This is a decision tree which allows to choose which action to
                // perform, checking as few conditions as possible.
                // Actions are available after the tree.

                if (condition_o) {
                    if (condition_n) {
                        if (condition_j) {
                            if (condition_i) {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_c) {
                                    if (condition_h) {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    //Action_11: Merge labels of block Q and S
                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                    continue;
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_d) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                if (condition_h) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_12: Merge labels of block R and S
                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_6: Assign label of block S
                                    img_labels_row[c] = img_labels_row[c - 2];
                                    continue;
                                }
                            }
                            else {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                        }
                    }
                    else {
                        if (condition_r) {
                            if (condition_j) {
                                if (condition_m) {
                                    if (condition_h) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_11: Merge labels of block Q and S
                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_c) {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                            else {
                                                //Action_14: Merge labels of block P_, Q and S
                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_i) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_d) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            if (condition_i) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                if (condition_c) {
                                                                    //Action_6: Assign label of block S
                                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    if (condition_i) {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                                else {
                                                    //Action_16: labels of block Q, R and S
                                                    img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_h) {
                                                    if (condition_d) {
                                                        if (condition_c) {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_15: Merge labels of block P_, R and S
                                                            img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_15: Merge labels of block P_, R and S
                                                        img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_m) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_9 Merge labels of block P_ and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_m) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_9 Merge labels of block P_ and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_j) {
                                if (condition_i) {
                                    //Action_4: Assign label of block Q
                                    img_labels_row[c] = img_labels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_c) {
                                            //Action_4: Assign label of block Q
                                            img_labels_row[c] = img_labels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            //Action_7: Merge labels of block P_ and Q
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_i) {
                                            if (condition_d) {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_10 Merge labels of block Q and R
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_c) {
                                                        //Action_5: Assign label of block R
                                                        img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_8: Merge labels of block P_ and R
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_8: Merge labels of block P_ and R
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            //Action_4: Assign label of block Q
                                            img_labels_row[c] = img_labels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            if (condition_h) {
                                                //Action_3: Assign label of block P_
                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                img_labels_row[c] = lunique;
                                                P_[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            //Action_3: Assign label of block P_
                                            img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                            img_labels_row[c] = lunique;
                                            P_[lunique] = lunique;
                                            lunique = lunique + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else {
                    if (condition_s) {
                        if (condition_p) {
                            if (condition_n) {
                                if (condition_j) {
                                    if (condition_i) {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_c) {
                                            if (condition_h) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_d) {
                                            if (condition_i) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_c) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_12: Merge labels of block R and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_r) {
                                    if (condition_j) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_d) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                if (condition_i) {
                                                                    //Action_6: Assign label of block S
                                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    if (condition_c) {
                                                                        //Action_6: Assign label of block S
                                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                                        continue;
                                                                    }
                                                                    else {
                                                                        //Action_12: Merge labels of block R and S
                                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_i) {
                                                    if (condition_m) {
                                                        if (condition_h) {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_g) {
                                                                if (condition_b) {
                                                                    //Action_12: Merge labels of block R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_16: labels of block Q, R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_16: labels of block Q, R and S
                                                        img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_j) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_5: Assign label of block R
                                                    img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                    continue;
                                                }
                                                else {
                                                    // ACTION_10 Merge labels of block Q and R
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                //Action_4: Assign label of block Q
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                img_labels_row[c] = lunique;
                                                P_[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_r) {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_n) {
                                    //Action_6: Assign label of block S
                                    img_labels_row[c] = img_labels_row[c - 2];
                                    continue;
                                }
                                else {
                                    //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                    img_labels_row[c] = lunique;
                                    P_[lunique] = lunique;
                                    lunique = lunique + 1;
                                    continue;
                                }
                            }
                        }
                    }
                    else {
                        if (condition_p) {
                            if (condition_j) {
                                //Action_4: Assign label of block Q
                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                continue;
                            }
                            else {
                                if (condition_k) {
                                    if (condition_i) {
                                        if (condition_d) {
                                            //Action_5: Assign label of block R
                                            img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_10 Merge labels of block Q and R
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_5: Assign label of block R
                                        img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                        continue;
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                        img_labels_row[c] = lunique;
                                        P_[lunique] = lunique;
                                        lunique = lunique + 1;
                                        continue;
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_t) {
                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                img_labels_row[c] = lunique;
                                P_[lunique] = lunique;
                                lunique = lunique + 1;
                                continue;
                            }
                            else {
                                // Action_1: No action (the block has no foreground pixels)
                                img_labels_row[c] = 0;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        //second scan
        unsigned n_labels = ls_.Flatten(P_, lunique);

        if (img_labels_.rows & 1) {
            if (img_labels_.cols & 1) {
                //Case 1: both rows and cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
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
                                        imgLabels_row_fol[c] = iLabel;
                                    else
                                        imgLabels_row_fol[c] = 0;
                                    if (img_row_fol[c + 1] > 0)
                                        imgLabels_row_fol[c + 1] = iLabel;
                                    else
                                        imgLabels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    imgLabels_row_fol[c] = iLabel;
                                else
                                    imgLabels_row_fol[c] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    imgLabels_row_fol[c] = 0;
                                    imgLabels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                imgLabels_row_fol[c] = 0;
                            }
                        }
                    }
                }
            }//END Case 1
            else {
                //Case 2: only rows odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
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
                                    imgLabels_row_fol[c] = iLabel;
                                else
                                    imgLabels_row_fol[c] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    imgLabels_row_fol[c + 1] = iLabel;
                                else
                                    imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            if (r + 1 < img_labels_.rows) {
                                imgLabels_row_fol[c] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END Case 2
        }
        else {
            if (img_labels_.cols & 1) {
                //Case 3: only cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (img_row_fol[c] > 0)
                                imgLabels_row_fol[c] = iLabel;
                            else
                                imgLabels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    imgLabels_row_fol[c + 1] = iLabel;
                                else
                                    imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            imgLabels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END case 3
            else {
                //Case 4: nothing odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (img_row[c + 1] > 0)
                                img_labels_row[c + 1] = iLabel;
                            else
                                img_labels_row[c + 1] = 0;
                            if (img_row_fol[c] > 0)
                                imgLabels_row_fol[c] = iLabel;
                            else
                                imgLabels_row_fol[c] = 0;
                            if (img_row_fol[c + 1] > 0)
                                imgLabels_row_fol[c + 1] = iLabel;
                            else
                                imgLabels_row_fol[c + 1] = 0;
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            imgLabels_row_fol[c] = 0;
                            imgLabels_row_fol[c + 1] = 0;
                        }
                    }
                }
            }//END case 4
        }

        cv::fastFree(P_);
        return n_labels;

#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e

#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k

#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p

#undef condition_r
#undef condition_s
#undef condition_t
    }

    unsigned PerformLabelingWithSteps() override
    {
        //perf_.start("Allocate");
        AllocateMemory();
        //perf_.stop("Allocate");

        //perf_.start("FirstScan");
        unsigned lunique = FirstScan();
        //perf_.stop("FirstScan");

        //perf_.start("SecondScan");
        unsigned n_labels = SecondScan(lunique);
        //perf_.stop("SecondScan");

        //perf_.start("Deallocate");
        DeallocateMemory();
        //perf_.stop("Deallocate");

        return n_labels;
    }

    unsigned PerformLabelingMem(std::vector<unsigned long int>& accesses) override
    {
        const int h = img_.rows;
        const int w = img_.cols;

        const size_t P_length = UPPER_BOUND_8_CONNECTIVITY;

        //Data structure for memory test
        MemMat<uchar> img(img_);
        MemMat<int> img_labels(img_.size(), 0);
        MemVector<unsigned> P(P_length);						 // Vector P for equivalences resolution

        P[0] = 0;	//first label is for background pixels
        unsigned lunique = 1;

        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {
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

#define condition_b c-1>=0 && r-2>=0 && img(r-2, c-1)>0
#define condition_c r-2>=0 && img(r-2, c)>0
#define condition_d c+1<w && r-2>=0 && img(r-2, c+1)>0
#define condition_e c+2<w && r-2>=0 && img(r-2, c+2)>0

#define condition_g c-2>=0 && r-1>=0 && img(r-1, c-2)>0
#define condition_h c-1>=0 && r-1>=0 && img(r-1, c-1)>0
#define condition_i r-1>=0 && img(r-1, c)>0
#define condition_j c+1<w && r-1>=0 && img(r-1, c+1)>0
#define condition_k c+2<w && r-1>=0 && img(r-1, c+2)>0

#define condition_m c-2>=0 && img(r, c-2)>0
#define condition_n c-1>=0 && img(r, c-1)>0
#define condition_o img(r,c)>0
#define condition_p c+1<w && img(r,c+1)>0

#define condition_r c-1>=0 && r+1<h && img(r+1, c-1)>0
#define condition_s r+1<h && img(r+1, c)>0
#define condition_t c+1<w && r+1<h && img(r+1, c+1)>0

                // This is a decision tree which allows to choose which action to
                // perform, checking as few conditions as possible.
                // Actions are available after the tree.

                if (condition_o) {
                    if (condition_n) {
                        if (condition_j) {
                            if (condition_i) {
                                goto action_6;
                            }
                            else {
                                if (condition_c) {
                                    if (condition_h) {
                                        goto action_6;
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                goto action_6;
                                            }
                                            else {
                                                goto action_11;
                                            }
                                        }
                                        else {
                                            goto action_11;
                                        }
                                    }
                                }
                                else {
                                    goto action_11;
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_d) {
                                        if (condition_i) {
                                            goto action_6;
                                        }
                                        else {
                                            if (condition_c) {
                                                if (condition_h) {
                                                    goto action_6;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            goto action_6;
                                                        }
                                                        else {
                                                            goto action_12;
                                                        }
                                                    }
                                                    else {
                                                        goto action_12;
                                                    }
                                                }
                                            }
                                            else {
                                                goto action_12;
                                            }
                                        }
                                    }
                                    else {
                                        goto action_12;
                                    }
                                }
                                else {
                                    goto action_6;
                                }
                            }
                            else {
                                goto action_6;
                            }
                        }
                    }
                    else {
                        if (condition_r) {
                            if (condition_j) {
                                if (condition_m) {
                                    if (condition_h) {
                                        if (condition_i) {
                                            goto action_6;
                                        }
                                        else {
                                            if (condition_c) {
                                                goto action_6;
                                            }
                                            else {
                                                goto action_11;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                if (condition_i) {
                                                    goto action_6;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                            }
                                            else {
                                                goto action_11;
                                            }
                                        }
                                        else {
                                            goto action_11;
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        goto action_11;
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_c) {
                                                goto action_11;
                                            }
                                            else {
                                                goto action_14;
                                            }
                                        }
                                        else {
                                            goto action_11;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_i) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            goto action_6;
                                                        }
                                                        else {
                                                            goto action_12;
                                                        }
                                                    }
                                                }
                                                else {
                                                    goto action_12;
                                                }
                                            }
                                            else {
                                                if (condition_d) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            if (condition_i) {
                                                                goto action_6;
                                                            }
                                                            else {
                                                                if (condition_c) {
                                                                    goto action_6;
                                                                }
                                                                else {
                                                                    goto action_12;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            goto action_12;
                                                        }
                                                    }
                                                    else {
                                                        goto action_12;
                                                    }
                                                }
                                                else {
                                                    if (condition_i) {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                goto action_12;
                                                            }
                                                            else {
                                                                goto action_16;
                                                            }
                                                        }
                                                        else {
                                                            goto action_16;
                                                        }
                                                    }
                                                    else {
                                                        goto action_12;
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    goto action_12;
                                                }
                                                else {
                                                    goto action_16;
                                                }
                                            }
                                            else {
                                                if (condition_h) {
                                                    if (condition_d) {
                                                        if (condition_c) {
                                                            goto action_12;
                                                        }
                                                        else {
                                                            goto action_15;
                                                        }
                                                    }
                                                    else {
                                                        goto action_15;
                                                    }
                                                }
                                                else {
                                                    goto action_12;
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_m) {
                                                goto action_6;
                                            }
                                            else {
                                                goto action_9;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            goto action_6;
                                                        }
                                                        else {
                                                            goto action_11;
                                                        }
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                                else {
                                                    goto action_11;
                                                }
                                            }
                                            else {
                                                goto action_6;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_m) {
                                            goto action_6;
                                        }
                                        else {
                                            goto action_9;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                                else {
                                                    goto action_11;
                                                }
                                            }
                                            else {
                                                goto action_11;
                                            }
                                        }
                                        else {
                                            goto action_6;
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_j) {
                                if (condition_i) {
                                    goto action_4;
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_c) {
                                            goto action_4;
                                        }
                                        else {
                                            goto action_7;
                                        }
                                    }
                                    else {
                                        goto action_4;
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_i) {
                                            if (condition_d) {
                                                goto action_5;
                                            }
                                            else {
                                                goto action_10;
                                            }
                                        }
                                        else {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_c) {
                                                        goto action_5;
                                                    }
                                                    else {
                                                        goto action_8;
                                                    }
                                                }
                                                else {
                                                    goto action_8;
                                                }
                                            }
                                            else {
                                                goto action_5;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            goto action_4;
                                        }
                                        else {
                                            if (condition_h) {
                                                goto action_3;
                                            }
                                            else {
                                                goto action_2;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        goto action_4;
                                    }
                                    else {
                                        if (condition_h) {
                                            goto action_3;
                                        }
                                        else {
                                            goto action_2;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else {
                    if (condition_s) {
                        if (condition_p) {
                            if (condition_n) {
                                if (condition_j) {
                                    if (condition_i) {
                                        goto action_6;
                                    }
                                    else {
                                        if (condition_c) {
                                            if (condition_h) {
                                                goto action_6;
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                                else {
                                                    goto action_11;
                                                }
                                            }
                                        }
                                        else {
                                            goto action_11;
                                        }
                                    }
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_d) {
                                            if (condition_i) {
                                                goto action_6;
                                            }
                                            else {
                                                if (condition_c) {
                                                    if (condition_h) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                goto action_6;
                                                            }
                                                            else {
                                                                goto action_12;
                                                            }
                                                        }
                                                        else {
                                                            goto action_12;
                                                        }
                                                    }
                                                }
                                                else {
                                                    goto action_12;
                                                }
                                            }
                                        }
                                        else {
                                            goto action_12;
                                        }
                                    }
                                    else {
                                        goto action_6;
                                    }
                                }
                            }
                            else {
                                if (condition_r) {
                                    if (condition_j) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_i) {
                                                    goto action_6;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        if (condition_i) {
                                                            goto action_6;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                goto action_6;
                                                            }
                                                            else {
                                                                goto action_11;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        goto action_11;
                                                    }
                                                }
                                                else {
                                                    goto action_11;
                                                }
                                            }
                                        }
                                        else {
                                            goto action_11;
                                        }
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_d) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        if (condition_i) {
                                                            goto action_6;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                goto action_6;
                                                            }
                                                            else {
                                                                goto action_12;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                if (condition_i) {
                                                                    goto action_6;
                                                                }
                                                                else {
                                                                    if (condition_c) {
                                                                        goto action_6;
                                                                    }
                                                                    else {
                                                                        goto action_12;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                goto action_12;
                                                            }
                                                        }
                                                        else {
                                                            goto action_12;
                                                        }
                                                    }
                                                }
                                                else {
                                                    goto action_12;
                                                }
                                            }
                                            else {
                                                if (condition_i) {
                                                    if (condition_m) {
                                                        if (condition_h) {
                                                            goto action_12;
                                                        }
                                                        else {
                                                            if (condition_g) {
                                                                if (condition_b) {
                                                                    goto action_12;
                                                                }
                                                                else {
                                                                    goto action_16;
                                                                }
                                                            }
                                                            else {
                                                                goto action_16;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        goto action_16;
                                                    }
                                                }
                                                else {
                                                    goto action_12;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        goto action_6;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                goto action_6;
                                                            }
                                                            else {
                                                                goto action_11;
                                                            }
                                                        }
                                                        else {
                                                            goto action_11;
                                                        }
                                                    }
                                                }
                                                else {
                                                    goto action_11;
                                                }
                                            }
                                            else {
                                                goto action_6;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_j) {
                                        goto action_4;
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    goto action_5;
                                                }
                                                else {
                                                    goto action_10;
                                                }
                                            }
                                            else {
                                                goto action_5;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                goto action_4;
                                            }
                                            else {
                                                goto action_2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_r) {
                                goto action_6;
                            }
                            else {
                                if (condition_n) {
                                    goto action_6;
                                }
                                else {
                                    goto action_2;
                                }
                            }
                        }
                    }
                    else {
                        if (condition_p) {
                            if (condition_j) {
                                goto action_4;
                            }
                            else {
                                if (condition_k) {
                                    if (condition_i) {
                                        if (condition_d) {
                                            goto action_5;
                                        }
                                        else {
                                            goto action_10;
                                        }
                                    }
                                    else {
                                        goto action_5;
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        goto action_4;
                                    }
                                    else {
                                        goto action_2;
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_t) {
                                goto action_2;
                            }
                            else {
                                goto action_1;
                            }
                        }
                    }
                }

                // Actions: the blocks label are provisionally stored in the top left
                // pixel of the block in the labels image

            action_1:	//Action_1: No action (the block has no foreground pixels)
                img_labels(r, c) = 0;
                continue;
            action_2:	//Action_2: New label (the block has foreground pixels and is not connected to anything else)
                img_labels(r, c) = lunique;
                P[lunique] = lunique;
                lunique = lunique + 1;
                continue;
            action_3:	//Action_3: Assign label of block P
                img_labels(r, c) = img_labels(r - 2, c - 2);
                continue;
            action_4:	//Action_4: Assign label of block Q
                img_labels(r, c) = img_labels(r - 2, c);
                continue;
            action_5:	//Action_5: Assign label of block R
                img_labels(r, c) = img_labels(r - 2, c + 2);
                continue;
            action_6:	//Action_6: Assign label of block S
                img_labels(r, c) = img_labels(r, c - 2);
                continue;
            action_7:	//Action_7: Merge labels of block P and Q
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c - 2), (unsigned)img_labels(r - 2, c));
                continue;
            action_8:	//Action_8: Merge labels of block P and R
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c - 2), (unsigned)img_labels(r - 2, c + 2));
                continue;
            action_9:	// Action_9 Merge labels of block P and S
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c - 2), (unsigned)img_labels(r, c - 2));
                continue;
            action_10:	// Action_10 Merge labels of block Q and R
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c), (unsigned)img_labels(r - 2, c + 2));
                continue;
            action_11:	//Action_11: Merge labels of block Q and S
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c), (unsigned)img_labels(r, c - 2));
                continue;
            action_12:	//Action_12: Merge labels of block R and S
                img_labels(r, c) = ls_.Merge(P, (unsigned)img_labels(r - 2, c + 2), (unsigned)img_labels(r, c - 2));
                continue;
            action_14:	//Action_14: Merge labels of block P, Q and S
                img_labels(r, c) = ls_.Merge(P, ls_.Merge(P, (unsigned)img_labels(r - 2, c - 2), (unsigned)img_labels(r - 2, c)), (unsigned)img_labels(r, c - 2));
                continue;
            action_15:	//Action_15: Merge labels of block P, R and S
                img_labels(r, c) = ls_.Merge(P, ls_.Merge(P, (unsigned)img_labels(r - 2, c - 2), (unsigned)img_labels(r - 2, c + 2)), (unsigned)img_labels(r, c - 2));
                continue;
            action_16:	//Action_16: labels of block Q, R and S
                img_labels(r, c) = ls_.Merge(P, ls_.Merge(P, (unsigned)img_labels(r - 2, c), (unsigned)img_labels(r - 2, c + 2)), (unsigned)img_labels(r, c - 2));
                continue;
            }
        }

        unsigned n_labels = ls_.Flatten(P, lunique);

        //Second scan
        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {
                int iLabel = img_labels(r, c);
                if (iLabel > 0) {
                    iLabel = P[iLabel];
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

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAcesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAcesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)P.GetTotalAcesses();

        //a = img_labels.GetImage();
#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e

#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k

#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p

#undef condition_r
#undef condition_s
#undef condition_t

        return n_labels;
    }

private:
    LabelsSolver ls_;
    unsigned *P_;

    void AllocateMemory() override
    {
        const size_t P_length = UPPER_BOUND_8_CONNECTIVITY;
        //array P_ for equivalences resolution
        P_ = (unsigned *)cv::fastMalloc(sizeof(unsigned) * P_length);
    }
    void DeallocateMemory() override
    {
        cv::fastFree(P_);
    }
    unsigned FirstScan() override
    {
        int w(img_.cols), h(img_.rows);

        img_labels_ = cv::Mat1i(img_.size());

        //Background
        P_[0] = 0;
        unsigned lunique = 1;

        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            const uchar* const img_row_prev_prev = (uchar *)(((char *)img_row_prev) - img_.step.p[0]);
            const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
            for (int c = 0; c < w; c += 2) {
                // We work with 2x2 blocks
                // +-+-+-+
                // |P_|Q|R|
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

#define condition_b c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define condition_c r-2>=0 && img_row_prev_prev[c]>0
#define condition_d c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define condition_e c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define condition_g c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define condition_h c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define condition_i r-1>=0 && img_row_prev[c]>0
#define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define condition_m c-2>=0 && img_row[c-2]>0
#define condition_n c-1>=0 && img_row[c-1]>0
#define condition_o img_row[c]>0
#define condition_p c+1<w && img_row[c+1]>0

#define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define condition_s r+1<h && img_row_fol[c]>0
#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0

                // This is a decision tree which allows to choose which action to
                // perform, checking as few conditions as possible.
                // Actions are available after the tree.

                if (condition_o) {
                    if (condition_n) {
                        if (condition_j) {
                            if (condition_i) {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_c) {
                                    if (condition_h) {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    //Action_11: Merge labels of block Q and S
                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                    continue;
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_d) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                if (condition_h) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_12: Merge labels of block R and S
                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_6: Assign label of block S
                                    img_labels_row[c] = img_labels_row[c - 2];
                                    continue;
                                }
                            }
                            else {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                        }
                    }
                    else {
                        if (condition_r) {
                            if (condition_j) {
                                if (condition_m) {
                                    if (condition_h) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_g) {
                                            if (condition_b) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_11: Merge labels of block Q and S
                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_c) {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                            else {
                                                //Action_14: Merge labels of block P_, Q and S
                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_i) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_d) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            if (condition_i) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                if (condition_c) {
                                                                    //Action_6: Assign label of block S
                                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    if (condition_i) {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                                else {
                                                    //Action_16: labels of block Q, R and S
                                                    img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_h) {
                                                    if (condition_d) {
                                                        if (condition_c) {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_15: Merge labels of block P_, R and S
                                                            img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_15: Merge labels of block P_, R and S
                                                        img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_m) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_9 Merge labels of block P_ and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_m) {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_9 Merge labels of block P_ and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_6: Assign label of block S
                                            img_labels_row[c] = img_labels_row[c - 2];
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_j) {
                                if (condition_i) {
                                    //Action_4: Assign label of block Q
                                    img_labels_row[c] = img_labels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_c) {
                                            //Action_4: Assign label of block Q
                                            img_labels_row[c] = img_labels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            //Action_7: Merge labels of block P_ and Q
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_p) {
                                    if (condition_k) {
                                        if (condition_i) {
                                            if (condition_d) {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_10 Merge labels of block Q and R
                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_c) {
                                                        //Action_5: Assign label of block R
                                                        img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_8: Merge labels of block P_ and R
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_8: Merge labels of block P_ and R
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            //Action_4: Assign label of block Q
                                            img_labels_row[c] = img_labels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            if (condition_h) {
                                                //Action_3: Assign label of block P_
                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                img_labels_row[c] = lunique;
                                                P_[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            //Action_3: Assign label of block P_
                                            img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                            img_labels_row[c] = lunique;
                                            P_[lunique] = lunique;
                                            lunique = lunique + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else {
                    if (condition_s) {
                        if (condition_p) {
                            if (condition_n) {
                                if (condition_j) {
                                    if (condition_i) {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_c) {
                                            if (condition_h) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_d) {
                                            if (condition_i) {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_c) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_12: Merge labels of block R and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_6: Assign label of block S
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_r) {
                                    if (condition_j) {
                                        if (condition_m) {
                                            if (condition_h) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_d) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            img_labels_row[c] = img_labels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                if (condition_i) {
                                                                    //Action_6: Assign label of block S
                                                                    img_labels_row[c] = img_labels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    if (condition_c) {
                                                                        //Action_6: Assign label of block S
                                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                                        continue;
                                                                    }
                                                                    else {
                                                                        //Action_12: Merge labels of block R and S
                                                                        img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_i) {
                                                    if (condition_m) {
                                                        if (condition_h) {
                                                            //Action_12: Merge labels of block R and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_g) {
                                                                if (condition_b) {
                                                                    //Action_12: Merge labels of block R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_16: labels of block Q, R and S
                                                                    img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_16: labels of block Q, R and S
                                                        img_labels_row[c] = ls_.Merge(P_, ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        //Action_6: Assign label of block S
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_6: Assign label of block S
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_11: Merge labels of block Q and S
                                                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_6: Assign label of block S
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_j) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_k) {
                                            if (condition_i) {
                                                if (condition_d) {
                                                    //Action_5: Assign label of block R
                                                    img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                    continue;
                                                }
                                                else {
                                                    // ACTION_10 Merge labels of block Q and R
                                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_5: Assign label of block R
                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                //Action_4: Assign label of block Q
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                continue;
                                            }
                                            else {
                                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                                img_labels_row[c] = lunique;
                                                P_[lunique] = lunique;
                                                lunique = lunique + 1;
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_r) {
                                //Action_6: Assign label of block S
                                img_labels_row[c] = img_labels_row[c - 2];
                                continue;
                            }
                            else {
                                if (condition_n) {
                                    //Action_6: Assign label of block S
                                    img_labels_row[c] = img_labels_row[c - 2];
                                    continue;
                                }
                                else {
                                    //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                    img_labels_row[c] = lunique;
                                    P_[lunique] = lunique;
                                    lunique = lunique + 1;
                                    continue;
                                }
                            }
                        }
                    }
                    else {
                        if (condition_p) {
                            if (condition_j) {
                                //Action_4: Assign label of block Q
                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                continue;
                            }
                            else {
                                if (condition_k) {
                                    if (condition_i) {
                                        if (condition_d) {
                                            //Action_5: Assign label of block R
                                            img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_10 Merge labels of block Q and R
                                            img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_5: Assign label of block R
                                        img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                        continue;
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                        img_labels_row[c] = lunique;
                                        P_[lunique] = lunique;
                                        lunique = lunique + 1;
                                        continue;
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_t) {
                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                img_labels_row[c] = lunique;
                                P_[lunique] = lunique;
                                lunique = lunique + 1;
                                continue;
                            }
                            else {
                                // Action_1: No action (the block has no foreground pixels)
                                img_labels_row[c] = 0;
                                continue;
                            }
                        }
                    }
                }
            }
        }

#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e

#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k

#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p

#undef condition_r
#undef condition_s
#undef condition_t

        return lunique;
    }
    unsigned SecondScan(const unsigned& lunique) override

    {
        unsigned n_labels = ls_.Flatten(P_, lunique);

        // Second scan
        if (img_labels_.rows & 1) {
            if (img_labels_.cols & 1) {
                //Case 1: both rows and cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
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
                                        imgLabels_row_fol[c] = iLabel;
                                    else
                                        imgLabels_row_fol[c] = 0;
                                    if (img_row_fol[c + 1] > 0)
                                        imgLabels_row_fol[c + 1] = iLabel;
                                    else
                                        imgLabels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                if (img_row_fol[c] > 0)
                                    imgLabels_row_fol[c] = iLabel;
                                else
                                    imgLabels_row_fol[c] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                if (r + 1 < img_labels_.rows) {
                                    imgLabels_row_fol[c] = 0;
                                    imgLabels_row_fol[c + 1] = 0;
                                }
                            }
                            else if (r + 1 < img_labels_.rows) {
                                imgLabels_row_fol[c] = 0;
                            }
                        }
                    }
                }
            }//END Case 1
            else {
                //Case 2: only rows odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
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
                                    imgLabels_row_fol[c] = iLabel;
                                else
                                    imgLabels_row_fol[c] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    imgLabels_row_fol[c + 1] = iLabel;
                                else
                                    imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            if (r + 1 < img_labels_.rows) {
                                imgLabels_row_fol[c] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END Case 2
        }
        else {
            if (img_labels_.cols & 1) {
                //Case 3: only cols odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (img_row_fol[c] > 0)
                                imgLabels_row_fol[c] = iLabel;
                            else
                                imgLabels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                if (img_row[c + 1] > 0)
                                    img_labels_row[c + 1] = iLabel;
                                else
                                    img_labels_row[c + 1] = 0;
                                if (img_row_fol[c + 1] > 0)
                                    imgLabels_row_fol[c + 1] = iLabel;
                                else
                                    imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                        else {
                            img_labels_row[c] = 0;
                            imgLabels_row_fol[c] = 0;
                            if (c + 1 < img_labels_.cols) {
                                img_labels_row[c + 1] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                            }
                        }
                    }
                }
            }// END case 3
            else {
                //Case 4: nothing odd
                for (int r = 0; r < img_labels_.rows; r += 2) {
                    // Get rows pointer
                    const uchar* const img_row = img_.ptr<uchar>(r);
                    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);

                    unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
                    unsigned* const imgLabels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
                    // Get rows pointer
                    for (int c = 0; c < img_labels_.cols; c += 2) {
                        int iLabel = img_labels_row[c];
                        if (iLabel > 0) {
                            iLabel = P_[iLabel];
                            if (img_row[c] > 0)
                                img_labels_row[c] = iLabel;
                            else
                                img_labels_row[c] = 0;
                            if (img_row[c + 1] > 0)
                                img_labels_row[c + 1] = iLabel;
                            else
                                img_labels_row[c + 1] = 0;
                            if (img_row_fol[c] > 0)
                                imgLabels_row_fol[c] = iLabel;
                            else
                                imgLabels_row_fol[c] = 0;
                            if (img_row_fol[c + 1] > 0)
                                imgLabels_row_fol[c + 1] = iLabel;
                            else
                                imgLabels_row_fol[c + 1] = 0;
                        }
                        else {
                            img_labels_row[c] = 0;
                            img_labels_row[c + 1] = 0;
                            imgLabels_row_fol[c] = 0;
                            imgLabels_row_fol[c + 1] = 0;
                        }
                    }
                }
            }//END case 4
        }

        return n_labels;
    }
};

#endif // !YACCLAB_LABELING_GRANA_2010_H_