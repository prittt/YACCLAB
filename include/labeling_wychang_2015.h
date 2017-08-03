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
class CCIT : public Labeling {
public:
    CCIT() {}

    void PerformLabeling()
    {
        
        img_labels_ = cv::Mat1i(img_.size(), 0);

        int w = img_labels_.cols;
        int h = img_labels_.rows;

        int m = 1;
        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        int lx, u, v, k;

#define CONDITION_B1 img_row[x]==byF
#define CONDITION_B2 x+1<w && img_row[x+1]==byF              // WRONG in the original code -> add missing condition 
#define CONDITION_B3 y+1<h && img_row_fol[x]==byF            // WRONG in the original code -> add missing condition
#define CONDITION_B4 x+1<w && y+1<h && img_row_fol[x+1]==byF // WRONG in the original code -> add missing condition
#define CONDITION_U1 x-1>0 && img_row_prev[x-1]==byF         // WRONG in the original code -> add missing condition
#define CONDITION_U2 img_row_prev[x]==byF
#define CONDITION_U3 x+1<w && img_row_prev[x+1]==byF         // WRONG in the original code -> add missing condition
#define CONDITION_U4 x+2<w && img_row_prev[x+2]==byF         // WRONG in the original code -> add missing condition
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

            if (CONDITION_B1) {
                NEW_LABEL;
                if (CONDITION_B2 || CONDITION_B4)
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (CONDITION_B2) {
                NEW_LABEL;
                nextprocedure2 = true;
            }
            else if (CONDITION_B3) {
                NEW_LABEL;
                if (CONDITION_B4)
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (CONDITION_B4) {
                NEW_LABEL;
                nextprocedure2 = true;
            }
            else {
                nextprocedure2 = false;
            }

            while (nextprocedure2 && x + 2 < w) {
                x = x + 2;

                if (CONDITION_B1) {
                    ASSIGN_LX;
                    if (CONDITION_B2 || CONDITION_B4)
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;
                }
                else if (CONDITION_B2) {

                    if (CONDITION_B3) {
                        ASSIGN_LX;
                    }
                    else {
                        NEW_LABEL;
                    }
                    nextprocedure2 = true;
                }
                else if (CONDITION_B3) {
                    ASSIGN_LX;
                    if (CONDITION_B4)
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;
                }
                else if (CONDITION_B4) {
                    NEW_LABEL;
                    nextprocedure2 = true;
                }
                else {
                    nextprocedure2 = false;
                }

            }
        }

        for (int y = 2; y < h; y += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
            unsigned int* const img_labels_row_prev_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
            for (int x = 0; x < w; x += 2) {
                if (CONDITION_B1) {
                    if (CONDITION_B2) {
                        if (CONDITION_U2) {
                            lx = ASSIGN_Q;
                            if (CONDITION_U3) {

                            }
                            else {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_RV;
                                    RESOLVE_2(u, v);
                                }
                            }
                        }
                        else if (CONDITION_U3) {
                            lx = ASSIGN_Q;
                            if (CONDITION_U1) {
                                LOAD_LX;
                                LOAD_PV;
                                RESOLVE_2(u, v);
                            }

                        }
                        else if (CONDITION_U1) {
                            lx = ASSIGN_P;
                            if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                        }
                        else if (CONDITION_U4) {
                            lx = ASSIGN_R;
                        }
                        else {
                            NEW_LABEL;
                        }
                        nextprocedure2 = true;
                    }
                    else {
                        if (CONDITION_U2) {
                            lx = ASSIGN_Q;
                        }
                        else if (CONDITION_U1) {
                            lx = ASSIGN_P;
                            if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);

                            }
                        }
                        else if (CONDITION_U3) {
                            lx = ASSIGN_Q;
                        }
                        else {
                            NEW_LABEL;
                        }
                        if (CONDITION_B4)
                            nextprocedure2 = true;
                        else
                            nextprocedure2 = false;

                    }
                }
                else if (CONDITION_B2) {
                    if (CONDITION_U3) {
                        lx = ASSIGN_Q;
                    }
                    else if (CONDITION_U2) {
                        lx = ASSIGN_Q;
                        if (CONDITION_U4) {
                            LOAD_LX;
                            LOAD_RV;
                            RESOLVE_2(u, v);
                        }
                    }
                    else if (CONDITION_U4) {
                        lx = ASSIGN_R;
                    }
                    else {
                        NEW_LABEL;
                    }
                    nextprocedure2 = true;
                }
                else if (CONDITION_B3) {
                    NEW_LABEL;
                    if (CONDITION_B4)
                        nextprocedure2 = true;//
                    else
                        nextprocedure2 = false;
                }
                else if (CONDITION_B4) {
                    NEW_LABEL;
                    nextprocedure2 = true;
                }
                else {
                    nextprocedure2 = false;
                }

                while (nextprocedure2 && x + 2 < w) {
                    x = x + 2;
                    if (CONDITION_B1) {

                        ASSIGN_LX;
                        if (CONDITION_B2) {
                            if (CONDITION_U2) {
                                if (CONDITION_U3) {
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                                else {
                                    if (CONDITION_U4) {
                                        LOAD_LX;
                                        LOAD_QV;
                                        LOAD_RK;
                                        RESOLVE_3(u, v, k);
                                    }
                                    else {
                                        LOAD_LX;
                                        LOAD_QV;
                                        RESOLVE_2(u, v);
                                    }
                                }

                            }
                            else if (CONDITION_U3) {
                                if (CONDITION_U1) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_QK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    // Reslove S, Q
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U1) {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_RK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_PV;
                                    RESOLVE_2(u, v);
                                }

                            }
                            else if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                            nextprocedure2 = true;
                        }
                        else {
                            ASSIGN_LX;
                            if (CONDITION_U2) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }
                            else if (CONDITION_U1) {
                                if (CONDITION_U3) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_QK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_PV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }

                            if (CONDITION_B4)
                                nextprocedure2 = true;//
                            else
                                nextprocedure2 = false;
                        }

                    }
                    else if (CONDITION_B2) {
                        if (CONDITION_B3) {
                            ASSIGN_LX;
                            if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }
                            else if (CONDITION_U2) {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_QV;
                                    LOAD_RK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                            }if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                        }
                        else {
                            if (CONDITION_U3) {
                                lx = ASSIGN_Q;
                            }
                            else if (CONDITION_U2) {
                                lx = ASSIGN_Q;
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_RV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U4) {
                                lx = ASSIGN_R;
                            }
                            else {
                                NEW_LABEL;
                            }
                        }

                        nextprocedure2 = true;
                    }
                    else if (CONDITION_B3) {
                        ASSIGN_LX;
                        if (CONDITION_B4)
                            nextprocedure2 = true;
                        else
                            nextprocedure2 = false;
                    }
                    else if (CONDITION_B4) {
                        NEW_LABEL;
                        nextprocedure2 = true;
                    }
                    else {
                        nextprocedure2 = false;
                    }
                }
            }
        }
        
        // Renew label number
        n_labels_ = LabelsSolver::Flatten();

        // Second scan
        for (int y = 0; y < h; y += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            for (int x = 0; x < w; x += 2) {
                int iLabel = img_labels_row[x];
                if (iLabel > 0) {
                    iLabel = LabelsSolver::GetLabel(iLabel);
                    if (img_row[x] == byF)
                        img_labels_row[x] = iLabel;
                    else
                        img_labels_row[x] = 0;
                    if (x + 1 < w) {
                        if (img_row[x + 1] == byF)
                            img_labels_row[x + 1] = iLabel;
                        else
                            img_labels_row[x + 1] = 0;
                        if (y + 1 < h) {
                            if (img_row_fol[x] == byF)
                                img_labels_row_fol[x] = iLabel;
                            else
                                img_labels_row_fol[x] = 0;
                            if (img_row_fol[x + 1] == byF)
                                img_labels_row_fol[x + 1] = iLabel;
                            else
                                img_labels_row_fol[x + 1] = 0;
                        }
                    }
                    else if (y + 1 < h) {
                        if (img_row_fol[x] == byF)
                            img_labels_row_fol[x] = iLabel;
                        else
                            img_labels_row_fol[x] = 0;
                    }
                }
                else {
                    img_labels_row[x] = 0;
                    if (x + 1 < w) {
                        img_labels_row[x + 1] = 0;
                        if (y + 1 < h) {
                            img_labels_row_fol[x] = 0;
                            img_labels_row_fol[x + 1] = 0;
                        }
                    }
                    else if (y + 1 < h) {
                        img_labels_row_fol[x] = 0;
                    }
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

private:
    unsigned char byF = 1; // Byte of foreground (fixed for YACCLAB)
};

#endif //YACCLAB_LABELING_WYCHANG_2015_H_