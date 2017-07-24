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

#ifndef YACCLAB_LABELING_LACASSAGNE_2011_H_
#define YACCLAB_LABELING_LACASSAGNE_2011_H_

#include <numeric>

#include <opencv2/opencv.hpp>

#include "labels_solver.h"
#include "labeling_algorithms.h"
#include "memory_tester.h"

using namespace std;
using namespace cv;

template <typename LabelsSolver>
class LSL_STD : public Labeling {
public:
    void PerformLabeling() {

        int rows = img_.rows;
        int cols = img_.cols;

        // Step 1
        Mat1i ER(rows, cols);   // Matrix of relative label (1 label/pixel) 
        Mat1i RLC(rows, (cols + 1) & ~1); // MISSING in the paper: RLC requires 2 values/run in row, so width must be next multiple of 2
        int *ner = new int[rows]; //vector<int> ner(rows); // Number of runs 

        for (int r = 0; r < rows; ++r) {
            // Get pointers to rows
            const unsigned char* img_r = img_.ptr<unsigned char>(r);
            unsigned* ER_r = ER.ptr<unsigned>(r);
            unsigned* RLC_r = RLC.ptr<unsigned>(r);
            int x0;
            int x1 = 0; // Previous value of X
            int f = 0;  // Front detection
            int b = 0;  // Right border compensation
            int er = 0;
            for (int c = 0; c < cols; ++c)
            {
                x0 = img_r[c] > 0;
                f = x0 ^ x1;
                RLC_r[er] = c - b;
                b = b ^ f;
                er = er + f;
                ER_r[c] = er;
                x1 = x0;
            }
            x0 = 0;
            f = x0 ^ x1;
            RLC_r[er] = cols - b;
            er = er + f;
            ner[r] = er;
        }

        // Step 2
        Mat1i ERA(rows, cols + 1, 0); // MISSING in the paper: ERA must have one column more than the input image 
                                      // in order to handle special cases (e.g. lines with chessboard pattern 
                                      // starting with a foreground pixel) 

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        // First row
        {
            unsigned* ERA_r = ERA.ptr<unsigned>(0);
            for (int er = 1; er <= ner[0]; er += 2) {
                ERA_r[er] = LabelsSolver::NewLabel();
            }
        }
        for (int r = 1; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ERA_r_prev = (unsigned *)(((char *)ERA_r) - ERA.step.p[0]);
            const unsigned* ER_r_prev = ER.ptr<unsigned>(r - 1);
            const unsigned* RLC_r = RLC.ptr<unsigned>(r);
            for (int er = 1; er <= ner[r]; er += 2) {
                int j0 = RLC_r[er - 1];
                int j1 = RLC_r[er];
                // Check extension in case of 8-connect algorithm
                if (j0 > 0)
                    j0--;
                if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1"
                    j1++;
                int er0 = ER_r_prev[j0];
                int er1 = ER_r_prev[j1];
                // Check label parity: segments are odd
                if (er0 % 2 == 0)
                    er0++;
                if (er1 % 2 == 0)
                    er1--;
                if (er1 >= er0) {
                    int ea = ERA_r_prev[er0];
                    int a = LabelsSolver::FindRoot(ea);
                    for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2"
                        int eak = ERA_r_prev[erk];
                        int ak = LabelsSolver::FindRoot(eak);
                        // Min extraction and propagation
                        if (a < ak)
                            LabelsSolver::UpdateTable(ak, a);
                        if (a > ak)
                        {
                            LabelsSolver::UpdateTable(a, ak);
                            a = ak;
                        }
                    }
                    ERA_r[er] = a; // The global min
                }
                else
                {
                    ERA_r[er] = LabelsSolver::NewLabel();
                }
            }
        }

        // Step 3
        //Mat1i EA(rows, cols);
        //for (int r = 0; r < rows; ++r) {
        //	for (int c = 0; c < cols; ++c) {
        //		EA(r, c) = ERA(r, ER(r, c));
        //	}
        //}
        // Sorry, but we really don't get why this shouldn't be included in the last step

        // Step 4
        n_labels_ = LabelsSolver::Flatten();

        // Step 5
        img_labels_ = Mat1i(rows, cols);
        for (int r = 0; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* labels_r = img_labels_.ptr<unsigned>(r);
            const unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ER_r = ER.ptr<unsigned>(r);
            for (int c = 0; c < cols; ++c)
            {
                //labels(r, c) = A[EA(r, c)];
                labels_r[c] = LabelsSolver::GetLabel(ERA_r[ER_r[c]]); // This is Step 3 and 5 together
            }
        }

        delete[] ner;
        LabelsSolver::Dealloc();
    }
    void PerformLabelingWithSteps()
    {
        perf_.start();
        Alloc();
        perf_.stop();
        double alloc_timing = perf_.last();

        perf_.start();
        AllScans();
        perf_.stop();
        perf_.store(Step(StepType::ALL_SCANS), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
    }

    void PerformLabelingMem(std::vector<unsigned long int>& accesses) {
        int rows = img_.rows;
        int cols = img_.cols;

        MemMat<int> img(img_);

        // Step 1
        MemMat<int> ER(rows, cols);   // Matrix of relative label (1 label/pixel) 
        MemMat<int> RLC(rows, (cols + 1) & ~1); // MISSING in the paper: RLC requires 2 values/run in row, so width must be next multiple of 2
        MemVector<int> ner(rows); //vector<int> ner(rows); // Number of runs 

        for (int r = 0; r < rows; ++r) {
            int x0;
            int x1 = 0; // Previous value of X
            int f = 0;  // Front detection
            int b = 0;  // Right border compensation
            int er = 0;
            for (int c = 0; c < cols; ++c)
            {
                x0 = img(r,c) > 0;
                f = x0 ^ x1;
                RLC(r,er) = c - b;
                b = b ^ f;
                er = er + f;
                ER(r,c) = er;
                x1 = x0;
            }
            x0 = 0;
            f = x0 ^ x1;
            RLC(r,er) = cols - b;
            er = er + f;
            ner[r] = er;
        }

        // Step 2
        MemMat<int> ERA(rows, cols + 1, 0); // MISSING in the paper: ERA must have one column more than the input image 
                                            // in order to handle special cases (e.g. lines with chessboard pattern 
                                            // starting with a foreground pixel) 

        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::MemSetup();

        // First row
        {
            for (int er = 1; er <= ner[0]; er += 2) {
                ERA(0,er) = LabelsSolver::MemNewLabel();
            }
        }
        for (int r = 1; r < rows; ++r)
        {
            for (int er = 1; er <= ner[r]; er += 2) {
                int j0 = RLC(r,er - 1);
                int j1 = RLC(r,er);
                // Check extension in case of 8-connect algorithm
                if (j0 > 0)
                    j0--;
                if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1"
                    j1++;
                int er0 = ER(r-1, j0);
                int er1 = ER(r-1, j1);
                // Check label parity: segments are odd
                if (er0 % 2 == 0)
                    er0++;
                if (er1 % 2 == 0)
                    er1--;
                if (er1 >= er0) {
                    int ea = ERA(r-1,er0);
                    int a = LabelsSolver::MemFindRoot(ea);
                    for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2"
                        int eak = ERA(r-1,erk);
                        int ak = LabelsSolver::MemFindRoot(eak);
                        // Min extraction and propagation
                        if (a < ak)
                            LabelsSolver::MemUpdateTable(ak, a);
                        if (a > ak)
                        {
                            LabelsSolver::MemUpdateTable(a, ak);
                            a = ak;
                        }
                    }
                    ERA(r,er) = a; // The global min
                }
                else
                {
                    ERA(r,er) = LabelsSolver::MemNewLabel();
                }
            }
        }

        // Step 3
        //Mat1i EA(rows, cols);
        //for (int r = 0; r < rows; ++r) {
        //	for (int c = 0; c < cols; ++c) {
        //		EA(r, c) = ERA(r, ER(r, c));
        //	}
        //}
        // Sorry, but we really don't get why this shouldn't be included in the last step

        // Step 4
        n_labels_ = LabelsSolver::MemFlatten();

        // Step 5
        MemMat<int> labels(rows, cols);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                //labels(r, c) = A[EA(r, c)];
                labels(r,c) = LabelsSolver::MemGetLabel(ERA(r, ER(r, c))); // This is Step 3 and 5 together
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)LabelsSolver::MemTotalAccesses();
        accesses[MD_OTHER] = (unsigned long int)(ER.GetTotalAccesses() + RLC.GetTotalAccesses() + ner.GetTotalAccesses() + ERA.GetTotalAccesses());

        img_labels_ = labels.GetImage();

        LabelsSolver::MemDealloc();
    }

private:
    int *ner;
    Mat1i ER, RLC, ERA;

    void Alloc() {
        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        img_labels_ = cv::Mat1i(img_.size()); // Memory allocation of the output image

        int rows = img_.rows;
        int cols = img_.cols;

        ER = Mat1i(rows, cols); // Matrix of relative label (1 label/pixel) 
        RLC = Mat1i(rows, (cols + 1) & ~1); // MISSING in the paper: RLC requires 2 values/run in row, so width must be next multiple of 2
        ner = new int[rows]; //vector<int> ner(rows); // Number of runs 

        ERA = Mat1i(rows, cols + 1, 0); // MISSING in the paper: ERA must have one column more than the input image 
                                // in order to handle special cases (e.g. lines with chessboard pattern 
                                // starting with a foreground pixel) 
    }
    void Dealloc() {
        ERA.release();

        delete[] ner;
        RLC.release();
        ER.release();

        // No free for img_labels_ because it is required at the end of the algorithm 
        LabelsSolver::Dealloc();
    }
    void AllScans()
    {
        int rows = img_.rows;
        int cols = img_.cols;

        // Step 1
        for (int r = 0; r < rows; ++r) {
            // Get pointers to rows
            const unsigned char* img_r = img_.ptr<unsigned char>(r);
            unsigned* ER_r = ER.ptr<unsigned>(r);
            unsigned* RLC_r = RLC.ptr<unsigned>(r);
            int x0;
            int x1 = 0; // Previous value of X
            int f = 0;  // Front detection
            int b = 0;  // Right border compensation
            int er = 0;
            for (int c = 0; c < cols; ++c)
            {
                x0 = img_r[c] > 0;
                f = x0 ^ x1;
                RLC_r[er] = c - b;
                b = b ^ f;
                er = er + f;
                ER_r[c] = er;
                x1 = x0;
            }
            x0 = 0;
            f = x0 ^ x1;
            RLC_r[er] = cols - b;
            er = er + f;
            ner[r] = er;
        }

        // Step 2
        LabelsSolver::Setup();

        // First row
        {
            unsigned* ERA_r = ERA.ptr<unsigned>(0);
            for (int er = 1; er <= ner[0]; er += 2) {
                ERA_r[er] = LabelsSolver::NewLabel();
            }
        }
        for (int r = 1; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ERA_r_prev = (unsigned *)(((char *)ERA_r) - ERA.step.p[0]);
            const unsigned* ER_r_prev = ER.ptr<unsigned>(r - 1);
            const unsigned* RLC_r = RLC.ptr<unsigned>(r);
            for (int er = 1; er <= ner[r]; er += 2) {
                int j0 = RLC_r[er - 1];
                int j1 = RLC_r[er];
                // Check extension in case of 8-connect algorithm
                if (j0 > 0)
                    j0--;
                if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1"
                    j1++;
                int er0 = ER_r_prev[j0];
                int er1 = ER_r_prev[j1];
                // Check label parity: segments are odd
                if (er0 % 2 == 0)
                    er0++;
                if (er1 % 2 == 0)
                    er1--;
                if (er1 >= er0) {
                    int ea = ERA_r_prev[er0];
                    int a = LabelsSolver::FindRoot(ea);
                    for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2"
                        int eak = ERA_r_prev[erk];
                        int ak = LabelsSolver::FindRoot(eak);
                        // Min extraction and propagation
                        if (a < ak)
                            LabelsSolver::UpdateTable(ak, a);
                        if (a > ak)
                        {
                            LabelsSolver::UpdateTable(a, ak);
                            a = ak;
                        }
                    }
                    ERA_r[er] = a; // The global min
                }
                else
                {
                    ERA_r[er] = LabelsSolver::NewLabel();
                }
            }
        }

        // Step 3
        //Mat1i EA(rows, cols);
        //for (int r = 0; r < rows; ++r) {
        //	for (int c = 0; c < cols; ++c) {
        //		EA(r, c) = ERA(r, ER(r, c));
        //	}
        //}
        // Sorry, but we really don't get why this shouldn't be included in the last step

        // Step 4
        n_labels_ = LabelsSolver::Flatten();

        // Step 5

        for (int r = 0; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* labels_r = img_labels_.ptr<unsigned>(r);
            const unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ER_r = ER.ptr<unsigned>(r);
            for (int c = 0; c < cols; ++c)
            {
                //labels(r, c) = A[EA(r, c)];
                labels_r[c] = LabelsSolver::GetLabel(ERA_r[ER_r[c]]); // This is Step 3 and 5 together
            }
        }
    }
};

template <typename LabelsSolver>
class LSL_RLE : public Labeling {
public:
    void PerformLabeling() {

        int rows = img_.rows, cols = img_.cols;

        // Step 1
        Mat1i ER(rows, cols);  // Matrix of relative label (1 label/pixel)
        Mat1i RLC(rows, (cols + 1) & ~1); // MISSING in the paper: RLC requires 2 values/run in row, so width must be next multiple of 2
        int *ner = new int[rows]; //vector<int> ner(rows); // Number of runs 
        for (int r = 0; r < rows; ++r) {
            // Get pointers to rows
            const unsigned char* img_r = img_.ptr<unsigned char>(r);
            unsigned* ER_r = ER.ptr<unsigned>(r);
            unsigned* RLC_r = RLC.ptr<unsigned>(r);
            int x0;
            int x1 = 0; // Previous value of X
            int f = 0;  // Front detection
            int b = 0;  // Right border compensation
            int er = 0;
            for (int c = 0; c < cols; ++c)
            {
                x0 = img_r[c] > 0;
                f = x0 ^ x1;
                if (f) {
                    RLC_r[er] = c - b;
                    b = b ^ 1; //b = b ^ f;
                    er = er + 1; //er = er + f;
                }
                ER_r[c] = er;
                x1 = x0;
            }
            x0 = 0;
            f = x0 ^ x1;
            RLC_r[er] = cols - b;
            er = er + f;
            ner[r] = er;
        }

        // Step 2
        Mat1i ERA(rows, cols + 1, 0); // MISSING in the paper: ERA must have one column more than the input image 
                                      // in order to handle special cases (e.g. lines with chessboard pattern 
                                      // starting with a foreground pixel) 

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        // First row
        {
            unsigned* ERA_r = ERA.ptr<unsigned>(0);
            for (int er = 1; er <= ner[0]; er += 2) {
                ERA_r[er] = LabelsSolver::NewLabel();
            }
        }
        for (int r = 1; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ERA_r_prev = (unsigned *)(((char *)ERA_r) - ERA.step.p[0]);
            const unsigned* ER_r_prev = ER.ptr<unsigned>(r - 1);
            const unsigned* RLC_r = RLC.ptr<unsigned>(r);
            for (int er = 1; er <= ner[r]; er += 2) {
                int j0 = RLC_r[er - 1];
                int j1 = RLC_r[er];
                // Check extension in case of 8-connect algorithm
                if (j0 > 0)
                    j0--;
                if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1"
                    j1++;
                int er0 = ER_r_prev[j0];
                int er1 = ER_r_prev[j1];
                // Check label parity: segments are odd
                if (er0 % 2 == 0)
                    er0++;
                if (er1 % 2 == 0)
                    er1--;
                if (er1 >= er0) {
                    int ea = ERA_r_prev[er0];
                    int a = LabelsSolver::FindRoot(ea);
                    for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2"
                        int eak = ERA_r_prev[erk];
                        int ak = LabelsSolver::FindRoot(eak);
                        // Min extraction and propagation
                        if (a < ak)
                            LabelsSolver::UpdateTable(ak, a);
                        if (a > ak)
                        {
                            LabelsSolver::UpdateTable(a, ak);
                            a = ak;
                        }
                    }
                    ERA_r[er] = a; // The global min
                }
                else
                {
                    ERA_r[er] = LabelsSolver::NewLabel();
                }
            }
        }

        // Step 3
        //Mat1i EA(rows, cols);
        //for (int r = 0; r < rows; ++r) {
        //	for (int c = 0; c < cols; ++c) {
        //		EA(r, c) = ERA(r, ER(r, c));
        //	}
        //}
        // Sorry, but we really don't get why this shouldn't be included in the last step

        // Step 4
        n_labels_ = LabelsSolver::Flatten();

        // Step 5
        img_labels_ = Mat1i(rows, cols);
        for (int r = 0; r < rows; ++r)
        {
            // Get pointers to rows
            unsigned* labels_r = img_labels_.ptr<unsigned>(r);
            const unsigned* ERA_r = ERA.ptr<unsigned>(r);
            const unsigned* ER_r = ER.ptr<unsigned>(r);
            for (int c = 0; c < cols; ++c)
            {
                //labels(r, c) = A[EA(r, c)];
                labels_r[c] = LabelsSolver::GetLabel(ERA_r[ER_r[c]]); // This is Step 3 and 5 together
            }
        }

        delete[] ner;
        LabelsSolver::Dealloc();
    }
};

#endif // !YACCLAB_LABELING_LACASSAGNE_2011_H_
