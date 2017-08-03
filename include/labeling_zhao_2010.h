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

#ifndef YACCLAB_LABELING_ZHAO_2010_H_
#define YACCLAB_LABELING_ZHAO_2010_H_

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class SBLA : public Labeling {
public:
    SBLA() {}

    void PerformLabeling()
    {
        int N = img_.cols;
        int M = img_.rows;
        img_labels_ = cv::Mat1i(M, N);

        // Fix for first pixel!
        n_labels_ = 0;
        uchar firstpixel = img_(0, 0);
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_)(0, 0) = 0;
            if (img_(0, 1) == 0 && img_(1, 0) == 0 && img_(1, 1) == 0)
                n_labels_ = 1;
        }

        // Stripe extraction and representation
        unsigned int* img_labels_row_prev = nullptr;
        int rN = 0;
        int r1N = N;
        int N2 = N * 2;
        for (int r = 0; r < M; r += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            for (int c = 0; c < N; ++c) {
                img_labels_row[c] = img_row[c];
                if (r + 1 < M)
                    img_labels_row_fol[c] = img_row_fol[c];
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels_row[c];
                int oddpix = r + 1 < M ? img_labels_row_fol[c] : 0;

                // Step 2
                int Gp;
                if (oddpix) {
                    img_labels_row_fol[c] = Gp = -(r1N + c);
                    if (evenpix)
                        img_labels_row[c] = Gp;
                }
                else if (evenpix)
                    img_labels_row[c] = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels_row[c];
                    int oddpix = r + 1 < M ? img_labels_row_fol[c] : 0;

                    if (oddpix) {
                        img_labels_row_fol[c] = Gp;
                        if (evenpix)
                            img_labels_row[c] = Gp;
                    }
                    else if (evenpix)
                        img_labels_row[c] = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels_row[i];
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels_row[i]);
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels_row_prev[j];
                        if (!curpix)
                            continue;

                        int newroot = FindRoot(reinterpret_cast<int*>(img_labels_.data), curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels_row_prev[j]);
                    }
                }
            }
            img_labels_row_prev = img_labels_row_fol;
            rN += N2;
            r1N += N2;
        }

        // Label assignment
        int *img_labelsdata = reinterpret_cast<int*>(img_labels_.data);
        for (int i = 0; i < M * N; ++i) {
            // FindRoot_GetLabel
            int pos = img_labelsdata[i];
            if (pos >= 0)
                continue;

            while (pos != img_labelsdata[-pos] && img_labelsdata[-pos] < 0)
                pos = img_labelsdata[-pos];
            if (img_labelsdata[-pos] < 0)
                img_labelsdata[-pos] = ++n_labels_;

            // Assign final label
            img_labelsdata[i] = img_labelsdata[-pos];
        }

        // Fix for first pixel!
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_)(0, 0) = firstpixel;
            if (img_labels_(0, 1))
                img_labels_(0, 0) = img_labels_(0, 1);
            else if (img_labels_(1, 0))
                img_labels_(0, 0) = img_labels_(1, 0);
            else if (img_labels_(1, 1))
                img_labels_(0, 0) = img_labels_(1, 1);
            else
                img_labels_(0, 0) = 1;
        }

        n_labels_++; // To count also background
    }

private:
    inline int FindRoot(int *img_labels, int pos)
    {
        while (true) {
            int tmppos = img_labels[-pos];
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
        return pos;
    }

    inline void FindRootAndCompress(int *img_labels, int pos, int newroot)
    {
        while (true) {
            int tmppos = img_labels[-pos];
            if (tmppos == newroot)
                break;
            img_labels[-pos] = newroot;
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
    }
};


#endif YACCLA_LABELING_ZHAO_2010_H_