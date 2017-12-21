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

#ifndef YACCLAB_LABELING_FCHANG_2003_H_
#define YACCLAB_LABELING_FCHANG_2003_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class CT : public Labeling {
public:
    CT() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0);

        n_labels_ = 0;
        for (int y = 0; y < img_.rows; y++) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            unsigned int* const img_labels_out_row = img_labels_.ptr<unsigned int>(y);
            for (int x = 0; x < img_.cols; x++) {

                if (img_row[x] == byF) {
                    // Case 1
                    if (img_labels_out_row[x] == 0 && (x == 0 || img_row[x - 1] != byF)) {
                        n_labels_++;
                        ContourTracing(x, y, n_labels_, true);
                        continue;
                    }
                    // Case 2
                    else if (x < img_.cols - 1 && img_row[x + 1] != byF && img_labels_out_row[x + 1] != -1) {
                        if (img_labels_out_row[x] == 0) {
                            // Current pixel unlabeled
                            // Assing label of left pixel
                            ContourTracing(x, y, img_labels_out_row[x - 1], false);
                        }
                        else {
                            ContourTracing(x, y, img_labels_out_row[x], false);
                        }
                        continue;
                    }
                    // case 3
                    else if (img_labels_out_row[x] == 0) {
                        img_labels_out_row[x] = img_labels_out_row[x - 1];
                    }
                }
            }
        }

        // Reassign to contour background value (0)
        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            for (int c = 0; c < img_labels_.cols; ++c) {
                if (img_labels_row[c] == -1)
                    img_labels_row[c] = 0;
            }
        }

        n_labels_++; // To count also background label
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
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        n_labels_ = 0;
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {

                if (img(y,x) == byF) {
                    // Case 1
                    if (img_labels(y,x) == 0 && (x == 0 || img(y, x - 1) != byF)) {
                        n_labels_++;
                        MemContourTracing(img, img_labels, x, y, n_labels_, true);
                        continue;
                    }
                    // Case 2
                    else if (x < img.cols - 1 && img(y, x + 1) != byF && img_labels(y, x + 1) != -1) {
                        if (img_labels(y,x) == 0) {
                            // Current pixel unlabeled
                            // Assing label of left pixel
                            MemContourTracing(img, img_labels, x, y, img_labels(y, x - 1), false);
                        }
                        else {
                            MemContourTracing(img, img_labels, x, y, img_labels(y, x), false);
                        }
                        continue;
                    }
                    // case 3
                    else if (img_labels(y, x) == 0) {
                        img_labels(y, x) = img_labels(y, x - 1);
                    }
                }
            }
        }

        // Reassign to contour background value (0)
        for (int r = 0; r < img_labels.rows; ++r) {
            for (int c = 0; c < img_labels.cols; ++c) {
                if (img_labels(r, c) == -1)
                    img_labels(r, c) = 0;
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();

        img_labels_ = img_labels.GetImage();

        n_labels_++; // To count also background label
    }

private:
    unsigned char byF = 1; // Byte of foreground (fixed for YACCLAB)

    cv::Point2i Tracer(const cv::Point2i &p, int &i_prev, bool &b_isolated) {
        int i_first, i_next;

        // Find the direction to be analyzed
        i_first = i_next = (i_prev + 2) % 8;

        cv::Point2i crd_next;
        do {
            switch (i_next) {
            case 0: crd_next = p + cv::Point2i(1, 0); break;
            case 1: crd_next = p + cv::Point2i(1, 1); break;
            case 2: crd_next = p + cv::Point2i(0, 1); break;
            case 3: crd_next = p + cv::Point2i(-1, 1); break;
            case 4: crd_next = p + cv::Point2i(-1, 0); break;
            case 5: crd_next = p + cv::Point2i(-1, -1); break;
            case 6: crd_next = p + cv::Point2i(0, -1); break;
            case 7: crd_next = p + cv::Point2i(1, -1); break;
            }

            if (crd_next.y >= 0 && crd_next.x >= 0 && crd_next.y < img_.rows && crd_next.x < img_.cols) {
                if (img_(crd_next.y, crd_next.x) == byF) {
                    i_prev = (i_next + 4) % 8;
                    return crd_next;
                }
                else
                    img_labels_(crd_next.y, crd_next.x) = -1;
            }

            i_next = (i_next + 1) % 8;
        } while (i_next != i_first);

        b_isolated = true;
        return p;
    }
    void ContourTracing(int x, int y, int i_label, bool b_external) {
        cv::Point2i s(x, y), T, crd_next_point, crd_cur_point;

        // The current point is labeled 
        img_labels_(s.y, s.x) = i_label;

        bool b_isolated(false);
        int i_previous_contour_point;
        if (b_external)
            i_previous_contour_point = 6;
        else
            i_previous_contour_point = 7;

        // First call to Tracer
        crd_next_point = T = Tracer(s, i_previous_contour_point, b_isolated);
        if (b_isolated)
            return;

        do {
            crd_cur_point = crd_next_point;
            img_labels_(crd_cur_point.y, crd_cur_point.x) = i_label;
            crd_next_point = Tracer(crd_cur_point, i_previous_contour_point, b_isolated);
        } while (!(crd_cur_point == s && crd_next_point == T));
    }

    cv::Point2i MemTracer(MemMat<unsigned char> &img, MemMat<int> &img_labels, const cv::Point2i &p, int &i_prev, bool &b_isolated) {
        int i_first, i_next;

        // Find the direction to be analyzed
        i_first = i_next = (i_prev + 2) % 8;

        cv::Point2i crd_next;
        do {
            switch (i_next) {
            case 0: crd_next = p + cv::Point2i(1, 0); break;
            case 1: crd_next = p + cv::Point2i(1, 1); break;
            case 2: crd_next = p + cv::Point2i(0, 1); break;
            case 3: crd_next = p + cv::Point2i(-1, 1); break;
            case 4: crd_next = p + cv::Point2i(-1, 0); break;
            case 5: crd_next = p + cv::Point2i(-1, -1); break;
            case 6: crd_next = p + cv::Point2i(0, -1); break;
            case 7: crd_next = p + cv::Point2i(1, -1); break;
            }

            if (crd_next.y >= 0 && crd_next.x >= 0 && crd_next.y < img.rows && crd_next.x < img.cols) {
                if (img(crd_next.y, crd_next.x) == byF) {
                    i_prev = (i_next + 4) % 8;
                    return crd_next;
                }
                else
                    img_labels(crd_next.y, crd_next.x) = -1;
            }

            i_next = (i_next + 1) % 8;
        } while (i_next != i_first);

        b_isolated = true;
        return p;
    }
    void MemContourTracing(MemMat<unsigned char> &img, MemMat<int> &img_labels, int x, int y, int i_label, bool b_external) {
        cv::Point2i s(x, y), T, crd_next_point, crd_cur_point;

        // The current point is labeled 
        img_labels(s.y, s.x) = i_label;

        bool b_isolated(false);
        int i_previous_contour_point;
        if (b_external)
            i_previous_contour_point = 6;
        else
            i_previous_contour_point = 7;

        // First call to Tracer
        crd_next_point = T = MemTracer(img, img_labels, s, i_previous_contour_point, b_isolated);
        if (b_isolated)
            return;

        do {
            crd_cur_point = crd_next_point;
            img_labels(crd_cur_point.y, crd_cur_point.x) = i_label;
            crd_next_point = MemTracer(img, img_labels, crd_cur_point, i_previous_contour_point, b_isolated);
        } while (!(crd_cur_point == s && crd_next_point == T));
    }
    
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
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization

        n_labels_ = 0;
        for (int y = 0; y < img_.rows; y++) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            unsigned int* const img_labels_out_row = img_labels_.ptr<unsigned int>(y);
            for (int x = 0; x < img_.cols; x++) {

                if (img_row[x] == byF) {
                    // Case 1
                    if (img_labels_out_row[x] == 0 && (x == 0 || img_row[x - 1] != byF)) {
                        n_labels_++;
                        ContourTracing(x, y, n_labels_, true);
                        continue;
                    }
                    // Case 2
                    else if (x < img_.cols - 1 && img_row[x + 1] != byF && img_labels_out_row[x + 1] != -1) {
                        if (img_labels_out_row[x] == 0) {
                            // Current pixel unlabeled
                            // Assing label of left pixel
                            ContourTracing(x, y, img_labels_out_row[x - 1], false);
                        }
                        else {
                            ContourTracing(x, y, img_labels_out_row[x], false);
                        }
                        continue;
                    }
                    // case 3
                    else if (img_labels_out_row[x] == 0) {
                        img_labels_out_row[x] = img_labels_out_row[x - 1];
                    }
                }
            }
        }

        // Reassign to contour background value (0)
        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            for (int c = 0; c < img_labels_.cols; ++c) {
                if (img_labels_row[c] == -1)
                    img_labels_row[c] = 0;
            }
        }

        n_labels_++; // To count also background label
    }
};

#endif // YACCLAB_LABELING_FCHANG_2003_H_