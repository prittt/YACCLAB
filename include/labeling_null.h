// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_NULL_H_
#define YACCLAB_LABELING_NULL_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class labeling_NULL : public Labeling2D<Connectivity2D::CONN_8> {
public:

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size());

        for (int r = 0; r < img_labels_.rows; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);

            for (int c = 0; c < img_labels_.cols; ++c) {
                    img_labels_row[c] = img_row[c];
            }
        }
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

    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    {
        MemMat<uchar> img(img_);
        MemMat<int> img_labels(img_.size());

        for (int r = 0; r < img_labels.rows; ++r) {
            for (int c = 0; c < img_labels.cols; ++c) {
                img_labels(r, c) = img(r, c);
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
    }

private:
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
        for (int r = 0; r < img_labels_.rows; ++r) {
            // Get rows pointer
            const uchar* const img_row = img_.ptr<uchar>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);

            for (int c = 0; c < img_labels_.cols; ++c) {
                img_labels_row[c] = img_row[c];
            }
        }
    }
};

#endif // !YACCLAB_LABELING_NULL_H_