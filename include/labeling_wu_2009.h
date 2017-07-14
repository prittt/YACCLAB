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

#ifndef YACCLAB_LABELING_WU_2009_H_
#define YACCLAB_LABELING_WU_2009_H_

#include <opencv2/core.hpp>

#include <vector>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class SAUF : public Labeling {
public:
    SAUF() {}

    unsigned PerformLabeling() override
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(), 0);

        const size_t P_length = UPPER_BOUND_8_CONNECTIVITY;
        //array P_ for equivalences resolution
        P_ = (unsigned *)cv::fastMalloc(sizeof(unsigned) * P_length);

        P_[0] = 0;	//first label is for background pixels
        unsigned lunique = 1;

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // first scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            uchar const * const img_row = img_.ptr<uchar>(r);
            uchar const * const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
#define condition_p c>0 && r>0 && img_row_prev[c - 1]>0
#define condition_q r>0 && img_row_prev[c]>0
#define condition_r c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                if (condition_x) {
                    if (condition_q) {
                        //x <- q
                        img_labels_row[c] = img_labels_row_prev[c];
                    }
                    else {
                        // q = 0
                        if (condition_r) {
                            if (condition_p) {
                                // x <- merge(p,r)
                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]);
                            }
                            else {
                                // p = q = 0
                                if (condition_s) {
                                    // x <- merge(s,r)
                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row[c - 1], img_labels_row_prev[c + 1]);
                                }
                                else {
                                    // p = q = s = 0
                                    // x <- r
                                    img_labels_row[c] = img_labels_row_prev[c + 1];
                                }
                            }
                        }
                        else {
                            // r = q = 0
                            if (condition_p) {
                                // x <- p
                                img_labels_row[c] = img_labels_row_prev[c - 1];
                            }
                            else {
                                // r = q = p = 0
                                if (condition_s) {
                                    img_labels_row[c] = img_labels_row[c - 1];
                                }
                                else {
                                    //new label
                                    img_labels_row[c] = lunique;
                                    P_[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                    }
                }
                else {
                    //Nothing to do, x is a background pixel
                }
            }
        }

        //second scan
        unsigned n_labels = ls_.Flatten(P_, lunique);

        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = P_[*img_row_start];
            }
        }

        cv::fastFree(P_);

        return n_labels;

#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
    }

    unsigned PerformLabelingMem(std::vector<unsigned long int>& accesses) override
    {
        const int h = img_.rows;
        const int w = img_.cols;

        const size_t P_length = UPPER_BOUND_8_CONNECTIVITY;

        //Data structure for memory test
        MemMat<uchar> img(img_);
        MemMat<int> img_labels(img_.size(), 0);
        MemVector<unsigned> P_(P_length);						 // Vector P_ for equivalences resolution

        P_[0] = 0;	//first label is for background pixels
        unsigned lunique = 1;

        //first scan

        //Rosenfeld Mask
        //+-+-+-+
        //|p|q|r|
        //+-+-+-+
        //|s|x|
        //+-+-+

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
#define condition_p c>0 && r>0 && img(r-1 , c-1)>0
#define condition_q r>0 && img(r-1, c)>0
#define condition_r c < w - 1 && r > 0 && img(r-1,c+1)>0
#define condition_s c > 0 && img(r,c-1)>0
#define condition_x img(r,c)>0

                if (condition_x) {
                    if (condition_q) {
                        //x <- q
                        img_labels(r, c) = img_labels(r - 1, c);
                    }
                    else {
                        //q = 0
                        if (condition_r) {
                            if (condition_p) {
                                //x <- merge(p,r)
                                img_labels(r, c) = ls_.Merge(P_, (unsigned)img_labels(r - 1, c - 1), (unsigned)img_labels(r - 1, c + 1));
                            }
                            else {
                                //p = q = 0
                                if (condition_s) {
                                    //x <- merge(s,r)
                                    img_labels(r, c) = ls_.Merge(P_, (unsigned)img_labels(r, c - 1), (unsigned)img_labels(r - 1, c + 1));
                                }
                                else {
                                    //p = q = s = 0
                                    //x <- r
                                    img_labels(r, c) = img_labels(r - 1, c + 1);
                                }
                            }
                        }
                        else {
                            //r = q = 0
                            if (condition_p) {
                                //x <- p
                                img_labels(r, c) = img_labels(r - 1, c - 1);
                            }
                            else {
                                //r = q = p = 0
                                if (condition_s) {
                                    img_labels(r, c) = img_labels(r, c - 1);
                                }
                                else {
                                    //new label
                                    img_labels(r, c) = lunique;
                                    P_[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                    }
                }
                else {
                    //Nothing to do, x is a background pixel
                }
            }
        }

        //second scan
        unsigned n_labels = ls_.Flatten(P_, lunique);

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                img_labels(r, c) = P_[img_labels(r, c)];
            }
        }

        //Store total accesses in the output vector 'accesses'
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAcesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAcesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)P_.GetTotalAcesses();

        //a = img_labels.GetImage();

        return n_labels;

#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
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
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(), 0);
        P_[0] = 0;	//first label is for background pixels
        unsigned lunique = 1;

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // first scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            uchar const * const img_row = img_.ptr<uchar>(r);
            uchar const * const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
#define condition_p c>0 && r>0 && img_row_prev[c - 1]>0
#define condition_q r>0 && img_row_prev[c]>0
#define condition_r c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                if (condition_x) {
                    if (condition_q) {
                        //x <- q
                        img_labels_row[c] = img_labels_row_prev[c];
                    }
                    else {
                        // q = 0
                        if (condition_r) {
                            if (condition_p) {
                                // x <- merge(p,r)
                                img_labels_row[c] = ls_.Merge(P_, img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]);
                            }
                            else {
                                // p = q = 0
                                if (condition_s) {
                                    // x <- merge(s,r)
                                    img_labels_row[c] = ls_.Merge(P_, img_labels_row[c - 1], img_labels_row_prev[c + 1]);
                                }
                                else {
                                    // p = q = s = 0
                                    // x <- r
                                    img_labels_row[c] = img_labels_row_prev[c + 1];
                                }
                            }
                        }
                        else {
                            // r = q = 0
                            if (condition_p) {
                                // x <- p
                                img_labels_row[c] = img_labels_row_prev[c - 1];
                            }
                            else {
                                // r = q = p = 0
                                if (condition_s) {
                                    img_labels_row[c] = img_labels_row[c - 1];
                                }
                                else {
                                    //new label
                                    img_labels_row[c] = lunique;
                                    P_[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                    }
                }
                else {
                    //Nothing to do, x is a background pixel
                }
            }
        }

        return lunique;
    }
    unsigned SecondScan(const unsigned& lunique) override
    {
        unsigned n_labels = ls_.Flatten(P_, lunique);

        //second scan
        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = P_[*img_row_start];
            }
        }

        return n_labels;
    }
};

#endif // !YACCLAB_LABELING_WU_2009_H_