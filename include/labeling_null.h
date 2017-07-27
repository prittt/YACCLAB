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

#ifndef YACCLAB_LABELING_NULL_H_
#define YACCLAB_LABELING_NULL_H_

#include <opencv2/core.hpp>

#include <vector>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

//class labeling_NULL : public Labeling {
//public:
//    labeling_NULL() {}
//    ~labeling_NULL() {}
//    void AllocateMemory() override {}
//    void DeallocateMemory() override {}
//
//    unsigned PerformLabeling() override
//    {
//        img_labels_ = cv::Mat1i(img_.size(), 0);
//
//        for (int r = 0; r < img_labels_.rows; ++r) {
//            // Get rows pointer
//            const uchar* const img_row = img_.ptr<uchar>(r);
//            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
//
//            for (int c = 0; c < img_labels_.cols; ++c) {
//                if (img_row[c] > 0) {
//                    img_labels_row[c] = 1;
//                }
//            }
//        }
//
//        return 0;
//    }
//};

#endif // !YACCLAB_LABELING_NULL_H_