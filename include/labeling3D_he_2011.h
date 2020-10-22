// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#ifndef YACCLAB_LABELING3D_HE_2011_H_
#define YACCLAB_LABELING3D_HE_2011_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"


//Conditions:
#define CONDITION_V   img_slice00_row00[c] > 0
#define CONDITION_V1  c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_V2  c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_V3  r > 0 && img_slice00_row11[c] > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_V6  r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_V8  c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_V9  s > 0 && img_slice11_row00[c] > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0

// nothing
#define ACTION_1 img_labels_slice00_row00[c] = 0; 
// v <- v9
#define ACTION_2 img_labels_slice00_row00[c] = img_labels_slice11_row00[c];
// v <- v3 
#define ACTION_3 img_labels_slice00_row00[c] = img_labels_slice00_row11[c];
// merge(V3, v12)
#define ACTION_4 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c]);
// merge(V3, v11)
#define ACTION_5 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c - 1]);
// merge(V3, v13)
#define ACTION_6 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]);
// v <- v6
#define ACTION_7 img_labels_slice00_row00[c] = img_labels_slice11_row11[c];
// v <- v1
#define ACTION_8 img_labels_slice00_row00[c] = img_labels_slice00_row00[c - 1]; 
// merge(v1, v10)
#define ACTION_9 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row00[c + 1]); 
// merge(v1, v4)
#define ACTION_10 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]);
// merge(v1, v7)
#define ACTION_11 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v1, v13)
#define ACTION_12 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]);
// v <- v8
#define ACTION_13 img_labels_slice00_row00[c] = img_labels_slice11_row00[c - 1];
// v <- v10
#define ACTION_14 img_labels_slice00_row00[c] = img_labels_slice11_row00[c + 1];
// merge(v2, v10)
#define ACTION_15 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row00[c + 1]);
// merge(v5, v10)
#define ACTION_16 LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v11, v10)
#define ACTION_17 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]);
// v <- v2
#define ACTION_18 img_labels_slice00_row00[c] = img_labels_slice00_row11[c - 1];
// merge(v4, v2)
#define ACTION_19 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]);
// merge(v7, v2)
#define ACTION_20 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v2, V12)
#define ACTION_21 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]);
// merge(v2, V11)
#define ACTION_22 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]);
// merge(v2, V13)
#define ACTION_23 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]);
// merge(v6, V12)
#define ACTION_24 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c]);
// merge(v6, V11)
#define ACTION_25 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c]);
// merge(v6, V13)
#define ACTION_26 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c]);
// merge(v8, v10)
#define ACTION_27 LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v4)
#define ACTION_28 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v7)
#define ACTION_29 LabelsSolver::Merge(img_labels_slice11_row00[c - 1], img_labels_slice11_row11[c + 1]);
// v <- v5
#define ACTION_30 img_labels_slice00_row00[c] = img_labels_slice11_row11[c - 1];
// merge(v4, v5)
#define ACTION_31 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v7, v5)
#define ACTION_32 LabelsSolver::Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V12)
#define ACTION_33 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c - 1]);
// merge(v5, V11)
#define ACTION_34 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V13)
#define ACTION_35 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c - 1]);
// v <- v12
#define ACTION_36 img_labels_slice00_row00[c] = img_labels_slice11_row01[c];
// merge(v12, v4)
#define ACTION_37 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]);
// merge(v12, v7)
#define ACTION_38 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]);
// v <- v4
#define ACTION_39 img_labels_slice00_row00[c] = img_labels_slice00_row11[c + 1];
// merge(v11, v4)
#define ACTION_40 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]);
// merge(v13, v4)
#define ACTION_41 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]);
// v <- v7
#define ACTION_42 img_labels_slice00_row00[c] = img_labels_slice11_row11[c + 1];
// merge(v11, v7)
#define ACTION_43 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v13, v7)
#define ACTION_44 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]);
// v <- v11
#define ACTION_45 img_labels_slice00_row00[c] = img_labels_slice11_row01[c - 1];
// merge(v13, v11)
#define ACTION_46 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]);
// v <- v13
#define ACTION_47 img_labels_slice00_row00[c] = img_labels_slice11_row01[c + 1];
// v <- newlabel
#define ACTION_48 img_labels_slice00_row00[c] = LabelsSolver::NewLabel();
// merge(v8, v13)
#define ACTION_49 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]);


template <typename LabelsSolver>
class LEB_3D : public Labeling3D<Connectivity3D::CONN_26> {
public:
    LEB_3D() {}

	void PerformLabeling()
	{
		img_labels_.create(3, img_.size.p, CV_32SC1);

		LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

		// 3D Rosenfeld Mask
        // +---+---+---+
        // |v5 |v6 |v7 |
        // +---+---+---+
        // |v8 |v9 |v10|
        // +---+---+---+
        // |v11|v12|v13|
        // +---+---+---+
        //
        // +---+---+---+
		// |v2 |v3 |v4 |
		// +---+---+---+
		// |v1 |v  |
		// +---+---+

		// First scan
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s++) {

			for (unsigned int r = 0; r < h; r++) {

				// Row pointers for the input image (current slice)
				const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
				const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);

				// Row pointers for the input image (previous slice)
				const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
				unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

				// Row pointers for the output image (previous slice)
				unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);

				for (unsigned int c = 0; c < w; c++) {

#include "labeling3D_he_2011_tree.inc.h"

				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {
					img_row[c] = LabelsSolver::GetLabel(img_row[c]);
				}
				img_row += img_labels_.step[1] / sizeof(int);
			}
		}

		LabelsSolver::Dealloc(); // Memory deallocation of the labels solver

	}

	void PerformLabelingWithSteps() {
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

		{
#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

//Conditions:
#define CONDITION_V   img(s, r, c) > 0
#define CONDITION_V1  c > 0 && img(s, r, c - 1) > 0
#define CONDITION_V2  c > 0 && r > 0 && img(s, r - 1, c - 1) > 0
#define CONDITION_V3  r > 0 && img(s, r - 1, c) > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img(s, r - 1, c + 1) > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img(s - 1, r - 1, c - 1) > 0
#define CONDITION_V6  r > 0 && s > 0 && img(s - 1, r - 1, c) > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img(s - 1, r - 1, c + 1) > 0
#define CONDITION_V8  c > 0 && s > 0 && img(s - 1, r, c - 1) > 0
#define CONDITION_V9  s > 0 && img(s - 1, r, c) > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img(s - 1, r, c + 1) > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img(s - 1, r + 1, c - 1) > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img(s - 1, r + 1, c) > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 1) > 0

// nothing
#define ACTION_1 img_labels(s, r, c) = 0; 
// v <- v9
#define ACTION_2 img_labels(s, r, c) = img_labels(s - 1, r, c);
// v <- v3 
#define ACTION_3 img_labels(s, r, c) = img_labels(s, r - 1, c);
// merge(V3, v12)
#define ACTION_4 LabelsSolver::MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c));
// merge(V3, v11)
#define ACTION_5 LabelsSolver::MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c - 1));
// merge(V3, v13)
#define ACTION_6 LabelsSolver::MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c + 1));
// v <- v6
#define ACTION_7 img_labels(s, r, c) = img_labels(s - 1, r - 1, c);
// v <- v1
#define ACTION_8 img_labels(s, r, c) = img_labels(s, r, c - 1); 
// merge(v1, v10)
#define ACTION_9 LabelsSolver::MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r, c + 1)); 
// merge(v1, v4)
#define ACTION_10 LabelsSolver::MemMerge(img_labels(s, r, c - 1), img_labels(s, r - 1, c + 1));
// merge(v1, v7)
#define ACTION_11 LabelsSolver::MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v1, v13)
#define ACTION_12 LabelsSolver::MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r + 1, c + 1));
// v <- v8
#define ACTION_13 img_labels(s, r, c) = img_labels(s - 1, r, c - 1);
// v <- v10
#define ACTION_14 img_labels(s, r, c) = img_labels(s - 1, r, c + 1);
// merge(v2, v10)
#define ACTION_15 LabelsSolver::MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r, c + 1));
// merge(v5, v10)
#define ACTION_16 LabelsSolver::MemMerge(img_labels(s - 1, r, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v11, v10)
#define ACTION_17 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r, c + 1));
// v <- v2
#define ACTION_18 img_labels(s, r, c) = img_labels(s, r - 1, c - 1);
// merge(v4, v2)
#define ACTION_19 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s, r - 1, c - 1));
// merge(v7, v2)
#define ACTION_20 LabelsSolver::MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v2, V12)
#define ACTION_21 LabelsSolver::MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c));
// merge(v2, V11)
#define ACTION_22 LabelsSolver::MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c - 1));
// merge(v2, V13)
#define ACTION_23 LabelsSolver::MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c + 1));
// merge(v6, V12)
#define ACTION_24 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c));
// merge(v6, V11)
#define ACTION_25 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c));
// merge(v6, V13)
#define ACTION_26 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c));
// merge(v8, v10)
#define ACTION_27 LabelsSolver::MemMerge(img_labels(s - 1, r, c + 1), img_labels(s - 1, r, c - 1));
// merge(v8, v4)
#define ACTION_28 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r, c - 1));
// merge(v8, v7)
#define ACTION_29 LabelsSolver::MemMerge(img_labels(s - 1, r, c - 1), img_labels(s - 1, r - 1, c + 1));
// v <- v5
#define ACTION_30 img_labels(s, r, c) = img_labels(s - 1, r - 1, c - 1);
// merge(v4, v5)
#define ACTION_31 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v7, v5)
#define ACTION_32 LabelsSolver::MemMerge(img_labels(s - 1, r - 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V12)
#define ACTION_33 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V11)
#define ACTION_34 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V13)
#define ACTION_35 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// v <- v12
#define ACTION_36 img_labels(s, r, c) = img_labels(s - 1, r + 1, c);
// merge(v12, v4)
#define ACTION_37 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c));
// merge(v12, v7)
#define ACTION_38 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c + 1));
// v <- v4
#define ACTION_39 img_labels(s, r, c) = img_labels(s, r - 1, c + 1);
// merge(v11, v4)
#define ACTION_40 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c - 1));
// merge(v13, v4)
#define ACTION_41 LabelsSolver::MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c + 1));
// v <- v7
#define ACTION_42 img_labels(s, r, c) = img_labels(s - 1, r - 1, c + 1);
// merge(v11, v7)
#define ACTION_43 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v13, v7)
#define ACTION_44 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c + 1));
// v <- v11
#define ACTION_45 img_labels(s, r, c) = img_labels(s - 1, r + 1, c - 1);
// merge(v13, v11)
#define ACTION_46 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r + 1, c - 1));
// v <- v13
#define ACTION_47 img_labels(s, r, c) = img_labels(s - 1, r + 1, c + 1);
// v <- newlabel
#define ACTION_48 img_labels(s, r, c) = LabelsSolver::MemNewLabel();
// merge(v8, v13)
#define ACTION_49 LabelsSolver::MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r, c - 1));
		}

		LabelsSolver::MemAlloc(UPPER_BOUND_26_CONNECTIVITY); // Equivalence solver

		MemVol<unsigned char> img(img_);
		MemVol<int> img_labels(img_.size.p);

		LabelsSolver::MemSetup();

        //uint64_t accesses_count = 0;

		// First scan
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {

#include "labeling3D_he_2011_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::MemFlatten();

		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {
					img_labels(s, r, c) = LabelsSolver::MemGetLabel(img_labels(s, r, c));
				}
			}
		}

		// Store total accesses in the output vector 'accesses'
		accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

		accesses[MD_BINARY_MAT] = (unsigned long)img.GetTotalAccesses();
		accesses[MD_LABELED_MAT] = (unsigned long)img_labels.GetTotalAccesses();
		accesses[MD_EQUIVALENCE_VEC] = (unsigned long)LabelsSolver::MemTotalAccesses();

		img_labels_ = img_labels.GetImage();

		LabelsSolver::MemDealloc(); // Memory deallocation of the labels solver

		{
#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

			//Conditions:
#define CONDITION_V   img_slice00_row00[c] > 0
#define CONDITION_V1  c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_V2  c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_V3  r > 0 && img_slice00_row11[c] > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_V6  r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_V8  c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_V9  s > 0 && img_slice11_row00[c] > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0

// nothing
#define ACTION_1 img_labels_slice00_row00[c] = 0; 
// v <- v9
#define ACTION_2 img_labels_slice00_row00[c] = img_labels_slice11_row00[c];
// v <- v3 
#define ACTION_3 img_labels_slice00_row00[c] = img_labels_slice00_row11[c];
// merge(V3, v12)
#define ACTION_4 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c]);
// merge(V3, v11)
#define ACTION_5 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c - 1]);
// merge(V3, v13)
#define ACTION_6 LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]);
// v <- v6
#define ACTION_7 img_labels_slice00_row00[c] = img_labels_slice11_row11[c];
// v <- v1
#define ACTION_8 img_labels_slice00_row00[c] = img_labels_slice00_row00[c - 1]; 
// merge(v1, v10)
#define ACTION_9 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row00[c + 1]); 
// merge(v1, v4)
#define ACTION_10 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]);
// merge(v1, v7)
#define ACTION_11 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v1, v13)
#define ACTION_12 LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]);
// v <- v8
#define ACTION_13 img_labels_slice00_row00[c] = img_labels_slice11_row00[c - 1];
// v <- v10
#define ACTION_14 img_labels_slice00_row00[c] = img_labels_slice11_row00[c + 1];
// merge(v2, v10)
#define ACTION_15 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row00[c + 1]);
// merge(v5, v10)
#define ACTION_16 LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v11, v10)
#define ACTION_17 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]);
// v <- v2
#define ACTION_18 img_labels_slice00_row00[c] = img_labels_slice00_row11[c - 1];
// merge(v4, v2)
#define ACTION_19 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]);
// merge(v7, v2)
#define ACTION_20 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v2, V12)
#define ACTION_21 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]);
// merge(v2, V11)
#define ACTION_22 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]);
// merge(v2, V13)
#define ACTION_23 LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]);
// merge(v6, V12)
#define ACTION_24 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c]);
// merge(v6, V11)
#define ACTION_25 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c]);
// merge(v6, V13)
#define ACTION_26 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c]);
// merge(v8, v10)
#define ACTION_27 LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v4)
#define ACTION_28 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v7)
#define ACTION_29 LabelsSolver::Merge(img_labels_slice11_row00[c - 1], img_labels_slice11_row11[c + 1]);
// v <- v5
#define ACTION_30 img_labels_slice00_row00[c] = img_labels_slice11_row11[c - 1];
// merge(v4, v5)
#define ACTION_31 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v7, v5)
#define ACTION_32 LabelsSolver::Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V12)
#define ACTION_33 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c - 1]);
// merge(v5, V11)
#define ACTION_34 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V13)
#define ACTION_35 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c - 1]);
// v <- v12
#define ACTION_36 img_labels_slice00_row00[c] = img_labels_slice11_row01[c];
// merge(v12, v4)
#define ACTION_37 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]);
// merge(v12, v7)
#define ACTION_38 LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]);
// v <- v4
#define ACTION_39 img_labels_slice00_row00[c] = img_labels_slice00_row11[c + 1];
// merge(v11, v4)
#define ACTION_40 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]);
// merge(v13, v4)
#define ACTION_41 LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]);
// v <- v7
#define ACTION_42 img_labels_slice00_row00[c] = img_labels_slice11_row11[c + 1];
// merge(v11, v7)
#define ACTION_43 LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v13, v7)
#define ACTION_44 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]);
// v <- v11
#define ACTION_45 img_labels_slice00_row00[c] = img_labels_slice11_row01[c - 1];
// merge(v13, v11)
#define ACTION_46 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]);
// v <- v13
#define ACTION_47 img_labels_slice00_row00[c] = img_labels_slice11_row01[c + 1];
// v <- newlabel
#define ACTION_48 img_labels_slice00_row00[c] = LabelsSolver::NewLabel();
// merge(v8, v13)
#define ACTION_49 LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]);
		}

	}


private:
	double Alloc()
	{
		// Memory allocation of the labels solver
		double ls_t = LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY, perf_);
		// Memory allocation for the output image
		perf_.start();
		img_labels_.create(3, img_.size.p, CV_32SC1);
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
	void Dealloc() {
		LabelsSolver::Dealloc();
		// No free for img_labels_ because it is required at the end of the algorithm 
	}
	void FirstScan() {
		LabelsSolver::Setup(); // Labels solver initialization
		
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s++) {

			for (unsigned int r = 0; r < h; r++) {

				// Row pointers for the input image (current slice)
				const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
				const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);

				// Row pointers for the input image (previous slice)
				const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
				unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

				// Row pointers for the output image (previous slice)
				unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);

				for (unsigned int c = 0; c < w; c++) {

#include "labeling3D_he_2011_tree.inc.h"

				}
			} // Rows cycle end
		} // Planes cycle end
	}
		

	void SecondScan() {
		LabelsSolver::Flatten();

		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {
					img_row[c] = LabelsSolver::GetLabel(img_row[c]);
				}
				img_row += img_labels_.step.p[1] / sizeof(int);
			}
		}
	}
};


#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

#endif // YACCLAB_LABELING3D_HE_2011_H_