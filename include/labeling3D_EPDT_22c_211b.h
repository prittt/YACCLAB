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

#ifndef YACCLAB_LABELING3D_BBDT_22c_H_
#define YACCLAB_LABELING3D_BBDT_22c_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LA r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_RA r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_TB c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UA r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UB c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VA c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_NA c > 1 && s > 0 && img_slice11_row00[c - 2] > 0
#define CONDITION_PB c < w - 3 && s > 0 && img_slice11_row00[c + 3] > 0
#define CONDITION_WA c > 1 && img_slice00_row00[c - 2] > 0

//Actions:
#include "labeling3D_BBDT_22c_action_definition.inc.h"

template <typename LabelsSolver>
class BBDT_3D_22c : public Labeling3D<CONN_26> {
public:
	BBDT_3D_22c() {}

	void PerformLabeling()
	{
		img_labels_.create(3, img_.size.p, CV_32SC1);

		LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

		// First scan
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {

			for (unsigned int r = 0; r < h; r += 1) {

				const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
				// T, W slice
				//const unsigned char* const img_slice00_row12 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -2);
				const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * -1);
				// img_slice00_row00 defined above
				//const unsigned char* const img_slice00_row01 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * 1);

				// K, N, Q slice
				//const unsigned char* const img_slice11_row12 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
				const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
				//const unsigned char* const img_slice11_row02 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
				//const unsigned char* const img_slice11_row03 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);


				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
				// T, W slice
				//unsigned* const img_labels_slice00_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -2);
				unsigned* const img_labels_slice00_row11 = (unsigned*)(((char*)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);
				// img_labels_slice00_row00 defined above
				//unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * 1);

				// K, N, Q slice
				//unsigned* const img_labels_slice11_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -2);
				unsigned* const img_labels_slice11_row11 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);
				//unsigned* const img_labels_slice11_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 2);
				//unsigned* const img_labels_slice11_row03 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 3);

				// V -- old -- V
				//// Row pointers for the output image (current slice)
				//unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
				//unsigned* const img_labels_slice00_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -2);
				//unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);
				//unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * 1);

				//// Row pointers for the output image (previous slice)
				//unsigned* const img_labels_slice11_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -2);
				//unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
				//unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
				//unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);
				//unsigned* const img_labels_slice11_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 2);
				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_BBDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::Flatten();
		//unsigned* const img_labels_slice12_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);

		//char * img_labels_row = reinterpret_cast<char*>(img_labels_.data);
		//unsigned char * img_row = reinterpret_cast<unsigned char*>(img_.data);

		//const unsigned char* const img_row = img_.ptr<unsigned char>(0);
		//unsigned* const img_labels_row = img_labels_.ptr<unsigned>(0);

		//const unsigned char* const img_row = img_.ptr<unsigned char>();
		//int* const img_labels_row = img_labels_.ptr<int>();

		// NEW VERSION BELOW, OLD IN labeling3D_BBDT_19c.h
		int rows = h;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
				// Get rows pointer
				const unsigned char* const img_row = img_.ptr<unsigned char>(s, r);

				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(s, r);
				int c = 0;
				for (; c < e_cols; c += 2) {
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
					}
					else {
						img_labels_row[c] = 0;
						img_labels_row[c + 1] = 0;
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					int iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = LabelsSolver::GetLabel(iLabel);
						if (img_row[c] > 0)
							img_labels_row[c] = iLabel;
						else
							img_labels_row[c] = 0;
					}
					else {
						img_labels_row[c] = 0;
					}
				}
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

	void PerformLabelingMem(std::vector<uint64_t>& accesses) {

		{
#undef CONDITION_KB 
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB 

#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

			//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img(s - 1, r - 1, c - 1) > 0
#define CONDITION_LA r > 0 && s > 0 && img(s - 1, r - 1, c) > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img(s - 1, r - 1, c + 1) > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img(s - 1, r - 1, c + 2) > 0
#define CONDITION_NA c > 1 && s > 0 && img(s - 1, r, c - 2) > 0
#define CONDITION_NB c > 0 && s > 0 && img(s - 1, r, c - 1) > 0
#define CONDITION_OA s > 0 && img(s - 1, r, c) > 0
#define CONDITION_OB c < w - 1 && s > 0 && img(s - 1, r, c + 1) > 0
#define CONDITION_PA c < w - 2 && s > 0 && img(s - 1, r, c + 2) > 0
#define CONDITION_PB c < w - 3 && s > 0 && img(s - 1, r, c + 3) > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img(s - 1, r + 1, c - 1) > 0
#define CONDITION_RA r < h - 1 && s > 0 && img(s - 1, r + 1, c) > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 1) > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 2) > 0
#define CONDITION_TB c > 0 && r > 0 && img(s, r - 1, c - 1) > 0
#define CONDITION_UA r > 0 && img(s, r - 1, c) > 0
#define CONDITION_UB c < w - 1 && r > 0 && img(s, r - 1, c + 1) > 0
#define CONDITION_VA c < w - 2 && r > 0 && img(s, r - 1, c + 2) > 0
#define CONDITION_WA c > 1 && img(s, r, c - 2) > 0
#define CONDITION_WB c > 0 && img(s, r, c - 1) > 0
#define CONDITION_XA img(s, r, c) > 0
#define CONDITION_XB c < w - 1 && img(s, r, c + 1) > 0

#include "labeling3D_BBDT_22c_action_definition_memory.inc.h"
		}

		LabelsSolver::MemAlloc(UPPER_BOUND_26_CONNECTIVITY); // Equivalence solver

		MemVol<unsigned char> img(img_);
		MemVol<int> img_labels(img_.size.p);

		LabelsSolver::MemSetup();

		// First scan
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {
			for (unsigned int r = 0; r < h; r += 1) {
				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_BBDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::MemFlatten();

		// NEW VERSION BELOW, OLD IN labeling3D_BBDT_19c.h
		int rows = h;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
				int c = 0;
				for (; c < e_cols; c += 2) {					
					int iLabel = img_labels(s, r, c);
					if (iLabel > 0) {
						iLabel = LabelsSolver::MemGetLabel(iLabel);
						if (img(s, r, c) > 0)
							img_labels(s, r, c) = iLabel;
						else
							img_labels(s, r, c) = 0;
						if (img(s, r, c + 1) > 0)
							img_labels(s, r, c + 1) = iLabel;
						else
							img_labels(s, r, c + 1) = 0;
					}
					else {
						img_labels(s, r, c) = 0;
						img_labels(s, r, c + 1) = 0;
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					int iLabel = img_labels(s, r, c);
					if (iLabel > 0) {	// Useless controls
						iLabel = LabelsSolver::MemGetLabel(iLabel);
						if (img(s, r, c) > 0)
							img_labels(s, r, c) = iLabel;
						else
							img_labels(s, r, c) = 0;
					}
					else {
						img_labels(s, r, c) = 0;
					}
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
#undef CONDITION_KB
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB

#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

			//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LA r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_RA r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_TB c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UA r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UB c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VA c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_NA c > 1 && s > 0 && img_slice11_row00[c - 2] > 0
#define CONDITION_PB c < w - 3 && s > 0 && img_slice11_row00[c + 3] > 0
#define CONDITION_WA c > 1 && img_slice00_row00[c - 2] > 0

//Actions:
#include "labeling3D_BBDT_22c_action_definition.inc.h"
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

		// First scan
		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {
			for (unsigned int r = 0; r < h; r += 1) {

				const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);

				// T, W slice
				const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * -1);

				// K, N, Q slice
				const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
				// T, W slice
				unsigned* const img_labels_slice00_row11 = (unsigned*)(((char*)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

				// K, N, Q slice
				unsigned* const img_labels_slice11_row11 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);

				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_BBDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end
	}

	void SecondScan() {
		// Second scan
		LabelsSolver::Flatten();

		unsigned int d = img_.size.p[0];
		unsigned int h = img_.size.p[1];
		unsigned int w = img_.size.p[2];

		//const unsigned char* const img_row = img_.ptr<unsigned char>();
		//int* const img_labels_row = img_labels_.ptr<int>();

		// NEW VERSION BELOW, OLD COMMENTED IN PerformLabeling
		int rows = h;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
				// Get rows pointer
				const unsigned char* const img_row = img_.ptr<unsigned char>(s, r);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(s, r);
				int c = 0;
				for (; c < e_cols; c += 2) {
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
					}
					else {
						img_labels_row[c] = 0;
						img_labels_row[c + 1] = 0;
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					int iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = LabelsSolver::GetLabel(iLabel);
						if (img_row[c] > 0)
							img_labels_row[c] = iLabel;
						else
							img_labels_row[c] = 0;
					}
					else {
						img_labels_row[c] = 0;
					}
				}
			}
		}
	}
};


#undef CONDITION_KB 
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB 

//Actions:
#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

#endif // YACCLAB_LABELING3D_BBDT_22c_H_