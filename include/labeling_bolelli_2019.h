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

#ifndef YACCLAB_LABELING_BOLELLI_2019_H_
#define YACCLAB_LABELING_BOLELLI_2019_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class Spaghetti : public Labeling2D<CONN_8> {
public:
	Spaghetti() {}

	void PerformLabeling()
	{
		const int h = img_.rows;
		const int w = img_.cols;

		int e_rows = h & 0xfffffffe;
		bool o_rows = h % 2 == 1;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		img_labels_ = cv::Mat1i(img_.size()); // Memory allocation for the output image

		LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

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
							   // the connectivity between these blocks and those pixels only matter
							   // when considering the outer connectivities

							   // A bunch of defines used to check if the pixels are foreground,
							   // without going outside the image limits.

							   // First scan

		// Define Conditions and Actions
		{
#define CONDITION_B img_row_prev_prev[c-1]>0
#define CONDITION_C img_row_prev_prev[c]>0
#define CONDITION_D img_row_prev_prev[c+1]>0
#define CONDITION_E img_row_prev_prev[c+2]>0

#define CONDITION_G img_row_prev[c-2]>0
#define CONDITION_H img_row_prev[c-1]>0
#define CONDITION_I img_row_prev[c]>0
#define CONDITION_J img_row_prev[c+1]>0
#define CONDITION_K img_row_prev[c+2]>0

#define CONDITION_M img_row[c-2]>0
#define CONDITION_N img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P img_row[c+1]>0

#define CONDITION_R img_row_fol[c-1]>0
#define CONDITION_S img_row_fol[c]>0
#define CONDITION_T img_row_fol[c+1]>0

			// Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; 
							   // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); 
							   //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2];
							   // Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c];
							   // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2];
							   // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; 
							   // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
							   //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
							   // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
							   // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
							   // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]);
							   // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
							   // Action 13: Merge labels of block P, Q and R
#define ACTION_13 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row_prev_prev[c + 2]);
							   // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
							   //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
							   //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
		}

		if (h == 1) {
			// Single line
			int r = 0;
			const unsigned char* const img_row = img_.ptr<unsigned char>(0);
			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(0);
			int c = -2;
#include "labeling_bolelli_2019_forest_singleline.inc.h"
		}
		else {
			// More than one line

			// First couple of lines
			{
				int r = 0;
				const unsigned char* const img_row = img_.ptr<unsigned char>(0);
				const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(0);
				int c = -2;

#include "labeling_bolelli_2019_forest_firstline.inc.h"
			}

			// Every other line but the last one if image has an odd number of rows
			for (int r = 2; r < e_rows; r += 2) {
				// Get rows pointer
				const unsigned char* const img_row = img_.ptr<unsigned char>(r);
				const unsigned char* const img_row_prev = (unsigned char*)(((char*)img_row) - img_.step.p[0]);
				const unsigned char* const img_row_prev_prev = (unsigned char*)(((char*)img_row_prev) - img_.step.p[0]);
				const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
				unsigned* const img_labels_row_prev_prev = (unsigned*)(((char*)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

				int c = -2;
				goto tree_0;

#include "labeling_bolelli_2019_forest.inc.h"
			}

			// Last line (in case the rows are odd)
			if (o_rows)
			{
				int r = h - 1;
				const unsigned char* const img_row = img_.ptr<unsigned char>(r);
				const unsigned char* const img_row_prev = (unsigned char*)(((char*)img_row) - img_.step.p[0]);
				const unsigned char* const img_row_prev_prev = (unsigned char*)(((char*)img_row_prev) - img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
				unsigned* const img_labels_row_prev_prev = (unsigned*)(((char*)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
				int c = -2;
#include "labeling_bolelli_2019_forest_lastline.inc.h"
			}
		}


		// Undef Conditions and Actions
		{
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


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
		}

		// Second scan
		n_labels_ = LabelsSolver::Flatten();

		int r = 0;
		for (; r < e_rows; r += 2) {
			// Get rows pointer
			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
			const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);

			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
			unsigned* const img_labels_row_fol = (unsigned*)(((char*)img_labels_row) + img_labels_.step.p[0]);
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
					if (img_row_fol[c] > 0)
						img_labels_row_fol[c] = iLabel;
					else
						img_labels_row_fol[c] = 0;
					if (img_row_fol[c + 1] > 0)
						img_labels_row_fol[c + 1] = iLabel;
					else
						img_labels_row_fol[c + 1] = 0;
				}
				else {
					img_labels_row[c] = 0;
					img_labels_row[c + 1] = 0;
					img_labels_row_fol[c] = 0;
					img_labels_row_fol[c + 1] = 0;
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
					if (img_row_fol[c] > 0)
						img_labels_row_fol[c] = iLabel;
					else
						img_labels_row_fol[c] = 0;
				}
				else {
					img_labels_row[c] = 0;
					img_labels_row_fol[c] = 0;
				}
			}
		}
		// Last row if the number of rows is odd
		if (o_rows) {
			// Get rows pointer
			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
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

		LabelsSolver::Dealloc();

	}

	void PerformLabelingWithSteps()
	{
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
		const int h = img_.rows;
		const int w = img_.cols;

		LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
		LabelsSolver::MemSetup();

		//Data structure for memory test
		MemMat<unsigned char> img(img_);
		MemMat<int> img_labels(img_.size(), 0);

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

#define CONDITION_B img(r-2, c-1)>0
#define CONDITION_C img(r-2, c)>0
#define CONDITION_D img(r-2, c+1)>0
#define CONDITION_E img(r-2, c+2)>0

#define CONDITION_G img(r-1, c-2)>0
#define CONDITION_H img(r-1, c-1)>0
#define CONDITION_I img(r-1, c)>0
#define CONDITION_J img(r-1, c+1)>0
#define CONDITION_K img(r-1, c+2)>0

#define CONDITION_M img(r, c-2)>0
#define CONDITION_N img(r, c-1)>0
#define CONDITION_O img(r,c)>0
#define CONDITION_P img(r,c+1)>0

#define CONDITION_R img(r+1, c-1)>0
#define CONDITION_S img(r+1, c)>0
#define CONDITION_T img(r+1, c+1)>0

				// Action 1: No action
#define ACTION_1 img_labels(r, c) = 0; 
				// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels(r, c) = LabelsSolver::MemNewLabel(); 
				//Action 3: Assign label of block P
#define ACTION_3 img_labels(r, c) = img_labels(r - 2, c - 2);
				// Action 4: Assign label of block Q 
#define ACTION_4 img_labels(r, c) = img_labels(r - 2, c);
				// Action 5: Assign label of block R
#define ACTION_5 img_labels(r, c) = img_labels(r - 2, c + 2);
				// Action 6: Assign label of block S
#define ACTION_6 img_labels(r, c) = img_labels(r, c - 2); 
				// Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c));
				//Action 8: Merge labels of block P and R
#define ACTION_8 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2));
				// Action 9 Merge labels of block P and S
#define ACTION_9 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r, c - 2));
				// Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2));
				// Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r, c - 2));
				// Action 12: Merge labels of block R and S
#define ACTION_12 img_labels(r, c) = LabelsSolver::MemMerge(img_labels(r - 2, c + 2), img_labels(r, c - 2));
				// Action 13: not used
#define ACTION_13 
				// Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c)), img_labels(r, c - 2));
				//Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2)), img_labels(r, c - 2));
				//Action 16: labels of block Q, R and S
#define ACTION_16 img_labels(r, c) = LabelsSolver::MemMerge(LabelsSolver::MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2)), img_labels(r, c - 2));


		if (h == 1) {
			// Single line
			int r = 0;
			int c = -2;
#include "labeling_bolelli_2019_forest_singleline.inc.h"
		}
		else {
			// More than one line

			// First couple of lines
			{
				int r = 0;
				int c = -2;
#include "labeling_bolelli_2019_forest_firstline.inc.h"
			}

			// Every other line but the last one if image has an odd number of rows
			for (int r = 2; r < h - (h % 2); r += 2) {
				int c = -2;
				goto tree_0;
#include "labeling_bolelli_2019_forest.inc.h"
			}

			// Last line (in case that the rows are odd)
			if ((h % 2) == 1)
			{
				int r = h - 1;
				int c = -2;
#include "labeling_bolelli_2019_forest_lastline.inc.h"
			}
		}

		n_labels_ = LabelsSolver::MemFlatten();

		// Second scan
		for (int r = 0; r < h; r += 2) {
			for (int c = 0; c < w; c += 2) {
				int iLabel = img_labels(r, c);
				if (iLabel > 0) {
					iLabel = LabelsSolver::MemGetLabel(iLabel);
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
		accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

		accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
		accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
		accesses[MD_EQUIVALENCE_VEC] = (uint64_t)LabelsSolver::MemTotalAccesses();

		img_labels_ = img_labels.GetImage();

		LabelsSolver::MemDealloc();

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


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
	}

private:

	int e_rows;
	bool o_rows;
	int e_cols;
	bool o_cols;

	double Alloc()
	{
		// Memory allocation of the labels solver
		double ls_t = LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);
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
		return ls_t + ma_t;
	}

	void Dealloc()
	{
		LabelsSolver::Dealloc();
		// No free for img_labels_ because it is required at the end of the algorithm 
	}

	void FirstScan()
	{
		const int h = img_.rows;
		const int w = img_.cols;

		e_rows = h & 0xfffffffe;
		o_rows = h % 2 == 1;
		e_cols = w & 0xfffffffe;
		o_cols = w % 2 == 1;

		LabelsSolver::Setup(); // Labels solver initialization

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
							   // the connectivity between these blocks and those pixels only matter
							   // when considering the outer connectivities

							   // A bunch of defines used to check if the pixels are foreground,
							   // without going outside the image limits.

							   // First scan

		// Define Conditions and Actions
		{
#define CONDITION_B img_row_prev_prev[c-1]>0
#define CONDITION_C img_row_prev_prev[c]>0
#define CONDITION_D img_row_prev_prev[c+1]>0
#define CONDITION_E img_row_prev_prev[c+2]>0

#define CONDITION_G img_row_prev[c-2]>0
#define CONDITION_H img_row_prev[c-1]>0
#define CONDITION_I img_row_prev[c]>0
#define CONDITION_J img_row_prev[c+1]>0
#define CONDITION_K img_row_prev[c+2]>0

#define CONDITION_M img_row[c-2]>0
#define CONDITION_N img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P img_row[c+1]>0

#define CONDITION_R img_row_fol[c-1]>0
#define CONDITION_S img_row_fol[c]>0
#define CONDITION_T img_row_fol[c+1]>0

			// Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; 
							   // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); 
							   //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2];
							   // Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c];
							   // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2];
							   // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; 
							   // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
							   //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
							   // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
							   // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
							   // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]);
							   // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
							   // Action 13: Merge labels of block P, Q and R
#define ACTION_13 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row_prev_prev[c + 2]);
							   // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
							   //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
							   //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
		}

		if (h == 1) {
			// Single line
			int r = 0;
			const unsigned char* const img_row = img_.ptr<unsigned char>(0);
			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(0);
			int c = -2;
#include "labeling_bolelli_2019_forest_singleline.inc.h"
		}
		else {
			// More than one line

			// First couple of lines
			{
				int r = 0;
				const unsigned char* const img_row = img_.ptr<unsigned char>(0);
				const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(0);
				int c = -2;

#include "labeling_bolelli_2019_forest_firstline.inc.h"
			}

			// Every other line but the last one if image has an odd number of rows
			for (int r = 2; r < e_rows; r += 2) {
				// Get rows pointer
				const unsigned char* const img_row = img_.ptr<unsigned char>(r);
				const unsigned char* const img_row_prev = (unsigned char*)(((char*)img_row) - img_.step.p[0]);
				const unsigned char* const img_row_prev_prev = (unsigned char*)(((char*)img_row_prev) - img_.step.p[0]);
				const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
				unsigned* const img_labels_row_prev_prev = (unsigned*)(((char*)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

				int c = -2;
				goto tree_0;

#include "labeling_bolelli_2019_forest.inc.h"
			}

			// Last line (in case the rows are odd)
			if (o_rows)
			{
				int r = h - 1;
				const unsigned char* const img_row = img_.ptr<unsigned char>(r);
				const unsigned char* const img_row_prev = (unsigned char*)(((char*)img_row) - img_.step.p[0]);
				const unsigned char* const img_row_prev_prev = (unsigned char*)(((char*)img_row_prev) - img_.step.p[0]);
				unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
				unsigned* const img_labels_row_prev_prev = (unsigned*)(((char*)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
				int c = -2;
#include "labeling_bolelli_2019_forest_lastline.inc.h"
			}
		}


		// Undef Conditions and Actions
		{
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


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
		}
	}

	void SecondScan()
	{   // Second scan
		n_labels_ = LabelsSolver::Flatten();

		int r = 0;
		for (; r < e_rows; r += 2) {
			// Get rows pointer
			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
			const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[0]);

			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
			unsigned* const img_labels_row_fol = (unsigned*)(((char*)img_labels_row) + img_labels_.step.p[0]);
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
					if (img_row_fol[c] > 0)
						img_labels_row_fol[c] = iLabel;
					else
						img_labels_row_fol[c] = 0;
					if (img_row_fol[c + 1] > 0)
						img_labels_row_fol[c + 1] = iLabel;
					else
						img_labels_row_fol[c + 1] = 0;
				}
				else {
					img_labels_row[c] = 0;
					img_labels_row[c + 1] = 0;
					img_labels_row_fol[c] = 0;
					img_labels_row_fol[c + 1] = 0;
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
					if (img_row_fol[c] > 0)
						img_labels_row_fol[c] = iLabel;
					else
						img_labels_row_fol[c] = 0;
				}
				else {
					img_labels_row[c] = 0;
					img_labels_row_fol[c] = 0;
				}
			}
		}
		// Last row if the number of rows is odd
		if (o_rows) {
			// Get rows pointer
			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
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
};

#endif // YACCLAB_LABELING_BOLELLI_2019_H_