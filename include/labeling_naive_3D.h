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

#ifndef YACCLAB_LABELING_NAIVE_3D_H_
#define YACCLAB_LABELING_NAIVE_3D_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class naive_3D : public Labeling3D<CONN_26> {
public:
	naive_3D() {}

	void PerformLabeling()
	{
		//img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image
		img_labels_.create(3, img_.size.p, CV_32SC1);

		LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		// First scan
		for (int z = 0; z < img_.size[0]; z++) {

			unsigned char const * const img_plane = img_.data + img_.step[0] * z;   //   img_.ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(img_labels_.data) + (img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + img_.step[1] * y;
					img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					img_prev_plane_rows[2] = (y + 1 < img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (img_labels_.step[1] / sizeof(int)) * y;
				labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (img_labels_.step[1] / sizeof(int));
				labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < img_.size[2]; x++) {
					int label = 0;
					if (img_row[x] > 0) {

						int const first_neighbour_x = (x > 0) ? (x - 1) : x;
						int const last_neighbour_x = (x + 1 < img_.size[2]) ? (x + 1) : x;

						// Previous plane
						if (img_prev_plane != nullptr) {
							for (int r = prev_plane_first_row; r <= prev_plane_last_row; r++) {
								for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
									if (img_prev_plane_rows[r][c] > 0) {
										if (label == 0) {
											label = labels_prev_plane_rows[r][c];
										}
										else {
											LabelsSolver::Merge(labels_prev_plane_rows[r][c], label);
										}
									}
								}
							}
						}

						// Previous row
						if (img_prev_row != nullptr) {
							for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
								if (img_prev_row[c] > 0) {
									if (label == 0) {
										label = labels_prev_row[c];
									}
									else {
										LabelsSolver::Merge(labels_prev_row[c], label);
									}
								}
							}
						}

						// Previous col
						if (x > 0) {
							if (img_row[x - 1] > 0) {
								if (label == 0) {
									label = labels_row[x - 1];
								}
								else {
									LabelsSolver::Merge(labels_row[x - 1], label);
								}
							}
						}

						if (label == 0) {
							label = LabelsSolver::NewLabel();
						}
					}
					labels_row[x] = label;
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (int z = 0; z < img_labels_.size[0]; z++) {
			for (int y = 0; y < img_labels_.size[1]; y++) {
				for (int x = 0; x < img_labels_.size[2]; x++) {
					img_row[x] = LabelsSolver::GetLabel(img_row[x]);
				}
				img_row += img_labels_.step[1] / sizeof(int);
			}
		}

		LabelsSolver::Dealloc(); // Memory deallocation of the labels solver

	}
	
	void PerformLabelingWithSteps()	{
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

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		// First scan
		for (int z = 0; z < img_.size[0]; z++) {

			unsigned char const * const img_plane = img_.data + img_.step[0] * z;   //   img_.ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(img_labels_.data) + (img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + img_.step[1] * y;
					img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					img_prev_plane_rows[2] = (y + 1 < img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (img_labels_.step[1] / sizeof(int)) * y;
				labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (img_labels_.step[1] / sizeof(int));
				labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < img_.size[2]; x++) {
					int label = 0;
					if (img_row[x] > 0) {

						int const first_neighbour_x = (x > 0) ? (x - 1) : x;
						int const last_neighbour_x = (x + 1 < img_.size[2]) ? (x + 1) : x;

						// Previous plane
						if (img_prev_plane != nullptr) {
							for (int r = prev_plane_first_row; r <= prev_plane_last_row; r++) {
								for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
									if (img_prev_plane_rows[r][c] > 0) {
										if (label == 0) {
											label = labels_prev_plane_rows[r][c];
										}
										else {
											LabelsSolver::Merge(labels_prev_plane_rows[r][c], label);
										}
									}
								}
							}
						}

						// Previous row
						if (img_prev_row != nullptr) {
							for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
								if (img_prev_row[c] > 0) {
									if (label == 0) {
										label = labels_prev_row[c];
									}
									else {
										LabelsSolver::Merge(labels_prev_row[c], label);
									}
								}
							}
						}

						// Previous col
						if (x > 0) {
							if (img_row[x - 1] > 0) {
								if (label == 0) {
									label = labels_row[x - 1];
								}
								else {
									LabelsSolver::Merge(labels_row[x - 1], label);
								}
							}
						}

						if (label == 0) {
							label = LabelsSolver::NewLabel();
						}
					}
					labels_row[x] = label;
				}
			} // Rows cycle end
		} // Planes cycle end
	}

	void SecondScan() {
		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (int z = 0; z < img_labels_.size[0]; z++) {
			for (int y = 0; y < img_labels_.size[1]; y++) {
				for (int x = 0; x < img_labels_.size[2]; x++) {
					img_row[x] = LabelsSolver::GetLabel(img_row[x]);
				}
				img_row += img_labels_.step[1] / sizeof(int);
			}
		}
	}
};

#endif // !YACCLAB_LABELING_NAIVE_3D_H_