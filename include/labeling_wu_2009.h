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

#ifndef YACCLAB_LABELING_WU_2009_H_
#define YACCLAB_LABELING_WU_2009_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class SAUF : public Labeling {
public:
    SAUF() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(),0); // Allocation + initialization of the output image

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
		LabelsSolver::Setup(); // Labels solver initialization

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
			unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
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
                                img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]);
                            }
                            else {
                                // p = q = 0
                                if (condition_s) {
                                    // x <- merge(s,r)
                                    img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]);
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
                                    // New label
									img_labels_row[c] = LabelsSolver::NewLabel();
                                }
                            }
                        }
                    }
                }
                else {
                    // Nothing to do, x is a background pixel
                }
            }
        }

        // Second scan
		n_labels_ = LabelsSolver::Flatten();

        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = LabelsSolver::GetLabel(*img_row_start);
            }
        }
      
		LabelsSolver::Dealloc(); // Memory deallocation of the labels solver

#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
    }

    void PerformLabelingWithSteps() 
    {
        perf_.start();
		Alloc();
        perf_.stop();
        double alloc_timing = perf_.last();

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

	void PerformLabelingMem(std::vector<unsigned long int>& accesses)
	{
		const int h = img_.rows;
		const int w = img_.cols;

		LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY); // Equivalence resolutor

		// Data structure for memory test
		MemMat<unsigned char> img(img_);
		MemMat<int> img_labels(img_.size(), 0);

		LabelsSolver::MemSetup(); 

		// First scan

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		for (int r = 0; r < h; ++r) {
			for (int c = 0; c < w; ++c) {
#define condition_p c>0 && r>0 && img(r-1 , c-1)>0
#define condition_q r>0 && img(r-1, c)>0
#define condition_r c < w - 1 && r > 0 && img(r-1,c+1)>0
#define condition_s c > 0 && img(r,c-1)>0
#define condition_x img(r,c)>0

				if (condition_x) {
					if (condition_q) {
						// x <- q
						img_labels(r, c) = img_labels(r - 1, c);
					}
					else {
						// q = 0
						if (condition_r) {
							if (condition_p) {
								// x <- merge(p,r)
								img_labels(r, c) = LabelsSolver::MemMerge((unsigned)img_labels(r - 1, c - 1), (unsigned)img_labels(r - 1, c + 1));
							}
							else {
								// p = q = 0
								if (condition_s) {
									// x <- merge(s,r)
									img_labels(r, c) = LabelsSolver::MemMerge((unsigned)img_labels(r, c - 1), (unsigned)img_labels(r - 1, c + 1));
								}
								else {
									// p = q = s = 0
									// x <- r
									img_labels(r, c) = img_labels(r - 1, c + 1);
								}
							}
						}
						else {
							// r = q = 0
							if (condition_p) {
								// x <- p
								img_labels(r, c) = img_labels(r - 1, c - 1);
							}
							else {
								// r = q = p = 0
								if (condition_s) {
									img_labels(r, c) = img_labels(r, c - 1);
								}
								else {
									// New label
									img_labels(r, c) = LabelsSolver::MemNewLabel();
								}
							}
						}
					}
				}
				else {
					// Nothing to do, x is a background pixel
				}
			}
		}

		// Second scan
		n_labels_ = LabelsSolver::MemFlatten();

		for (int r = 0; r < h; ++r) {
			for (int c = 0; c < w; ++c) {
				img_labels(r, c) = LabelsSolver::MemGetLabel(img_labels(r, c));
			}
		}

		// Store total accesses in the output vector 'accesses'
		accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

		accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
		accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
		accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)LabelsSolver::MemTotalAccesses();

		img_labels_ = img_labels.GetImage();

		LabelsSolver::MemDealloc();

#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x
	}

private:
	void Alloc(){
		LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
		img_labels_ = cv::Mat1i(img_.size()); // Memory allocation of the output image
	}
	void Dealloc() {
		LabelsSolver::Dealloc();
		// No free for img_labels_ because it is required at the end of the algorithm 
	}
	void FirstScan() {
	
		const int h = img_.rows;
		const int w = img_.cols;

		img_labels_ = cv::Mat1i::zeros(img_.size()); // Initialization

		LabelsSolver::Setup();

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		// First scan
		for (int r = 0; r < h; ++r) {
			// Get row pointers
			unsigned char const * const img_row = img_.ptr<unsigned char>(r);
			unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
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
						// x <- q
						img_labels_row[c] = img_labels_row_prev[c];
					}
					else {
						// q = 0
						if (condition_r) {
							if (condition_p) {
								// x <- merge(p,r)
								img_labels_row[c] = LabelsSolver::Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]);
							}
							else {
								// p = q = 0
								if (condition_s) {
									// x <- merge(s,r)
									img_labels_row[c] = LabelsSolver::Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]);
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
									// New label
									img_labels_row[c] = LabelsSolver::NewLabel();
								}
							}
						}
					}
				}
				else {
					// Nothing to do, x is a background pixel (already initialized)
				}
			}
		}
	
	}
    void SecondScan() 
    {
		n_labels_ = LabelsSolver::Flatten();

		for (int r = 0; r < img_labels_.rows; ++r) {
			unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
			unsigned * const img_row_end = img_row_start + img_labels_.cols;
			for (; img_row_start != img_row_end; ++img_row_start) {
				*img_row_start = LabelsSolver::GetLabel(*img_row_start);
			}
		}
    }
};

#endif // !YACCLAB_LABELING_WU_2009_H_