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

#ifndef YACCLAB_LABELING_DISTEFANO_H_
#define YACCLAB_LABELING_DISTEFANO_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class DiStefano : public Labeling {
public:
    DiStefano() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size());

        int i_new_label(0);

        // p q r		  p
        // s x			q x
        // lp,lq,lx: labels assigned to p,q,x

        // First scan
        int *a_class= new int[UPPER_BOUND_8_CONNECTIVITY];
        bool *a_single= new bool[UPPER_BOUND_8_CONNECTIVITY];
		int *a_renum = new int[UPPER_BOUND_8_CONNECTIVITY];

        for (int y = 0; y < img_.rows; y++) {

            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(y);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int x = 0; x < img_.cols; x++) {
                if (img_row[x]) {

                    int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
                    if (y > 0) {
                        if (x > 0)
                            lp = img_labels_row_prev[x - 1];
                        lq = img_labels_row_prev[x];
                        if (x < img_.cols - 1)
                            lr = img_labels_row_prev[x + 1];
                    }
                    if (x > 0)
                        ls = img_labels_row[x - 1];

                    // If everything around is background
                    if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
                        lx = ++i_new_label;
                        a_class[lx] = lx;
                        a_single[lx] = true;
                    }
                    else {
                        // p
                        lx = lp;
                        // q
                        if (lx == 0)
                            lx = lq;
                        // r
                        if (lx > 0) {
                            if (lr > 0 && a_class[lx] != a_class[lr]) {
                                if (a_single[a_class[lx]]) {
                                    a_class[lx] = a_class[lr];
                                    a_single[a_class[lr]] = false;
                                }
                                else if (a_single[a_class[lr]]) {
                                    a_class[lr] = a_class[lx];
                                    a_single[a_class[lx]] = false;
                                }
                                else {
                                    int i_class = a_class[lr];
                                    for (int k = 1; k <= i_new_label; k++) {
                                        if (a_class[k] == i_class) {
                                            a_class[k] = a_class[lx];
                                        }
                                    }
                                }
                            }
                        }
                        else
                            lx = lr;
                        // s
                        if (lx > 0) {
                            if (ls > 0 && a_class[lx] != a_class[ls]) {
                                if (a_single[a_class[lx]]) {
                                    a_class[lx] = a_class[ls];
                                    a_single[a_class[ls]] = false;
                                }
                                else if (a_single[a_class[ls]]) {
                                    a_class[ls] = a_class[lx];
                                    a_single[a_class[lx]] = false;
                                }
                                else {
                                    int i_class = a_class[ls];
                                    for (int k = 1; k <= i_new_label; k++) {
                                        if (a_class[k] == i_class) {
                                            a_class[k] = a_class[lx];
                                        }
                                    }
                                }
                            }
                        }
                        else
                            lx = ls;
                    }

                    img_labels_row[x] = lx;
                }
                else
                    img_labels_row[x] = 0;
            }
        }

        // Renumbering of labels
        n_labels_ = 0;
        for (int k = 1; k <= i_new_label; k++) {
            if (a_class[k] == k) {
                n_labels_++;
                a_renum[k] = n_labels_;
            }
        }
        for (int k = 1; k <= i_new_label; k++)
            a_class[k] = a_renum[a_class[k]];

        // Second scan
        for (int y = 0; y < img_labels_.rows; y++) {

            // Get rows pointer
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);

            for (int x = 0; x < img_labels_.cols; x++) {
                int iLabel = img_labels_row[x];
                if (iLabel > 0)
                    img_labels_row[x] = a_class[iLabel];
            }
        }

        n_labels_++; // To count also background

        delete[] a_class;
        delete[] a_single;
        delete[] a_renum;
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

    void PerformLabelingMem(std::vector<unsigned long int>& accesses)
    {
        MemMat<uchar> img(img_);
        MemMat<int> img_labels(img_.size());

        int i_new_label(0);
        // p q r		  p
        // s x			q x
        // lp,lq,lx: labels assigned to p,q,x
        
        // First scan:
        MemVector<int> a_class(UPPER_BOUND_8_CONNECTIVITY);
        MemVector<char> a_single(UPPER_BOUND_8_CONNECTIVITY);
		MemVector<int> a_renum(UPPER_BOUND_8_CONNECTIVITY);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (img(y, x)) {

                    int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
                    if (y > 0) {
                        if (x > 0)
                            lp = img_labels(y - 1, x - 1);
                        lq = img_labels(y - 1, x);
                        if (x < img.cols - 1)
                            lr = img_labels(y - 1, x + 1);
                    }
                    if (x > 0)
                        ls = img_labels(y, x - 1);

                    // If everything around is background
                    if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
                        lx = ++i_new_label;
                        a_class[lx] = lx;
                        a_single[lx] = true;
                    }
                    else {
                        // p
                        lx = lp;
                        // q
                        if (lx == 0)
                            lx = lq;
                        // r
                        if (lx > 0) {
                            if (lr > 0 && a_class[lx] != a_class[lr]) {
                                if (a_single[a_class[lx]]) {
                                    a_class[lx] = a_class[lr];
                                    a_single[a_class[lr]] = false;
                                }
                                else if (a_single[a_class[lr]]) {
                                    a_class[lr] = a_class[lx];
                                    a_single[a_class[lx]] = false;
                                }
                                else {
                                    int i_class = a_class[lr];
                                    for (int k = 1; k <= i_new_label; k++) {
                                        if (a_class[k] == i_class) {
                                            a_class[k] = a_class[lx];
                                        }
                                    }
                                }
                            }
                        }
                        else
                            lx = lr;
                        // s
                        if (lx > 0) {
                            if (ls > 0 && a_class[lx] != a_class[ls]) {
                                if (a_single[a_class[lx]]) {
                                    a_class[lx] = a_class[ls];
                                    a_single[a_class[ls]] = false;
                                }
                                else if (a_single[a_class[ls]]) {
                                    a_class[ls] = a_class[lx];
                                    a_single[a_class[lx]] = false;
                                }
                                else {
                                    int i_class = a_class[ls];
                                    for (int k = 1; k <= i_new_label; k++) {
                                        if (a_class[k] == i_class) {
                                            a_class[k] = a_class[lx];
                                        }
                                    }
                                }
                            }
                        }
                        else
                            lx = ls;
                    }

                    img_labels(y, x) = lx;
                }
                else
                    img_labels(y, x) = 0;
            }
        }

        // Renumbering of labels
        n_labels_ = 0;
        for (int k = 1; k <= i_new_label; k++) {
            if (a_class[k] == k) {
                n_labels_++;
                a_renum[k] = n_labels_;
            }
        }
        for (int k = 1; k <= i_new_label; k++)
            a_class[k] = a_renum[a_class[k]];

        // Second scan
        for (int y = 0; y < img_labels.rows; y++) {
            for (int x = 0; x < img_labels.cols; x++) {
                int i_label = img_labels(y, x);
                if (i_label > 0)
                    img_labels(y, x) = a_class[i_label];
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)(a_class.GetTotalAccesses() + a_single.GetTotalAccesses() + a_renum.GetTotalAccesses());

        img_labels_ = img_labels.GetImage(); 

        n_labels_++;
    }

private:
	int *a_class;
	bool *a_single;
	int *a_renum;
	int i_new_label;

    double Alloc()
    {
        // Memory allocation for the output image
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        a_class = new int[UPPER_BOUND_8_CONNECTIVITY];
        a_single = new bool[UPPER_BOUND_8_CONNECTIVITY];
        a_renum = new int[UPPER_BOUND_8_CONNECTIVITY];
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        memset(a_class, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(int));
        memset(a_single, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(bool));
        memset(a_renum, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(int));
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        memset(a_class, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(int));
        memset(a_single, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(bool));
        memset(a_renum, 0, UPPER_BOUND_8_CONNECTIVITY*sizeof(int));
        perf_.stop();
        double ma_t = t - perf_.last();
        // Return total time
        return ma_t;
    }
    void Dealloc()
	{
		delete[] a_class;
		delete[] a_single;
		delete[] a_renum;
		// No free for img_labels_ because it is required at the end of the algorithm 
	}
	void FirstScan()
	{
		i_new_label = 0;

		for (int y = 0; y < img_.rows; y++) {

			// Get rows pointer
			const unsigned char* const img_row = img_.ptr<unsigned char>(y);
			unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);
			unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);

			for (int x = 0; x < img_.cols; x++) {
				if (img_row[x]) {

					int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
					if (y > 0) {
						if (x > 0)
							lp = img_labels_row_prev[x - 1];
						lq = img_labels_row_prev[x];
						if (x < img_.cols - 1)
							lr = img_labels_row_prev[x + 1];
					}
					if (x > 0)
						ls = img_labels_row[x - 1];

					// If everything around is background
					if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
						lx = ++i_new_label;
						a_class[lx] = lx;
						a_single[lx] = true;
					}
					else {
						// p
						lx = lp;
						// q
						if (lx == 0)
							lx = lq;
						// r
						if (lx > 0) {
							if (lr > 0 && a_class[lx] != a_class[lr]) {
								if (a_single[a_class[lx]]) {
									a_class[lx] = a_class[lr];
									a_single[a_class[lr]] = false;
								}
								else if (a_single[a_class[lr]]) {
									a_class[lr] = a_class[lx];
									a_single[a_class[lx]] = false;
								}
								else {
									int i_class = a_class[lr];
									for (int k = 1; k <= i_new_label; k++) {
										if (a_class[k] == i_class) {
											a_class[k] = a_class[lx];
										}
									}
								}
							}
						}
						else
							lx = lr;
						// s
						if (lx > 0) {
							if (ls > 0 && a_class[lx] != a_class[ls]) {
								if (a_single[a_class[lx]]) {
									a_class[lx] = a_class[ls];
									a_single[a_class[ls]] = false;
								}
								else if (a_single[a_class[ls]]) {
									a_class[ls] = a_class[lx];
									a_single[a_class[lx]] = false;
								}
								else {
									int i_class = a_class[ls];
									for (int k = 1; k <= i_new_label; k++) {
										if (a_class[k] == i_class) {
											a_class[k] = a_class[lx];
										}
									}
								}
							}
						}
						else
							lx = ls;
					}

					img_labels_row[x] = lx;
				}
				else
					img_labels_row[x] = 0;
			}
		}
	}

	void SecondScan()
	{
		// Renumbering of labels
		n_labels_ = 0;
		for (int k = 1; k <= i_new_label; k++) {
			if (a_class[k] == k) {
				n_labels_++;
				a_renum[k] = n_labels_;
			}
		}
		for (int k = 1; k <= i_new_label; k++)
			a_class[k] = a_renum[a_class[k]];

		// Second scan
		for (int y = 0; y < img_labels_.rows; y++) {

			// Get rows pointer
			unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(y);

			for (int x = 0; x < img_labels_.cols; x++) {
				int iLabel = img_labels_row[x];
				if (iLabel > 0)
					img_labels_row[x] = a_class[iLabel];
			}
		}

		n_labels_++; // To count also background
	}
};

#endif // !YACCLAB_LABELING_DISTEFANO_H_

