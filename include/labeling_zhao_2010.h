// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_ZHAO_2010_H_
#define YACCLAB_LABELING_ZHAO_2010_H_

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class SBLA : public Labeling2D<Connectivity2D::CONN_8> {
public:
    SBLA() {}

    void PerformLabeling()
    {
        int N = img_.cols;
        int M = img_.rows;
        img_labels_ = cv::Mat1i(M, N);

        // Fix for first pixel!
        n_labels_ = 0;
        uchar firstpixel = img_.data[0];
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_).data[0] = 0;

			if (M == 1) {
				if (N == 1) 
					n_labels_ = 1;
				else 
					n_labels_ = img_.data[1] == 0;
			}
			else {
				if (N == 1) 
					n_labels_ = img_.data[img_.step[0]] == 0;
				else 
					n_labels_ = img_.data[1] == 0 && img_.data[img_.step[0]] == 0 && img_.data[img_.step[0] + 1] == 0;
			}
        }

        // Stripe extraction and representation
        unsigned int* img_labels_row_prev = nullptr;
        int rN = 0;
        int r1N = N;
        int N2 = N * 2;

		int e_rows = img_labels_.rows & 0xfffffffe;
		bool o_rows = img_labels_.rows % 2 == 1;

		int r = 0;
		for (; r < e_rows; r += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            for (int c = 0; c < N; ++c) {
                img_labels_row[c] = img_row[c];
                img_labels_row_fol[c] = img_row_fol[c];
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels_row[c];
                int oddpix = img_labels_row_fol[c];

                // Step 2
                int Gp;
                if (oddpix) {
                    img_labels_row_fol[c] = Gp = -(r1N + c);
                    if (evenpix)
                        img_labels_row[c] = Gp;
                }
                else if (evenpix)
                    img_labels_row[c] = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels_row[c];
                    int oddpix = img_labels_row_fol[c];

                    if (oddpix) {
                        img_labels_row_fol[c] = Gp;
                        if (evenpix)
                            img_labels_row[c] = Gp;
                    }
                    else if (evenpix)
                        img_labels_row[c] = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels_row[i];
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels_row[i]);
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels_row_prev[j];
                        if (!curpix)
                            continue;

                        int newroot = FindRoot(reinterpret_cast<int*>(img_labels_.data), curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels_row_prev[j]);
                    }
                }
            }
            img_labels_row_prev = img_labels_row_fol;
            rN += N2;
            r1N += N2;
        }
		// Last row if the number of rows is odd
		if (o_rows) {
			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
			unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
			for (int c = 0; c < N; ++c) {
				img_labels_row[c] = img_row[c];
			}

			for (int c = 0; c < N; ++c) {
				// Step 1
				int evenpix = img_labels_row[c];

				// Step 2
				int Gp;
				if (evenpix)
					img_labels_row[c] = Gp = -(rN + c);
				else
					continue;

				// Step 3
				int stripestart = c;
				while (++c < N) {
					int evenpix = img_labels_row[c];

					if (evenpix)
						img_labels_row[c] = Gp;
					else
						break;
				}
				int stripestop = c;

				if (r == 0)
					continue;

				// Stripe union
				int lastroot = INT_MIN;
				for (int i = stripestart; i < stripestop; ++i) {
					int linepix = img_labels_row[i];
					if (!linepix)
						continue;

					int runstart = std::max(0, i - 1);
					do
						i++;
					while (i < N && img_labels_row[i]);
					int runstop = std::min(N - 1, i);

					for (int j = runstart; j <= runstop; ++j) {
						int curpix = img_labels_row_prev[j];
						if (!curpix)
							continue;

						int newroot = FindRoot(reinterpret_cast<int*>(img_labels_.data), curpix);
						if (newroot > lastroot) {
							lastroot = newroot;
							FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), Gp, lastroot);
						}
						else if (newroot < lastroot) {
							FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), newroot, lastroot);
						}

						do
							++j;
						while (j <= runstop && img_labels_row_prev[j]);
					}
				}
			}
		}

        // Label assignment
        int *img_labelsdata = reinterpret_cast<int*>(img_labels_.data);
        for (int i = 0; i < M * N; ++i) {
            // FindRoot_GetLabel
            int pos = img_labelsdata[i];
            if (pos >= 0)
                continue;

            int tmp;
            while (true) {
                tmp = img_labelsdata[-pos];
                if (pos == tmp || tmp >= 0)
                    break;
                pos = tmp;
            }
            if (tmp < 0)
                img_labelsdata[-pos] = ++n_labels_;

            // Assign final label
            img_labelsdata[i] = img_labelsdata[-pos];
        }

        // Fix for first pixel!
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_).data[0] = firstpixel;
            if (N > 1 && img_labels_.data[1])
                img_labels_.data[0] = img_labels_.data[1];
            else if (M > 1 && img_labels_.data[img_.step[0]])
                img_labels_.data[0] = img_labels_.data[img_.step[0]];
			else if (N > 1 && M > 1 && img_labels_.data[img_.step[0] + 1])
                img_labels_.data[0] = img_labels_.data[img_.step[0] + 1];
            else
                img_labels_.data[0] = 1;
        }

        n_labels_++; // To count also background
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

        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size());

        int N = img_.cols;
        int M = img_.rows;

        // Fix for first pixel!
        n_labels_ = 0;
        uchar firstpixel = img(0, 0);
        if (firstpixel) {
            img(0, 0) = 0;

            if (M == 1) {
                if (N == 1)
                    n_labels_ = 1;
                else
                    n_labels_ = img(0, 1) == 0;
            }
            else {
                if (N == 1)
                    n_labels_ = img(1, 0) == 0;
                else
                    n_labels_ = img(0, 1) == 0 && img(1, 0) == 0 && img(1, 1) == 0;
            }
        }

        // Stripe extraction and representation
        int rN = 0;
        int r1N = N;
        int N2 = N * 2;

        int e_rows = img_labels.rows & 0xfffffffe;
        bool o_rows = img_labels.rows % 2 == 1;

        int r = 0;
        for (; r < e_rows; r += 2) {
            for (int c = 0; c < N; ++c) {
                img_labels(r, c) = img(r, c);
                img_labels(r + 1, c) = img(r + 1, c);
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels(r, c);
                int oddpix = img_labels(r + 1, c);

                // Step 2
                int Gp;
                if (oddpix) {
                    img_labels(r + 1, c) = Gp = -(r1N + c);
                    if (evenpix)
                        img_labels(r, c) = Gp;
                }
                else if (evenpix)
                    img_labels(r, c) = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels(r, c);
                    int oddpix = img_labels(r + 1, c);

                    if (oddpix) {
                        img_labels(r + 1, c) = Gp;
                        if (evenpix)
                            img_labels(r, c) = Gp;
                    }
                    else if (evenpix)
                        img_labels(r, c) = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels(r, i);
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels(r, i));
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels(r - 1, j);
                        if (!curpix)
                            continue;

                        int newroot = FindRootMem(img_labels, curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompressMem(img_labels, Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompressMem(img_labels, newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels(r - 1, j));
                    }
                }
            }
            rN += N2;
            r1N += N2;
        }
        // Last row if the number of rows is odd
        if (o_rows) {
            for (int c = 0; c < N; ++c) {
                img_labels(r, c) = img(r, c);
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels(r, c);

                // Step 2
                int Gp;
                if (evenpix)
                    img_labels(r, c) = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels(r, c);

                    if (evenpix)
                        img_labels(r, c) = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels(r, i);
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels(r, i));
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels(r - 1, j);
                        if (!curpix)
                            continue;

                        int newroot = FindRootMem(img_labels, curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompressMem(img_labels, Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompressMem(img_labels, newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels(r -1, j));
                    }
                }
            }
        }

        // Label assignment
        for (int i = 0; i < M * N; ++i) {
            // FindRoot_GetLabel
            int pos = img_labels(i);
            if (pos >= 0)
                continue;

            int tmp;
            while (true) {
                tmp = img_labels(-pos);
                if (pos == tmp || tmp >= 0)
                    break;
                pos = tmp;
            }
            if (tmp < 0)
                img_labels(-pos) = ++n_labels_;

            // Assign final label
            img_labels(i) = img_labels(-pos);
        }

        // Fix for first pixel!
        if (firstpixel) {
            img(0, 0) = firstpixel;
            if (N > 1 && img_labels(0, 1))
                img_labels(0, 0) = img_labels(0, 1);
            else if (M > 1 && img_labels(1, 0))
                img_labels(0, 0) = img_labels(1, 0);
            else if (N > 1 && M > 1 && img_labels(1, 1))
                img_labels(0, 0) = img_labels(1, 1);
            else
                img_labels(0, 0) = 1;
        }

        n_labels_++; // To count also background

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();

        img_labels_ = img_labels.GetImage();
    }

private:
    inline int FindRoot(int *img_labels, int pos)
    {
        while (true) {
            int tmppos = img_labels[-pos];
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
        return pos;
    }

    inline void FindRootAndCompress(int *img_labels, int pos, int newroot)
    {
        while (true) {
            int tmppos = img_labels[-pos];
            if (tmppos == newroot)
                break;
            img_labels[-pos] = newroot;
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
    }

    inline int FindRootMem(MemMat<int>& img_labels, int pos)
    {
        while (true) {
            int tmppos = img_labels(-pos);
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
        return pos;
    }

    inline void FindRootAndCompressMem(MemMat<int>& img_labels, int pos, int newroot)
    {
        while (true) {
            int tmppos = img_labels(-pos);
            if (tmppos == newroot)
                break;
            img_labels(-pos) = newroot;
            if (tmppos == pos)
                break;
            pos = tmppos;
        }
    }

    unsigned char firstpixel;
    int N;
    int M;

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
    }
    void FirstScan()
    {
        N = img_.cols;
        M = img_.rows;
        img_labels_ = cv::Mat1i(M, N);

        // Fix for first pixel!
        n_labels_ = 0;
        firstpixel = img_.data[0];
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_).data[0] = 0;

            if (M == 1) {
                if (N == 1)
                    n_labels_ = 1;
                else
                    n_labels_ = img_.data[1] == 0;
            }
            else {
                if (N == 1)
                    n_labels_ = img_.data[img_.step[0]] == 0;
                else
                    n_labels_ = img_.data[1] == 0 && img_.data[img_.step[0]] == 0 && img_.data[img_.step[0] + 1] == 0;
            }
        }

        // Stripe extraction and representation
        unsigned int* img_labels_row_prev = nullptr;
        int rN = 0;
        int r1N = N;
        int N2 = N * 2;

        int e_rows = img_labels_.rows & 0xfffffffe;
        bool o_rows = img_labels_.rows % 2 == 1;

        int r = 0;
        for (; r < e_rows; r += 2) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            for (int c = 0; c < N; ++c) {
                img_labels_row[c] = img_row[c];
                img_labels_row_fol[c] = img_row_fol[c];
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels_row[c];
                int oddpix = img_labels_row_fol[c];

                // Step 2
                int Gp;
                if (oddpix) {
                    img_labels_row_fol[c] = Gp = -(r1N + c);
                    if (evenpix)
                        img_labels_row[c] = Gp;
                }
                else if (evenpix)
                    img_labels_row[c] = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels_row[c];
                    int oddpix = img_labels_row_fol[c];

                    if (oddpix) {
                        img_labels_row_fol[c] = Gp;
                        if (evenpix)
                            img_labels_row[c] = Gp;
                    }
                    else if (evenpix)
                        img_labels_row[c] = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels_row[i];
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels_row[i]);
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels_row_prev[j];
                        if (!curpix)
                            continue;

                        int newroot = FindRoot(reinterpret_cast<int*>(img_labels_.data), curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels_row_prev[j]);
                    }
                }
            }
            img_labels_row_prev = img_labels_row_fol;
            rN += N2;
            r1N += N2;
        }
        // Last row if the number of rows is odd
        if (o_rows) {
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            for (int c = 0; c < N; ++c) {
                img_labels_row[c] = img_row[c];
            }

            for (int c = 0; c < N; ++c) {
                // Step 1
                int evenpix = img_labels_row[c];

                // Step 2
                int Gp;
                if (evenpix)
                    img_labels_row[c] = Gp = -(rN + c);
                else
                    continue;

                // Step 3
                int stripestart = c;
                while (++c < N) {
                    int evenpix = img_labels_row[c];

                    if (evenpix)
                        img_labels_row[c] = Gp;
                    else
                        break;
                }
                int stripestop = c;

                if (r == 0)
                    continue;

                // Stripe union
                int lastroot = INT_MIN;
                for (int i = stripestart; i < stripestop; ++i) {
                    int linepix = img_labels_row[i];
                    if (!linepix)
                        continue;

                    int runstart = std::max(0, i - 1);
                    do
                        i++;
                    while (i < N && img_labels_row[i]);
                    int runstop = std::min(N - 1, i);

                    for (int j = runstart; j <= runstop; ++j) {
                        int curpix = img_labels_row_prev[j];
                        if (!curpix)
                            continue;

                        int newroot = FindRoot(reinterpret_cast<int*>(img_labels_.data), curpix);
                        if (newroot > lastroot) {
                            lastroot = newroot;
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), Gp, lastroot);
                        }
                        else if (newroot < lastroot) {
                            FindRootAndCompress(reinterpret_cast<int*>(img_labels_.data), newroot, lastroot);
                        }

                        do
                            ++j;
                        while (j <= runstop && img_labels_row_prev[j]);
                    }
                }
            }
        }
    }
    void SecondScan()
    {
        // Label assignment
        int *img_labelsdata = reinterpret_cast<int*>(img_labels_.data);
        for (int i = 0; i < M * N; ++i) {
            // FindRoot_GetLabel
            int pos = img_labelsdata[i];
            if (pos >= 0)
                continue;

            int tmp;
            while (true) {
                tmp = img_labelsdata[-pos];
                if (pos == tmp || tmp >= 0)
                    break;
                pos = tmp;
            }
            if (tmp < 0)
                img_labelsdata[-pos] = ++n_labels_;

            // Assign final label
            img_labelsdata[i] = img_labelsdata[-pos];
        }

        // Fix for first pixel!
        if (firstpixel) {
            const_cast<cv::Mat1b&>(img_).data[0] = firstpixel;
            if (N > 1 && img_labels_.data[1])
                img_labels_.data[0] = img_labels_.data[1];
            else if (M > 1 && img_labels_.data[img_.step[0]])
                img_labels_.data[0] = img_labels_.data[img_.step[0]];
            else if (N > 1 && M > 1 && img_labels_.data[img_.step[0] + 1])
                img_labels_.data[0] = img_labels_.data[img_.step[0] + 1];
            else
                img_labels_.data[0] = 1;
        }

        n_labels_++; // To count also background
    }
};

#endif // YACCLAB_LABELING_ZHAO_2010_H_
