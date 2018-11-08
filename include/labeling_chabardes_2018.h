// copyright(c) 2018 theodore chabardes, petr dokladal, michel bilodeau
// all rights reserved.
//
// redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
//
// *redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
//
// * neither the name of yacclab nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// this software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose are
// disclaimed.in no event shall the copyright holder or contributors be liable
// for any direct, indirect, incidental, special, exemplary, or consequential
// damages(including, but not limited to, procurement of substitute goods or
// services; loss of use, data, or profits; or business interruption) however
// caused and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of the use
// of this software, even if advised of the possibility of such damage.

#ifndef YACCLAB_LABELING_CHABARDES_2018_H_
#define YACCLAB_LABELING_CHABARDES_2018_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "labeling_chabardes_2018_first_row.inc"
#include "labeling_chabardes_2018_middle_row.inc"
#include "labeling_chabardes_2018_last_row.inc"

template <typename LabelsSolver>
class BBDF : public Labeling {
public:
        BBDF() {}

        void PerformLabeling()
        {
                img_labels_ = cv::Mat1i(img_.size()); // Call to memset

                LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
                LabelsSolver::Setup();

                // Mask
                // +-+-+-+-+
                // |a|b|c|d|
                // +-+-+-+-+
                // |e|x|x|
                // +-+-+-+
                // |f|x|x|
                // +-+-+-+

                int w(img_.cols);
                int h(img_.rows);

                n_labels_ = 0;

                switch (w%2) {
                case 0:
                        if (h%2 == 0) {
                                FirstScan_opt_even_even(w,h);
                        } else {
                                FirstScan_opt_even_odd(w,h);
                        }
                        break;
                case 1:
                        if (h%2 == 0) {
                                FirstScan_opt_odd_even(w,h);
                        } else {
                                FirstScan_opt_odd_odd(w,h);
                        }
                        break;
                }

                // Second scan
                n_labels_ = LabelsSolver::Flatten();


                int e_rows = img_labels_.rows & 0xfffffffe;
                bool o_rows = img_labels_.rows % 2 == 1;
                int e_cols = img_labels_.cols & 0xfffffffe;
                bool o_cols = img_labels_.cols % 2 == 1;

            		int r = 0;
            		for (; r < e_rows; r += 2) {
            			// Get rows pointer
            			const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            			const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            
            			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            			unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
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

        void PerformLabelingMem(std::vector<unsigned long int>& accesses)
        {

                MemMat<unsigned char> img(img_);
                MemMat<int> img_labels(img_.size(), 0); 

                LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
                LabelsSolver::MemSetup();

                // First scan
                int w(img_.cols);
                int h(img_.rows);

                switch (w%2) {
                case 0:
                        if (h%2 == 0) {
                                FirstScan_mem_even_even(img,img_labels,w,h);
                        } else {
                                FirstScan_mem_even_odd(img,img_labels,w,h);
                        }
                        break;
                case 1:
                        if (h%2 == 0) {
                                FirstScan_mem_odd_even(img,img_labels,w,h);
                        } else {
                                FirstScan_mem_odd_odd(img,img_labels,w,h);
                        }
                        break;
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
            		accesses = std::vector<unsigned long int>((int)MD_SIZE, 0);
            
            		accesses[MD_BINARY_MAT] = (unsigned long int)img.GetTotalAccesses();
            		accesses[MD_LABELED_MAT] = (unsigned long int)img_labels.GetTotalAccesses();
            		accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)LabelsSolver::MemTotalAccesses();
            
            		img_labels_ = img_labels.GetImage();
            
            		LabelsSolver::MemDealloc();
        }
private:
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
        LabelsSolver::Setup();

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

        switch (w%2) {
        case 0:
                if (h%2 == 0) {
                        FirstScan_opt_even_even(w,h);
                } else {
                        FirstScan_opt_even_odd(w,h);
                }
                break;
        case 1:
                if (h%2 == 0) {
                        FirstScan_opt_odd_even(w,h);
                } else {
                        FirstScan_opt_odd_odd(w,h);
                }
                break;
        }
    }

    void SecondScan()
    {
     	// Second scan
    	n_labels_ = LabelsSolver::Flatten();
    
    	int e_rows = img_labels_.rows & 0xfffffffe;
    	bool o_rows = img_labels_.rows % 2 == 1;
    	int e_cols = img_labels_.cols & 0xfffffffe;
    	bool o_cols = img_labels_.cols % 2 == 1;
    
    	int r = 0;
    	for (; r < e_rows; r += 2) {
    		// Get rows pointer
    		const unsigned char* const img_row = img_.ptr<unsigned char>(r);
    		const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
    
    		unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
    		unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
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


// Optimised version
#define BLOCK_x imgLabels_row[c]
#define BLOCK_a imgLabels_row_prev_prev[c - 2]
#define BLOCK_b imgLabels_row_prev_prev[c]
#define BLOCK_c imgLabels_row_prev_prev[c]
#define BLOCK_d imgLabels_row_prev_prev[c + 2]
#define BLOCK_e imgLabels_row[c - 2]
#define BLOCK_f imgLabels_row[c - 2]

// Actions
#define MERGE(a,b)      LabelsSolver::Merge(a, b)
#define NEW_LABEL       LabelsSolver::NewLabel();

// Conditions. 
#define condition_x1 img_row[c] > 0
#define condition_x2 img_row[c + 1] > 0
#define condition_x3 img_row_fol[c] > 0
#define condition_x4 img_row_fol[c + 1] > 0
#define condition_a img_row_prev[c - 1] > 0
#define condition_b img_row_prev[c] > 0
#define condition_c img_row_prev[c + 1] > 0
#define condition_d img_row_prev[c + 2] > 0
#define condition_e img_row[c - 1] > 0
#define condition_f img_row_fol[c - 1] > 0

    void FirstScan_opt_even_even (const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_OPT_EVEN_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto OPT_EVEN_EVEN_END;\
                goto OPT_EVEN_EVEN_MIDDLE; \
        }
        {
                const uchar *const img_row = img_.ptr<uchar>(0);                          
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(0);                 

                FIRST_ROW_EVEN(OPT_EVEN_EVEN)
        }
#undef LOOP_HANDLING

        // Middle row
        OPT_EVEN_EVEN_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_OPT_EVEN_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto OPT_EVEN_EVEN_END;\
                goto OPT_EVEN_EVEN_MIDDLE; \
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);                 

                MIDDLE_ROW_EVEN(OPT_EVEN_EVEN)
        }
#undef LOOP_HANDLING

        OPT_EVEN_EVEN_END:;
    }

    void FirstScan_opt_odd_even (const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_OPT_ODD_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto OPT_ODD_EVEN_END;\
                goto OPT_ODD_EVEN_MIDDLE; \
        }
        {
                const uchar *const img_row = img_.ptr<uchar>(0);                          
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(0);                 

                FIRST_ROW_ODD(OPT_ODD_EVEN)
        }
#undef LOOP_HANDLING

        // Middle row
        OPT_ODD_EVEN_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_OPT_ODD_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto OPT_ODD_EVEN_END;\
                goto OPT_ODD_EVEN_MIDDLE; \
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);                 

                MIDDLE_ROW_ODD(OPT_ODD_EVEN)
        }
#undef LOOP_HANDLING

        OPT_ODD_EVEN_END:;

    }

    void FirstScan_opt_even_odd (const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;
        int r_up = ((h-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_OPT_EVEN_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto OPT_EVEN_ODD_LAST;\
                goto OPT_EVEN_ODD_MIDDLE; \
        }
        {
                const uchar *const img_row = img_.ptr<uchar>(0);                          
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(0);                 

                FIRST_ROW_EVEN(OPT_EVEN_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        OPT_EVEN_ODD_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_OPT_EVEN_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto OPT_EVEN_ODD_LAST;\
                goto OPT_EVEN_ODD_MIDDLE; \
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);
                MIDDLE_ROW_EVEN(OPT_EVEN_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        OPT_EVEN_ODD_LAST:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_LAST_ROW_OPT_EVEN_ODD;\
        } else if (c >= w) {\
                goto OPT_EVEN_ODD_END;\
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);         

                LAST_ROW_EVEN(OPT_EVEN_ODD)
        }
#undef LOOP_HANDLING

        OPT_EVEN_ODD_END:;

    }

    void FirstScan_opt_odd_odd (const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;
        int r_up = ((h-1)/2)*2;


        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_OPT_ODD_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto OPT_ODD_ODD_LAST;\
                goto OPT_ODD_ODD_MIDDLE; \
        }
        {
                const uchar *const img_row = img_.ptr<uchar>(0);                          
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(0);                 

                FIRST_ROW_ODD(OPT_ODD_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        OPT_ODD_ODD_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_OPT_ODD_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto OPT_ODD_ODD_LAST;\
                goto OPT_ODD_ODD_MIDDLE; \
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                const uchar *const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);                                       

                MIDDLE_ROW_ODD(OPT_ODD_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        OPT_ODD_ODD_LAST:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_LAST_ROW_OPT_ODD_ODD;\
        } else if (c >= w) {\
                goto OPT_ODD_ODD_END;\
        } 

        {
                const uchar *const img_row = img_.ptr<uchar>(r);
                const uchar *const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
                uint *const imgLabels_row = img_labels_.ptr<uint>(r);                 
                uint *const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);                                       

                LAST_ROW_ODD(OPT_ODD_ODD)
        }
#undef LOOP_HANDLING

        OPT_ODD_ODD_END:;

    }

#undef BLOCK_x
#undef BLOCK_a
#undef BLOCK_b
#undef BLOCK_c
#undef BLOCK_d
#undef BLOCK_e
#undef BLOCK_f
#undef MERGE
#undef NEW_LABEL
#undef condition_x1
#undef condition_x2
#undef condition_x3
#undef condition_x4
#undef condition_a
#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e
#undef condition_f

// Memory version
// Correspondance between pixels and blocks.
#define BLOCK_x img_labels(r,c)
#define BLOCK_a (uint) img_labels(r-2, c-2)
#define BLOCK_b (uint) img_labels(r-2, c)
#define BLOCK_c (uint) img_labels(r-2, c)
#define BLOCK_d (uint) img_labels(r-2, c+2)
#define BLOCK_e (uint) img_labels(r, c-2)
#define BLOCK_f (uint) img_labels(r, c-2)

// Actions
#define MERGE(a,b)      LabelsSolver::MemMerge(a, b)
#define NEW_LABEL       LabelsSolver::MemNewLabel()

// Conditions.
#define condition_x1 img(r, c) > 0
#define condition_x2 img(r, c+1) > 0
#define condition_x3 img(r+1, c) > 0
#define condition_x4 img(r+1, c+1) > 0
#define condition_a img(r-1, c-1) > 0
#define condition_b img(r-1, c) > 0
#define condition_c img(r-1, c+1) > 0
#define condition_d img(r-1, c+2) > 0
#define condition_e img(r, c-1) > 0
#define condition_f img(r+1, c-1) > 0

    void FirstScan_mem_even_even (MemMat<unsigned char> &img, MemMat<int> &img_labels, const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;


        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_MEM_EVEN_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto MEM_EVEN_EVEN_END;\
                goto MEM_EVEN_EVEN_MIDDLE; \
        }
        {
                FIRST_ROW_EVEN(MEM_EVEN_EVEN)
        }
#undef LOOP_HANDLING

        // Middle row
        MEM_EVEN_EVEN_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_MEM_EVEN_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto MEM_EVEN_EVEN_END;\
                goto MEM_EVEN_EVEN_MIDDLE; \
        } 

        {
                MIDDLE_ROW_EVEN(MEM_EVEN_EVEN)
        }
#undef LOOP_HANDLING

        MEM_EVEN_EVEN_END:;

    }

    void FirstScan_mem_odd_even (MemMat<unsigned char> &img, MemMat<int> &img_labels, const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_MEM_ODD_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto MEM_ODD_EVEN_END;\
                goto MEM_ODD_EVEN_MIDDLE; \
        }
        {
                FIRST_ROW_ODD(MEM_ODD_EVEN)
        }
#undef LOOP_HANDLING

        // Middle row
        MEM_ODD_EVEN_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_MEM_ODD_EVEN;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= h)\
                        goto MEM_ODD_EVEN_END;\
                goto MEM_ODD_EVEN_MIDDLE; \
        } 

        {
                MIDDLE_ROW_ODD(MEM_ODD_EVEN)
        }
#undef LOOP_HANDLING

        MEM_ODD_EVEN_END:;

    }

    void FirstScan_mem_even_odd (MemMat<unsigned char> &img, MemMat<int> &img_labels, const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;
        int r_up = ((h-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_MEM_EVEN_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto MEM_EVEN_ODD_LAST;\
                goto MEM_EVEN_ODD_MIDDLE; \
        }
        {
                FIRST_ROW_EVEN(MEM_EVEN_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        MEM_EVEN_ODD_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_MEM_EVEN_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto MEM_EVEN_ODD_LAST;\
                goto MEM_EVEN_ODD_MIDDLE; \
        } 

        {
                MIDDLE_ROW_EVEN(MEM_EVEN_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        MEM_EVEN_ODD_LAST:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_LAST_ROW_MEM_EVEN_ODD;\
        } else if (c >= w) {\
                goto MEM_EVEN_ODD_END;\
        } 

        {
                LAST_ROW_EVEN(MEM_EVEN_ODD)
        }
#undef LOOP_HANDLING

        MEM_EVEN_ODD_END:;

    }

    void FirstScan_mem_odd_odd (MemMat<unsigned char> &img, MemMat<int> &img_labels, const int w, const int h) {
        int c=0, r=0;
        int c_up = ((w-1)/2)*2;
        int r_up = ((h-1)/2)*2;

        // First row
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_FIRST_ROW_MEM_ODD_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto MEM_ODD_ODD_LAST;\
                goto MEM_ODD_ODD_MIDDLE; \
        }
        {
                FIRST_ROW_ODD(MEM_ODD_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        MEM_ODD_ODD_MIDDLE:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_MIDDLE_ROW_MEM_ODD_ODD;\
        } else if (c >= w) {\
                c=0;\
                r+=2;\
                if (r >= r_up)\
                        goto MEM_ODD_ODD_LAST;\
                goto MEM_ODD_ODD_MIDDLE; \
        } 

        {
                MIDDLE_ROW_ODD(MEM_ODD_ODD)
        }
#undef LOOP_HANDLING

// Middle row
        MEM_ODD_ODD_LAST:
#define LOOP_HANDLING\
        c += 2;\
        if (c == c_up) { \
                goto END_LAST_ROW_MEM_ODD_ODD;\
        } else if (c >= w) {\
                goto MEM_ODD_ODD_END;\
        } 

        {
                LAST_ROW_ODD(MEM_ODD_ODD)
        }
#undef LOOP_HANDLING

        MEM_ODD_ODD_END:;

    }


#undef BLOCK_x
#undef BLOCK_a
#undef BLOCK_b
#undef BLOCK_c
#undef BLOCK_d
#undef BLOCK_e
#undef BLOCK_f
#undef MERGE
#undef NEW_LABEL
#undef condition_x1
#undef condition_x2
#undef condition_x3
#undef condition_x4
#undef condition_a
#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e
#undef condition_f

};

#endif // YACCLAB_LABELING_CHABARDES_2018_H_
