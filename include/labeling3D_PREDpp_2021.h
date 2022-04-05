// Copyright (c) 2022, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_PREDPP_3D_H_
#define YACCLAB_LABELING_PREDPP_3D_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

//Actions:
// Action 1: nothing
#define ACTION_1 img_labels_slice00_row00[c] = 0; 
// Action 2: x<-newlabel
#define ACTION_2 img_labels_slice00_row00[c] = LabelsSolver::NewLabel(); 
// Action 3: x<-a
#define ACTION_3 img_labels_slice00_row00[c] = img_labels_slice11_row11[c - 1]; 
// Action 4: x<-b
#define ACTION_4 img_labels_slice00_row00[c] = img_labels_slice11_row11[c]; 
// Action 5: x<-c
#define ACTION_5 img_labels_slice00_row00[c] = img_labels_slice11_row11[c + 1]; 
// Action 6: x<-a+c
#define ACTION_6 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1]); 
// Action 7: x<-d
#define ACTION_7 img_labels_slice00_row00[c] = img_labels_slice11_row00[c - 1]; 
// Action 8: x<-c+d
#define ACTION_8 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row00[c - 1], img_labels_slice11_row11[c + 1]); 
// Action 9: x<-e
#define ACTION_9 img_labels_slice00_row00[c] = img_labels_slice11_row00[c]; 
// Action 10: x<-f
#define ACTION_10 img_labels_slice00_row00[c] = img_labels_slice11_row00[c + 1]; 
// Action 11: x<-a+f
#define ACTION_11 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row11[c - 1]); 
// Action 12: x<-d+f
#define ACTION_12 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row00[c - 1]); 
// Action 13: x<-g
#define ACTION_13 img_labels_slice00_row00[c] = img_labels_slice11_row01[c - 1]; 
// Action 14: x<-a+g
#define ACTION_14 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1]); 
// Action 15: x<-b+g
#define ACTION_15 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c]); 
// Action 16: x<-c+g
#define ACTION_16 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]); 
// Action 17: x<-a+c+g
#define ACTION_17 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]), img_labels_slice11_row11[c - 1]); 
// Action 18: x<-f+g
#define ACTION_18 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]); 
// Action 19: x<-a+f+g
#define ACTION_19 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]), img_labels_slice11_row11[c - 1]); 
// Action 20: x<-h
#define ACTION_20 img_labels_slice00_row00[c] = img_labels_slice11_row01[c]; 
// Action 21: x<-a+h
#define ACTION_21 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c - 1]); 
// Action 22: x<-b+h
#define ACTION_22 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c]); 
// Action 23: x<-c+h
#define ACTION_23 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]); 
// Action 24: x<-a+c+h
#define ACTION_24 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]), img_labels_slice11_row11[c - 1]); 
// Action 25: x<-i
#define ACTION_25 img_labels_slice00_row00[c] = img_labels_slice11_row01[c + 1]; 
// Action 26: x<-a+i
#define ACTION_26 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c - 1]); 
// Action 27: x<-b+i
#define ACTION_27 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c]); 
// Action 28: x<-c+i
#define ACTION_28 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]); 
// Action 29: x<-a+c+i
#define ACTION_29 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]), img_labels_slice11_row11[c - 1]); 
// Action 30: x<-d+i
#define ACTION_30 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]); 
// Action 31: x<-c+d+i
#define ACTION_31 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]), img_labels_slice11_row11[c + 1]); 
// Action 32: x<-g+i
#define ACTION_32 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]); 
// Action 33: x<-a+g+i
#define ACTION_33 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row11[c - 1]); 
// Action 34: x<-b+g+i
#define ACTION_34 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row11[c]); 
// Action 35: x<-c+g+i
#define ACTION_35 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row11[c + 1]); 
// Action 36: x<-a+c+g+i
#define ACTION_36 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]), LabelsSolver::Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1])); 
// Action 37: x<-j
#define ACTION_37 img_labels_slice00_row00[c] = img_labels_slice00_row11[c - 1]; 
// Action 38: x<-c+j
#define ACTION_38 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row11[c + 1]); 
// Action 39: x<-f+j
#define ACTION_39 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row00[c + 1]); 
// Action 40: x<-g+j
#define ACTION_40 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]); 
// Action 41: x<-c+g+j
#define ACTION_41 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row11[c + 1]); 
// Action 42: x<-f+g+j
#define ACTION_42 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row00[c + 1]); 
// Action 43: x<-h+j
#define ACTION_43 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]); 
// Action 44: x<-c+h+j
#define ACTION_44 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]), img_labels_slice11_row11[c + 1]); 
// Action 45: x<-i+j
#define ACTION_45 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]); 
// Action 46: x<-c+i+j
#define ACTION_46 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row11[c + 1]); 
// Action 47: x<-g+i+j
#define ACTION_47 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row01[c - 1]); 
// Action 48: x<-c+g+i+j
#define ACTION_48 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]), LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1])); 
// Action 49: x<-k
#define ACTION_49 img_labels_slice00_row00[c] = img_labels_slice00_row11[c]; 
// Action 50: x<-g+k
#define ACTION_50 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c - 1]); 
// Action 51: x<-h+k
#define ACTION_51 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c]); 
// Action 52: x<-i+k
#define ACTION_52 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]); 
// Action 53: x<-g+i+k
#define ACTION_53 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]), img_labels_slice11_row01[c - 1]); 
// Action 54: x<-l
#define ACTION_54 img_labels_slice00_row00[c] = img_labels_slice00_row11[c + 1]; 
// Action 55: x<-a+l
#define ACTION_55 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row11[c - 1]); 
// Action 56: x<-d+l
#define ACTION_56 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row00[c - 1]); 
// Action 57: x<-g+l
#define ACTION_57 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]); 
// Action 58: x<-a+g+l
#define ACTION_58 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]), img_labels_slice11_row11[c - 1]); 
// Action 59: x<-h+l
#define ACTION_59 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]); 
// Action 60: x<-a+h+l
#define ACTION_60 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]), img_labels_slice11_row11[c - 1]); 
// Action 61: x<-i+l
#define ACTION_61 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]); 
// Action 62: x<-a+i+l
#define ACTION_62 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row11[c - 1]); 
// Action 63: x<-d+i+l
#define ACTION_63 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row00[c - 1]); 
// Action 64: x<-g+i+l
#define ACTION_64 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row01[c - 1]); 
// Action 65: x<-a+g+i+l
#define ACTION_65 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]), LabelsSolver::Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1])); 
// Action 66: x<-j+l
#define ACTION_66 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]); 
// Action 67: x<-g+j+l
#define ACTION_67 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]), img_labels_slice11_row01[c - 1]); 
// Action 68: x<-h+j+l
#define ACTION_68 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]), img_labels_slice11_row01[c]); 
// Action 69: x<-i+j+l
#define ACTION_69 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]), img_labels_slice11_row01[c + 1]); 
// Action 70: x<-g+i+j+l
#define ACTION_70 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]), LabelsSolver::Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1])); 
// Action 71: x<-m
#define ACTION_71 img_labels_slice00_row00[c] = img_labels_slice00_row00[c - 1]; 
// Action 72: x<-c+m
#define ACTION_72 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row11[c + 1]); 
// Action 73: x<-f+m
#define ACTION_73 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row00[c + 1]); 
// Action 74: x<-i+m
#define ACTION_74 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]); 
// Action 75: x<-c+i+m
#define ACTION_75 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]), img_labels_slice11_row11[c + 1]); 
// Action 76: x<-l+m
#define ACTION_76 img_labels_slice00_row00[c] = LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]); 
// Action 77: x<-i+l+m
#define ACTION_77 img_labels_slice00_row00[c] = LabelsSolver::Merge(LabelsSolver::Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]), img_labels_slice11_row01[c + 1]);  


// Note. We are just considering special forests (first line, etc) based on the current slice. This means
// that first_line_first_slice etc are not considered.
template <typename LabelsSolver>
class PREDpp_3D : public Labeling3D<Connectivity3D::CONN_26> {
public:
	PREDpp_3D() {}

    void FirstSlice(int d, int w, int h) {
        // First slice
        //Conditions for the first slice: TODO generate also first slice forest
#define CONDITION_A false
#define CONDITION_B false
#define CONDITION_C false
#define CONDITION_D false
#define CONDITION_E false
#define CONDITION_F false
#define CONDITION_G false
#define CONDITION_H false
#define CONDITION_I false
#define CONDITION_J img_slice00_row11[c - 1] > 0
#define CONDITION_K img_slice00_row11[c] > 0
#define CONDITION_L img_slice00_row11[c + 1] > 0
#define CONDITION_M img_slice00_row00[c - 1] > 0
#define CONDITION_X img_slice00_row00[c] > 0

            // First line of the current slice
            {
                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(0, 0);

                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(0, 0);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);


                int c = -1;
                goto fl_tree_0;
#include "labeling3D_PREDpp_2021_fl_forest.inc.h"
            }

            // Center lines of the current slice
            for (int r = 1; r < h; r++) {

                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(0, r);
                const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);

                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(0, r);
                unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);

                int c = -1;
                goto ll_tree_0;
#include "labeling3D_PREDpp_2021_ll_forest.inc.h"

            } // Rows cycle end


#undef CONDITION_A
#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_F
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_L
#undef CONDITION_M
#undef CONDITION_X

    }

	void PerformLabeling()
	{
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
		int d = img_.size.p[0];
		int h = img_.size.p[1];
		int w = img_.size.p[2];

        // TODO Single line case is missing

        FirstSlice(d, w, h);

        //Conditions for all the slices but the first one
#define CONDITION_A img_slice11_row11[c - 1] > 0
#define CONDITION_B img_slice11_row11[c] > 0
#define CONDITION_C img_slice11_row11[c + 1] > 0
#define CONDITION_D img_slice11_row00[c - 1] > 0
#define CONDITION_E img_slice11_row00[c] > 0
#define CONDITION_F img_slice11_row00[c + 1] > 0
#define CONDITION_G img_slice11_row01[c - 1] > 0
#define CONDITION_H img_slice11_row01[c] > 0
#define CONDITION_I img_slice11_row01[c + 1] > 0
#define CONDITION_J img_slice00_row11[c - 1] > 0
#define CONDITION_K img_slice00_row11[c] > 0
#define CONDITION_L img_slice00_row11[c + 1] > 0
#define CONDITION_M img_slice00_row00[c - 1] > 0
#define CONDITION_X img_slice00_row00[c] > 0

        // All the other slices
		for (int s = 1; s < d; s++) {

            // First line of the current slice
            {
                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, 0);
               
                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, 0);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);
                int c = -1;
                goto fl_tree_0;
#include "labeling3D_PREDpp_2021_fl_forest.inc.h"
            }

            // Center lines of the current slice
			for (int r = 1; r < h - 1; r++) {

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
                int c = -1;
                goto cl_tree_0;
#include "labeling3D_PREDpp_2021_cl_forest.inc.h"
				
			} // Rows cycle end

            // Last line of the current slice
            {
                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, h - 1);
                const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);

                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                
                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, h - 1);
                unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                int c = -1;
                goto ll_tree_0;
#include "labeling3D_PREDpp_2021_ll_forest.inc.h"
            }
		} // Planes cycle end

#undef CONDITION_A
#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_F
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_L
#undef CONDITION_M
#undef CONDITION_X

		// Second scan
		LabelsSolver::Flatten();

		int * img_row = reinterpret_cast<int*>(img_labels_.data);
		for (int s = 0; s < d; s++) {
			for (int r = 0; r < h; r++) {
				for (int c = 0; c < w; c++) {
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

		int d = img_.size.p[0];
		int h = img_.size.p[1];
		int w = img_.size.p[2];

        // TODO Single line case is missing

        FirstSlice(d, w, h);

        //Conditions for all the slices but the first one
#define CONDITION_A img_slice11_row11[c - 1] > 0
#define CONDITION_B img_slice11_row11[c] > 0
#define CONDITION_C img_slice11_row11[c + 1] > 0
#define CONDITION_D img_slice11_row00[c - 1] > 0
#define CONDITION_E img_slice11_row00[c] > 0
#define CONDITION_F img_slice11_row00[c + 1] > 0
#define CONDITION_G img_slice11_row01[c - 1] > 0
#define CONDITION_H img_slice11_row01[c] > 0
#define CONDITION_I img_slice11_row01[c + 1] > 0
#define CONDITION_J img_slice00_row11[c - 1] > 0
#define CONDITION_K img_slice00_row11[c] > 0
#define CONDITION_L img_slice00_row11[c + 1] > 0
#define CONDITION_M img_slice00_row00[c - 1] > 0
#define CONDITION_X img_slice00_row00[c] > 0

        // All the other slices
        for (int s = 1; s < d; s++) {

            // First line of the current slice
            {
                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, 0);

                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);

                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, 0);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);
                int c = -1;
                goto fl_tree_0;
#include "labeling3D_PREDpp_2021_fl_forest.inc.h"
            }

            // Center lines of the current slice
            for (int r = 1; r < h - 1; r++) {

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
                int c = -1;
                goto cl_tree_0;
#include "labeling3D_PREDpp_2021_cl_forest.inc.h"

            } // Rows cycle end

            // Last line of the current slice
            {
                // Row pointers for the input image (current slice)
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, h - 1);
                const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);

                // Row pointers for the input image (previous slice)
                const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
                const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);

                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, h - 1);
                unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);

                // Row pointers for the output image (previous slice)
                unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
                unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                int c = -1;
                goto ll_tree_0;
#include "labeling3D_PREDpp_2021_ll_forest.inc.h"
            }
        } // Planes cycle end
	

#undef CONDITION_A
#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E
#undef CONDITION_F
#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I

#undef CONDITION_J
#undef CONDITION_K
#undef CONDITION_L
#undef CONDITION_M
#undef CONDITION_X
	
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
#undef ACTION_50
#undef ACTION_51
#undef ACTION_52
#undef ACTION_53
#undef ACTION_54
#undef ACTION_55
#undef ACTION_56
#undef ACTION_57
#undef ACTION_58
#undef ACTION_59
#undef ACTION_60
#undef ACTION_61
#undef ACTION_62
#undef ACTION_63
#undef ACTION_64
#undef ACTION_65
#undef ACTION_66
#undef ACTION_67
#undef ACTION_68
#undef ACTION_69
#undef ACTION_70
#undef ACTION_71
#undef ACTION_72
#undef ACTION_73
#undef ACTION_74
#undef ACTION_75
#undef ACTION_76
#undef ACTION_77


#endif // !YACCLAB_LABELING_PREDPP_3D_H_