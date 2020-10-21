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

#ifndef YACCLAB_LABELING3D_BBDT_26C_221B_H_
#define YACCLAB_LABELING3D_BBDT_26C_221B_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "labeling3D_BBDT_26c_221b_action_definition.inc.h"

        //Conditions:
#define CONDITION_KD c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LC r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LD c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MC c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_ND c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_OC r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_OD c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_PC c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 2 && s > 0 && img_slice11_row02[c - 1] > 0
#define CONDITION_RA r < h - 2 && s > 0 && img_slice11_row02[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 2 && s > 0 && img_slice11_row02[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 2 && s > 0 && img_slice11_row02[c + 2] > 0
#define CONDITION_TD c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UC r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UD c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VC c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_WD c > 0 && r < h - 1 && img_slice00_row01[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_XC r < h - 1 && img_slice00_row01[c] > 0
#define CONDITION_XD c < w - 1 && r < h - 1 && img_slice00_row01[c + 1] > 0


template <typename LabelsSolver>
class BBDT_3D_26c_221b : public Labeling3D<CONN_26> {
public:
    BBDT_3D_26c_221b() {}


    //    void FirstSlice(unsigned int d, unsigned int w, unsigned int h) {
    //        // First slice
    //        //Conditions for the first slice: TODO generate also first slice forest
    ////Conditions:
    //#define CONDITION_KH false
    //#define CONDITION_LG false
    //#define CONDITION_LH false
    //#define CONDITION_MG false
    //#define CONDITION_NF false
    //#define CONDITION_NH false
    //#define CONDITION_OE false
    //#define CONDITION_OF false
    //#define CONDITION_OG false
    //#define CONDITION_OH false
    //#define CONDITION_PE false
    //#define CONDITION_PG false
    //#define CONDITION_QF false
    //#define CONDITION_RE false
    //#define CONDITION_RF false
    //#define CONDITION_SE false
    //#define CONDITION_TD c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
    //#define CONDITION_TH c > 0 && r > 0 && s < d - 1 && img_slice01_row11[c - 1] > 0
    //#define CONDITION_UC r > 0 && img_slice00_row11[c] > 0
    //#define CONDITION_UD c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
    //#define CONDITION_UG r > 0 && s < d - 1 && img_slice01_row11[c] > 0
    //#define CONDITION_UH c < w - 1 && r > 0 && s < d - 1 && img_slice01_row11[c + 1] > 0
    //#define CONDITION_VC c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
    //#define CONDITION_VG c < w - 2 && r > 0 && s < d - 1 && img_slice01_row11[c + 2] > 0
    //#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
    //#define CONDITION_WD c > 0 && r < h - 1 && img_slice00_row01[c - 1] > 0
    //#define CONDITION_WF c > 0 && s < d - 1 && img_slice01_row00[c - 1] > 0
    //#define CONDITION_WH c > 0 && r < h - 1 && s < d - 1 && img_slice01_row01[c - 1] > 0
    //#define CONDITION_XA img_slice00_row00[c] > 0
    //#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
    //#define CONDITION_XC r < h - 1 && img_slice00_row01[c] > 0
    //#define CONDITION_XD c < w - 1 && r < h - 1 && img_slice00_row01[c + 1] > 0
    //#define CONDITION_XE s < d - 1 && img_slice01_row00[c] > 0
    //#define CONDITION_XF c < w - 1 && s < d - 1 && img_slice01_row00[c + 1] > 0
    //#define CONDITION_XG r < h - 1 && s < d - 1 && img_slice01_row01[c] > 0
    //#define CONDITION_XH c < w - 1 && r < h - 1 && s < d - 1 && img_slice01_row01[c + 1] > 0
    //		for (unsigned int s = 0; s < 2; s += 2) {
    //			for (unsigned int r = 0; r < h; r += 2) {
    //
    //				const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
    //				// T, W lower slice (Xe-Xh)
    //				//const unsigned char* const img_slice01_row12 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[0] + img_.step.p[1] * -2);
    //				const unsigned char* const img_slice01_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[0] + img_.step.p[1] * -1);
    //				const unsigned char* const img_slice01_row00 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[0] + img_.step.p[1] * 0);
    //				const unsigned char* const img_slice01_row01 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[0] + img_.step.p[1] * 1);
    //
    //				// T, W upper slice (Xa-Xd)
    //				//const unsigned char* const img_slice00_row12 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -2);
    //				const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -1);
    //				const unsigned char* const img_slice00_row01 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * 1);
    //
    //				// K, N, Q lower slice
    //				//const unsigned char* const img_slice11_row12 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
    //				const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
    //				const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
    //				const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
    //				const unsigned char* const img_slice11_row02 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
    //				//const unsigned char* const img_slice11_row03 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);
    //
    //				//// K, N, Q upper slice
    //				//const unsigned char* const img_slice12_row12 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
    //				//const unsigned char* const img_slice12_row11 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
    //				//const unsigned char* const img_slice12_row00 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
    //				//const unsigned char* const img_slice12_row01 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
    //				//const unsigned char* const img_slice12_row02 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
    //				//const unsigned char* const img_slice12_row03 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);
    //
    //				// Row pointers for the output image (current slice)
    //
    //				// T, W lower slice (Xe-Xh)
    //				//unsigned* const img_labels_slice01_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[0] + img_.step.p[1] * -2);
    //				//unsigned* const img_labels_slice01_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[0] + img_.step.p[1] * -1);
    //				//unsigned* const img_labels_slice01_row00 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[0] + img_.step.p[1] * 0);
    //				//unsigned* const img_labels_slice01_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[0] + img_.step.p[1] * 1);
    //
    //				unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
    //				// T, W upper slice (Xa-Xd)
    //				unsigned* const img_labels_slice00_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[1] * -2);
    //				//unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[1] * -1);
    //				//unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + img_.step.p[1] * 1);
    //
    //				// K, N, Q lower slice
    //				//unsigned* const img_labels_slice11_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
    //				//unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
    //				//unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
    //				//unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
    //				//unsigned* const img_labels_slice11_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
    //				//unsigned* const img_labels_slice11_row03 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);
    //
    //				//// K, N, Q upper slice
    //				unsigned* const img_labels_slice12_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
    //				//unsigned* const img_labels_slice12_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
    //				unsigned* const img_labels_slice12_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
    //				//unsigned* const img_labels_slice12_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
    //				unsigned* const img_labels_slice12_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
    //				//unsigned* const img_labels_slice12_row03 = (unsigned *)(((char *)img_labels_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);
    //				for (unsigned int c = 0; c < w; c += 2) {
    //#include "labeling3D_BBDT++3D_justequalsubtrees_tree.inc.h"
    //
    //				}
    //			}
    //        } // Rows cycle end
    //
    //#undef CONDITION_KH
    //#undef CONDITION_LG
    //#undef CONDITION_LH
    //#undef CONDITION_MG
    //#undef CONDITION_NF
    //#undef CONDITION_NH
    //#undef CONDITION_OE
    //#undef CONDITION_OF
    //#undef CONDITION_OG
    //#undef CONDITION_OH
    //#undef CONDITION_PE
    //#undef CONDITION_PG
    //#undef CONDITION_QF
    //#undef CONDITION_RE
    //#undef CONDITION_RF
    //#undef CONDITION_SE
    //#undef CONDITION_TD
    //#undef CONDITION_TH
    //#undef CONDITION_UC
    //#undef CONDITION_UD
    //#undef CONDITION_UG
    //#undef CONDITION_UH
    //#undef CONDITION_VC
    //#undef CONDITION_VG
    //#undef CONDITION_WB
    //#undef CONDITION_WD
    //#undef CONDITION_WF
    //#undef CONDITION_WH
    //#undef CONDITION_XA
    //#undef CONDITION_XB
    //#undef CONDITION_XC
    //#undef CONDITION_XD
    //#undef CONDITION_XE
    //#undef CONDITION_XF
    //#undef CONDITION_XG
    //#undef CONDITION_XH
    //
    //    }

    void PerformLabeling()
    {
        img_labels_.create(3, img_.size.p, CV_32SC1);

        LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
        LabelsSolver::Setup(); // Labels solver initialization

        //// DEBUG
        //LabelsSolver::MemAlloc(UPPER_BOUND_26_CONNECTIVITY); // Equivalence solver

        //MemVol<unsigned char> img(img_);
        //MemVol<int> img_labels(img_.size.p, 0);

        //LabelsSolver::MemSetup();
        //// DEBUG - END

        // Rosenfeld Mask 3D
        // +-+-+-+
        // |a|b|c|
        // +-+-+-+
        // |d|e|f|
        // +-+-+-+
        // |g|h|i|
        // +-+-+-+
        //
        // +-+-+-+
        // |j|k|l|
        // +-+-+-+
        // |m|x|
        // +-+-+

        // First scan
        unsigned int d = img_.size.p[0];
        unsigned int h = img_.size.p[1];
        unsigned int w = img_.size.p[2];

        // First slice
        //FirstSlice(d, w, h);

        for (unsigned int s = 0; s < d; s += 1) {

            for (unsigned int r = 0; r < h; r += 2) {

                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
                // T, W slice
                //const unsigned char* const img_slice00_row12 = (unsigned char *)(((char *)img_slice00_row00) + img_.step.p[1] * -2);
                const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * -1);
                // img_slice00_row00 defined above
                const unsigned char* const img_slice00_row01 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * 1);

                // K, N, Q slice
                //const unsigned char* const img_slice11_row12 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -2);
                const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
                const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
                const unsigned char* const img_slice11_row02 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);
                //const unsigned char* const img_slice11_row03 = (unsigned char *)(((char *)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 3);


                // Row pointers for the output image (current slice)
                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
                // T, W slice
                unsigned* const img_labels_slice00_row12 = (unsigned*)(((char*)img_labels_slice00_row00) + img_labels_.step.p[1] * -2);
                //unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * -1);
                // img_labels_slice00_row00 defined above
                //unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + img_labels_.step.p[1] * 1);

                // K, N, Q slice
                unsigned* const img_labels_slice11_row12 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -2);
                //unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -1);
                unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                //unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 1);
                unsigned* const img_labels_slice11_row02 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 2);
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
                    if (!((CONDITION_XA) || (CONDITION_XB) || (CONDITION_XC) || (CONDITION_XD))) {
                        ACTION_0;
                    }
#include "labeling3D_BBDT_26c_221b_tree.inc.h"
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

        const unsigned char* const img_row = img_.ptr<unsigned char>();
        int* const img_labels_row = img_labels_.ptr<int>();

        /*
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
        */


        //		for (unsigned s = 0; s < d; s += 1) {
        //			for (unsigned r = 0; r < h; r += 2) {
        //				for (unsigned c = 0; c < w; c += 2) {
        //					//int iLabel = *reinterpret_cast<int*>(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c));
        //					int iLabel = img_labels_row[c + r * w + s * h * w];
        //					if (iLabel > 0) {
        //						iLabel = LabelsSolver::GetLabel(iLabel);
        //
        //						if (img_row[c + r * w + s * h * w] > 0)
        //							img_labels_row[c + r * w + s * h * w] = iLabel;
        //						else
        //							img_labels_row[c + r * w + s * h * w] = 0;
        //
        //						if (c < w - 1) {
        //							if (img_row[(c + 1) + r * w + s * h * w] > 0)
        //								img_labels_row[(c + 1) + r * w + s * h * w] = iLabel;
        //							else
        //								img_labels_row[(c + 1) + r * w + s * h * w] = 0;
        //						}
        //
        //						if (r < h - 1) {
        //							if (img_row[c + (r + 1) * w + s * h * w] > 0)
        //								img_labels_row[c + (r + 1) * w + s * h * w] = iLabel;
        //							else
        //								img_labels_row[c + (r + 1) * w + s * h * w] = 0;
        //						}
        //
        //						if (c < w - 1 && r < h - 1) {
        //							if (img_row[(c + 1) + (r + 1) * w + s * h * w] > 0)
        //								img_labels_row[(c + 1) + (r + 1) * w + s * h * w] = iLabel;
        //							else
        //								img_labels_row[(c + 1) + (r + 1) * w + s * h * w] = 0;
        //						}
        //
        //					//	std::cout << "\nimg step p0: " << img_.step.p[0] << "img step p1: " << img_.step.p[1] << "img step p2: " << img_.step.p[2];
        //					//	std::cout << "\nd: " << d << " h: " << h << " w: " << w;
        ///*
        //						if (*(img_row + (img_.step.p[0] * s) + (img_.step.p[1] * r) + (img_.step.p[2] * c)) > 0)
        //							*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c)) = iLabel;
        //						else 
        //							*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c)) = 0;
        //
        //						if (*(img_row + (img_.step.p[0] * s) + (img_.step.p[1] * r) + (img_.step.p[2] * c + 1)) > 0)
        //							*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c + 1)) = iLabel;
        //						if (*(img_row + (img_.step.p[0] * s) + (img_.step.p[1] * r + 1) + (img_.step.p[2] * c)) > 0)
        //							*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c)) = iLabel;
        //						if (*(img_row + (img_.step.p[0] * s) + (img_.step.p[1] * r + 1) + (img_.step.p[2] * c + 1)) > 0)
        //							*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c + 1)) = iLabel;
        //*/
        //
        //						//if (*(img_row + (img_.step.p[0] * s + 1) + (img_.step.p[1] * r) + (img_.step.p[2] * c)) > 0)
        //						//	*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c)) = iLabel;
        //						//if (*(img_row + (img_.step.p[0] * s + 1) + (img_.step.p[1] * r) + (img_.step.p[2] * c + 1)) > 0)
        //						//	*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c + 1)) = iLabel;
        //						//if (*(img_row + (img_.step.p[0] * s + 1) + (img_.step.p[1] * r + 1) + (img_.step.p[2] * c)) > 0)
        //						//	*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c)) = iLabel;
        //						//if (*(img_row + (img_.step.p[0] * s + 1) + (img_.step.p[1] * r + 1) + (img_.step.p[2] * c + 1)) > 0)
        //						//	*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c + 1)) = iLabel;
        //					}
        //					else {
        //						//*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c)) = 0;
        //						if (c < w - 1) 
        //							img_labels_row[(c + 1) + r * w + s * h * w] = 0;
        //						if (r < h - 1) 
        //							img_labels_row[c + (r + 1) * w + s * h * w] = 0;
        //						if (c < w - 1 && r < h - 1) 
        //							img_labels_row[(c + 1) + (r + 1) * w + s * h * w] = 0;
        //					/*
        //						*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c + 1)) = 0;
        //						*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c)) = 0;
        //						*(img_labels_row + (img_labels_.step.p[0] * s) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c + 1)) = 0;
        //					*/
        //
        //						//*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c)) = 0;
        //						//*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r) + (img_labels_.step.p[2] * c + 1)) = 0;
        //						//*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c)) = 0;
        //						//*(img_labels_row + (img_labels_.step.p[0] * s + 1) + (img_labels_.step.p[1] * r + 1) + (img_labels_.step.p[2] * c + 1)) = 0;
        //					}
        //				}
        //			}
        //		}

        // NEW VERSION BELOW, OLD ABOVE
        int e_rows = h & 0xfffffffe;
        bool o_rows = h % 2 == 1;
        int e_cols = w & 0xfffffffe;
        bool o_cols = w % 2 == 1;

        for (unsigned s = 0; s < d; s++) {
            int r = 0;
            for (; r < e_rows; r += 2) {
                // Get rows pointer
                const unsigned char* const img_row = img_.ptr<unsigned char>(s, r);
                const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[1]);

                unsigned* const img_labels_row = img_labels_.ptr<unsigned>(s, r);
                unsigned* const img_labels_row_fol = (unsigned*)(((char*)img_labels_row) + img_labels_.step.p[1]);
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

    void PerformLabelingMem(std::vector<unsigned long>& accesses) {

        {
#undef CONDITION_KD
#undef CONDITION_LC
#undef CONDITION_LD
#undef CONDITION_MC
#undef CONDITION_NB
#undef CONDITION_ND
#undef CONDITION_OA
#undef CONDITION_OB
#undef CONDITION_OC
#undef CONDITION_OD
#undef CONDITION_PA
#undef CONDITION_PC
#undef CONDITION_QB
#undef CONDITION_RA
#undef CONDITION_RB
#undef CONDITION_SA
#undef CONDITION_TD
#undef CONDITION_UC
#undef CONDITION_UD
#undef CONDITION_VC
#undef CONDITION_WB
#undef CONDITION_WD
#undef CONDITION_XA
#undef CONDITION_XB
#undef CONDITION_XC
#undef CONDITION_XD

#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

            //Conditions:
#define CONDITION_KD c > 0 && r > 0 && s > 0 && img(s - 1, r - 1, c - 1) > 0
#define CONDITION_LC r > 0 && s > 0 && img(s - 1, r - 1, c) > 0
#define CONDITION_LD c < w - 1 && r > 0 && s > 0 && img(s - 1, r - 1, c + 1) > 0
#define CONDITION_MC c < w - 2 && r > 0 && s > 0 && img(s - 1, r - 1, c + 2) > 0
#define CONDITION_NB c > 0 && s > 0 && img(s - 1, r, c - 1) > 0
#define CONDITION_ND c > 0 && r < h - 1 && s > 0 && img(s - 1, r + 1, c - 1) > 0
#define CONDITION_OA s > 0 && img(s - 1, r, c) > 0
#define CONDITION_OB c < w - 1 && s > 0 && img(s - 1, r, c + 1) > 0
#define CONDITION_OC r < h - 1 && s > 0 && img(s - 1, r + 1, c) > 0
#define CONDITION_OD c < w - 1 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 1) > 0
#define CONDITION_PA c < w - 2 && s > 0 && img(s - 1, r, c + 2) > 0
#define CONDITION_PC c < w - 2 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 2) > 0
#define CONDITION_QB c > 0 && r < h - 2 && s > 0 && img(s - 1, r + 2, c - 1) > 0
#define CONDITION_RA r < h - 2 && s > 0 && img(s - 1, r + 2, c) > 0
#define CONDITION_RB c < w - 1 && r < h - 2 && s > 0 && img(s - 1, r + 2, c + 1) > 0
#define CONDITION_SA c < w - 2 && r < h - 2 && s > 0 && img(s - 1, r + 2, c + 2) > 0
#define CONDITION_TD c > 0 && r > 0 && img(s, r - 1, c - 1) > 0
#define CONDITION_UC r > 0 && img(s, r - 1, c) > 0
#define CONDITION_UD c < w - 1 && r > 0 && img(s, r - 1, c + 1) > 0
#define CONDITION_VC c < w - 2 && r > 0 && img(s, r - 1, c + 2) > 0
#define CONDITION_WB c > 0 && img(s, r, c - 1) > 0
#define CONDITION_WD c > 0 && r < h - 1 && img(s, r + 1, c - 1) > 0
#define CONDITION_XA img(s, r, c) > 0
#define CONDITION_XB c < w - 1 && img(s, r, c + 1) > 0
#define CONDITION_XC r < h - 1 && img(s, r + 1, c) > 0
#define CONDITION_XD c < w - 1 && r < h - 1 && img(s, r + 1, c + 1) > 0

#include "labeling3D_BBDT_26c_221b_action_definition_memory.inc.h"
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
            for (unsigned int r = 0; r < h; r += 2) {
                for (unsigned int c = 0; c < w; c += 2) {
                    if (!((CONDITION_XA) || (CONDITION_XB) || (CONDITION_XC) || (CONDITION_XD))) {
                        ACTION_0;
                    }
#include "labeling3D_BBDT_26c_221b_tree.inc.h"
                }
            } // Rows cycle end
        } // Planes cycle end
//
        // Second scan
        LabelsSolver::MemFlatten();

        int e_rows = h & 0xfffffffe;
        bool o_rows = h % 2 == 1;
        int e_cols = w & 0xfffffffe;
        bool o_cols = w % 2 == 1;

        for (unsigned s = 0; s < d; s++) {
            int r = 0;
            for (; r < e_rows; r += 2) {
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
                        if (img(s, r + 1, c) > 0)
                            img_labels(s, r + 1, c) = iLabel;
                        else
                            img_labels(s, r + 1, c) = 0;
                        if (img(s, r + 1, c + 1) > 0)
                            img_labels(s, r + 1, c + 1) = iLabel;
                        else
                            img_labels(s, r + 1, c + 1) = 0;
                    }
                    else {
                        img_labels(s, r, c) = 0;
                        img_labels(s, r, c + 1) = 0;
                        img_labels(s, r + 1, c) = 0;
                        img_labels(s, r + 1, c + 1) = 0;
                    }
                }
                // Last column if the number of columns is odd
                if (o_cols) {
                    int iLabel = img_labels(s, r, c);
                    if (iLabel > 0) {
                        iLabel = LabelsSolver::MemGetLabel(iLabel);
                        if (img(s, r, c) > 0)
                            img_labels(s, r, c) = iLabel;
                        else
                            img_labels(s, r, c) = 0;
                        if (img(s, r + 1, c) > 0)
                            img_labels(s, r + 1, c) = iLabel;
                        else
                            img_labels(s, r + 1, c) = 0;
                    }
                    else {
                        img_labels(s, r, c) = 0;
                        img_labels(s, r + 1, c) = 0;
                    }
                }
            }
            // Last row if the number of rows is odd
            if (o_rows) {
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
                    if (iLabel > 0) {
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
        accesses = std::vector<unsigned long>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (unsigned long)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (unsigned long)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (unsigned long)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        LabelsSolver::MemDealloc(); // Memory deallocation of the labels solver

        {
#undef CONDITION_KD
#undef CONDITION_LC
#undef CONDITION_LD
#undef CONDITION_MC
#undef CONDITION_NB
#undef CONDITION_ND
#undef CONDITION_OA
#undef CONDITION_OB
#undef CONDITION_OC
#undef CONDITION_OD
#undef CONDITION_PA
#undef CONDITION_PC
#undef CONDITION_QB
#undef CONDITION_RA
#undef CONDITION_RB
#undef CONDITION_SA
#undef CONDITION_TD
#undef CONDITION_UC
#undef CONDITION_UD
#undef CONDITION_VC
#undef CONDITION_WB
#undef CONDITION_WD
#undef CONDITION_XA
#undef CONDITION_XB
#undef CONDITION_XC
#undef CONDITION_XD

            //Actions:
#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

            //Conditions:
#define CONDITION_KD c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LC r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LD c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MC c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_ND c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_OC r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_OD c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_PC c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 2 && s > 0 && img_slice11_row02[c - 1] > 0
#define CONDITION_RA r < h - 2 && s > 0 && img_slice11_row02[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 2 && s > 0 && img_slice11_row02[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 2 && s > 0 && img_slice11_row02[c + 2] > 0
#define CONDITION_TD c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UC r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UD c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VC c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_WD c > 0 && r < h - 1 && img_slice00_row01[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_XC r < h - 1 && img_slice00_row01[c] > 0
#define CONDITION_XD c < w - 1 && r < h - 1 && img_slice00_row01[c + 1] > 0

//Actions:
#include "labeling3D_BBDT_26c_221b_action_definition.inc.h"
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


        for (unsigned int s = 0; s < d; s += 1) {
            for (unsigned int r = 0; r < h; r += 2) {
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);
                // T, W slice
                const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * -1);
                const unsigned char* const img_slice00_row01 = (unsigned char*)(((char*)img_slice00_row00) + img_.step.p[1] * 1);

                // K, N, Q slice
                const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * -1);
                const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 0);
                const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 1);
                const unsigned char* const img_slice11_row02 = (unsigned char*)(((char*)img_slice00_row00) - img_.step.p[0] + img_.step.p[1] * 2);


                unsigned* const img_labels_slice00_row00 = img_labels_.ptr<unsigned>(s, r);
                // T, W slice
                unsigned* const img_labels_slice00_row12 = (unsigned*)(((char*)img_labels_slice00_row00) + img_labels_.step.p[1] * -2);

                // K, N, Q slice
                unsigned* const img_labels_slice11_row12 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * -2);
                unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 0);
                unsigned* const img_labels_slice11_row02 = (unsigned*)(((char*)img_labels_slice00_row00) - img_labels_.step.p[0] + img_labels_.step.p[1] * 2);

                for (unsigned int c = 0; c < w; c += 2) {
                    if (!((CONDITION_XA) || (CONDITION_XB) || (CONDITION_XC) || (CONDITION_XD))) {
                        ACTION_0;
                    }
#include "labeling3D_BBDT_26c_221b_tree.inc.h"
                }
            } // Rows cycle end
        } // Planes cycle end
    }

    void SecondScan() {
        unsigned int d = img_.size.p[0];
        unsigned int h = img_.size.p[1];
        unsigned int w = img_.size.p[2];

        LabelsSolver::Flatten();

        const unsigned char* const img_row = img_.ptr<unsigned char>();
        int* const img_labels_row = img_labels_.ptr<int>();

        // NEW VERSION BELOW, OLD IN PerformLabeling
        int e_rows = h & 0xfffffffe;
        bool o_rows = h % 2 == 1;
        int e_cols = w & 0xfffffffe;
        bool o_cols = w % 2 == 1;

        for (unsigned s = 0; s < d; s++) {
            int r = 0;
            for (; r < e_rows; r += 2) {
                // Get rows pointer
                const unsigned char* const img_row = img_.ptr<unsigned char>(s, r);
                const unsigned char* const img_row_fol = (unsigned char*)(((char*)img_row) + img_.step.p[1]);

                unsigned* const img_labels_row = img_labels_.ptr<unsigned>(s, r);
                unsigned* const img_labels_row_fol = (unsigned*)(((char*)img_labels_row) + img_labels_.step.p[1]);
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

#undef CONDITION_KD
#undef CONDITION_LC
#undef CONDITION_LD
#undef CONDITION_MC
#undef CONDITION_NB
#undef CONDITION_ND
#undef CONDITION_OA
#undef CONDITION_OB
#undef CONDITION_OC
#undef CONDITION_OD
#undef CONDITION_PA
#undef CONDITION_PC
#undef CONDITION_QB
#undef CONDITION_RA
#undef CONDITION_RB
#undef CONDITION_SA
#undef CONDITION_TD
#undef CONDITION_UC
#undef CONDITION_UD
#undef CONDITION_VC
#undef CONDITION_WB
#undef CONDITION_WD
#undef CONDITION_XA
#undef CONDITION_XB
#undef CONDITION_XC
#undef CONDITION_XD

//Actions:
#include "labeling3D_BBDT_2829_action_undefinition.inc.h"

#endif // YACCLAB_LABELING3D_BBDT_26C_221B_H_