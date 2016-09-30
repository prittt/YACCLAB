// Copyright (C) 2015 - Wan-Yu Chang and Chung-Cheng Chiu
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
// 
// The "free" of library only licensed for the purposes of research and study. 
// For further information or business co-operation, please contact us at
// Wan-Yu Chang and Chung-Cheng Chiu - Chung Cheng Institute of Technology of National Defense University.
// No.75, Shiyuan Rd., Daxi Township, Taoyuan County 33551, Taiwan (R.O.C.)  - e-mail: david.cc.chiu@gmail.com 
//
// Specially thank for the help of Prof. Grana who provide his source code of the BBDT algorithm.

#include "labelingWYChang2015.h"

#include <stdint.h>

using namespace cv;
using namespace std;

int CCIT_OPT(const Mat1b& img, Mat1i& imgOut) {

    unsigned char byF = 1;

	// add image initialization with memset (in the original code it was made out of the labeling procedure but it must
	// be consider in the total ammount time request by the algorithm, like in all the other ones is done)
    imgOut = Mat1i(img.size(),0); 

    int w = imgOut.cols, h = imgOut.rows;

    int m = 1;
    int *aRTable = new int[w*h / 4];
    int *aNext = new int[w*h / 4];
    int *aTail = new int[w*h / 4];

    int lx, u, v, k;

    #define condition_b1 img_row[x]==byF
    #define condition_b2 x+1<w && img_row[x+1]==byF       // add necessary control condition
    #define condition_b3 y+1<h && img_row_fol[x]==byF     // add necessary control condition 
    #define condition_b4 x+1<w && y+1<h && img_row_fol[x+1]==byF    // add necessary control condition
    #define condition_u1 /*y-1>0 &&*/ x-1>0 && img_row_prev[x-1]==byF    // add necessary control consition
    #define condition_u2 /*y-1>0 &&*/ img_row_prev[x]==byF
    #define condition_u3 x+1<w && /*y-1>0 &&*/ img_row_prev[x+1]==byF   // add necessary control consition
    #define condition_u4 x+2<w && /*y-1>0 &&*/ img_row_prev[x+2]==byF   // add necessary control consition
    #define assign_S imgOut_row[x] = imgOut_row[x-2]
    #define assign_P imgOut_row[x] = imgOut_row_prev_prev[x-2]
    #define assign_Q imgOut_row[x] = imgOut_row_prev_prev[x]
    #define assign_R imgOut_row[x] = imgOut_row_prev_prev[x+2]
    #define newlabel imgOut_row[x] = m
    #define assign_lx imgOut_row[x] = lx
    #define load_lx u = aRTable[lx]
    #define load_Pu u = aRTable[imgOut_row_prev_prev[x-2]]
    #define load_Pv v = aRTable[imgOut_row_prev_prev[x-2]]
    #define load_Qu u = aRTable[imgOut_row_prev_prev[x]]
    #define load_Qv v = aRTable[imgOut_row_prev_prev[x]]
    #define load_Qk k = aRTable[imgOut_row_prev_prev[x]]
    #define load_Rv v = aRTable[imgOut_row_prev_prev[x+2]]
    #define load_Rk k = aRTable[imgOut_row_prev_prev[x+2]]
    #define newlabelprocess lx = newlabel; 	aRTable[m] = m;  aNext[m] = -1;  aTail[m] = m;	m = m + 1;
    #define reslove2(u, v); 		if (u<v) { int i = v; 	while (i>-1) {	aRTable[i] = u;	i = aNext[i];	}	aNext[aTail[u]] = v; aTail[u] = aTail[v]; 	}else if (u>v) {	int i = u; 	while (i>-1) { aRTable[i] = v; 	i = aNext[i]; }	aNext[aTail[v]] = u; aTail[v] = aTail[u]; };
    #define reslove3(u, v, k); 		if (u<v) { int i = v; 	while (i>-1) { 	aRTable[i] = u; i = aNext[i]; 	} 	aNext[aTail[u]] = v; aTail[u] = aTail[v];  k = aRTable[k]; if (u<k) { int i = k; 	while (i>-1) { 	aRTable[i] = u; i = aNext[i]; } aNext[aTail[u]] = k;  aTail[u] = aTail[k]; 	} else if (u>k) { int i = u;   while (i>-1) { aRTable[i] = k; i = aNext[i]; } aNext[aTail[k]] = u; 	aTail[k] = aTail[u]; } 	} else if (u>v) { int i = u; while (i>-1) { aRTable[i] = v;    i = aNext[i]; 	} 	aNext[aTail[v]] = u;  aTail[v] = aTail[u];	k = aRTable[k];	if (v<k) { int i = k; while (i>-1) { aRTable[i] = v;  i = aNext[i];	}   	   aNext[aTail[v]] = k; aTail[v] = aTail[k]; } else if (v>k) { int i = v;	while (i>-1) {	aRTable[i] = k; 	i = aNext[i]; } aNext[aTail[k]] = v; aTail[k] = aTail[v]; } }else { k = aRTable[k]; if (u<k) {	int i = k; while (i>-1) { aRTable[i] = u; i = aNext[i];	} aNext[aTail[u]] = k;	aTail[u] = aTail[k]; }else if (u>k) { int i = u;	while (i>-1) {	aRTable[i] = k;	i = aNext[i]; } aNext[aTail[k]] = u; aTail[k] = aTail[u]; }; };

    bool nextprocedure2;

    int y = 0; // extract from the first for
    const uchar* const img_row = img.ptr<uchar>(y);
    const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
    uint* const imgOut_row = imgOut.ptr<uint>(y);
    //prcess first two rows
    // cout << "." << endl;
    for (int x = 0; x<w; x += 2) {

        if (condition_b1){
            newlabelprocess;
            if (condition_b2 || condition_b4)
                nextprocedure2 = true;
            else
                nextprocedure2 = false;
        }
        else if (condition_b2){
            newlabelprocess;
            nextprocedure2 = true;
        }
        else if (condition_b3){
            newlabelprocess;
            if (condition_b4)
                nextprocedure2 = true;
            else
                nextprocedure2 = false;
        }
        else if (condition_b4){
            newlabelprocess;
            nextprocedure2 = true;
        }
        else{
            nextprocedure2 = false;
        }

        while (nextprocedure2 && x + 2<w){
            x = x + 2;

            if (condition_b1){
                assign_lx;
                if (condition_b2 || condition_b4)
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (condition_b2){

                if (condition_b3){
                    assign_lx;
                }
                else{
                    newlabelprocess;
                }
                nextprocedure2 = true;
            }
            else if (condition_b3){
                assign_lx;
                if (condition_b4)
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (condition_b4){
                newlabelprocess;
                nextprocedure2 = true;
            }
            else{
                nextprocedure2 = false;
            }

        }
    }

    // cout << "." << endl;
    for (int y = 2; y<h; y += 2) {
        const uchar* const img_row = img.ptr<uchar>(y);
        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
        const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
        uint* const imgOut_row = imgOut.ptr<uint>(y);
        uint* const imgOut_row_prev_prev = (uint *)(((char *)imgOut_row) - imgOut.step.p[0] - imgOut.step.p[0]);
        for (int x = 0; x<w; x += 2) {
            if (condition_b1){
                if (condition_b2){
                    if (condition_u2){
                        lx = assign_Q;
                        if (condition_u3){

                        }
                        else{
                            if (condition_u4){
                                load_lx;
                                load_Rv;
                                reslove2(u, v);
                            }
                        }
                    }
                    else if (condition_u3){
                        lx = assign_Q;
                        if (condition_u1){
                            load_lx;
                            load_Pv;
                            reslove2(u, v);
                        }

                    }
                    else if (condition_u1){
                        lx = assign_P;
                        if (condition_u4){
                            load_lx;
                            load_Rv;
                            reslove2(u, v);
                        }
                    }
                    else if (condition_u4){
                        lx = assign_R;
                    }
                    else{
                        newlabelprocess;
                    }
                    nextprocedure2 = true;
                }
                else{
                    if (condition_u2){
                        lx = assign_Q;
                    }
                    else if (condition_u1){
                        lx = assign_P;
                        if (condition_u3){
                            load_lx;
                            load_Qv;
                            reslove2(u, v);

                        }
                    }
                    else if (condition_u3){
                        lx = assign_Q;
                    }
                    else{
                        newlabelprocess;
                    }
                    if (condition_b4)
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;

                }
            }
            else if (condition_b2){
                if (condition_u3){
                    lx = assign_Q;
                }
                else if (condition_u2){
                    lx = assign_Q;
                    if (condition_u4){
                        load_lx;
                        load_Rv;
                        reslove2(u, v);
                    }
                }
                else if (condition_u4){
                    lx = assign_R;
                }
                else{
                    newlabelprocess;
                }
                nextprocedure2 = true;
            }
            else if (condition_b3){
                newlabelprocess;
                if (condition_b4)
                    nextprocedure2 = true;//
                else
                    nextprocedure2 = false;
            }
            else if (condition_b4){
                newlabelprocess;
                nextprocedure2 = true;
            }
            else{
                nextprocedure2 = false;
            }

            while (nextprocedure2 && x + 2<w){
                x = x + 2;
                if (condition_b1){

                    assign_lx;
                    if (condition_b2){
                        if (condition_u2){
                            if (condition_u3){
                                load_lx;
                                load_Qv;
                                reslove2(u, v);
                            }
                            else{
                                if (condition_u4){
                                    load_lx;
                                    load_Qv;
                                    load_Rk;
                                    reslove3(u, v, k);
                                }
                                else{
                                    load_lx;
                                    load_Qv;
                                    reslove2(u, v);
                                }
                            }

                        }
                        else if (condition_u3){
                            if (condition_u1){
                                load_lx;
                                load_Pv;
                                load_Qk;
                                reslove3(u, v, k);
                            }
                            else{
                                //reslove S, Q
                                load_lx;
                                load_Qv;
                                reslove2(u, v);
                            }
                        }
                        else if (condition_u1){
                            if (condition_u4){
                                load_lx;
                                load_Pv;
                                load_Rk;
                                reslove3(u, v, k);
                            }
                            else{
                                load_lx;
                                load_Pv;
                                reslove2(u, v);
                            }

                        }
                        else if (condition_u4){
                            load_lx;
                            load_Rv;
                            reslove2(u, v);
                        }
                        nextprocedure2 = true;
                    }
                    else{
                        assign_lx;
                        if (condition_u2){
                            load_lx;
                            load_Qv;
                            reslove2(u, v);
                        }
                        else if (condition_u1){
                            if (condition_u3){
                                load_lx;
                                load_Pv;
                                load_Qk;
                                reslove3(u, v, k);
                            }
                            else{
                                load_lx;
                                load_Pv;
                                reslove2(u, v);
                            }
                        }
                        else if (condition_u3){
                            load_lx;
                            load_Qv;
                            reslove2(u, v);
                        }

                        if (condition_b4)
                            nextprocedure2 = true;//
                        else
                            nextprocedure2 = false;
                    }

                }
                else if (condition_b2){
                    if (condition_b3){
                        assign_lx;
                        if (condition_u3){
                            load_lx;
                            load_Qv;
                            reslove2(u, v);
                        }
                        else if (condition_u2){
                            if (condition_u4){
                                load_lx;
                                load_Qv;
                                load_Rk;
                                reslove3(u, v, k);
                            }
                            else{
                                load_lx;
                                load_Qv;
                                reslove2(u, v);
                            }
                        }if (condition_u4){
                            load_lx;
                            load_Rv;
                            reslove2(u, v);
                        }
                    }
                    else{
                        if (condition_u3){
                            lx = assign_Q;
                        }
                        else if (condition_u2){
                            lx = assign_Q;
                            if (condition_u4){
                                load_lx;
                                load_Rv;
                                reslove2(u, v);
                            }
                        }
                        else if (condition_u4){
                            lx = assign_R;
                        }
                        else{
                            newlabelprocess;
                        }
                    }

                    nextprocedure2 = true;
                }
                else if (condition_b3){
                    assign_lx;
                    if (condition_b4)
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;
                }
                else if (condition_b4){
                    newlabelprocess;
                    nextprocedure2 = true;
                }
                else{
                    nextprocedure2 = false;
                }
            }
        }
    }
    // cout << "." << endl;
    //Renew label number
    int iCurLabel = 0;
    for (int i = 1; i<m; i++) {
        if (aRTable[i] == i) {
            iCurLabel++;
            aRTable[i] = iCurLabel;
        }
        else
            aRTable[i] = aRTable[aRTable[i]];
    }
    // cout << "." << endl;
    // SECOND SCAN 
    for (int y = 0; y<h; y += 2) {
        const uchar* const img_row = img.ptr<uchar>(y);
        const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
        uint* const imgOut_row = imgOut.ptr<uint>(y);
        uint* const imgOut_row_fol = (uint *)(((char *)imgOut_row) + imgOut.step.p[0]);
        for (int x = 0; x<w; x += 2) {
            int iLabel = imgOut_row[x];
            if (iLabel>0) {
                // cout << iLabel << "\n";
                iLabel = aRTable[iLabel];
                if (img_row[x] == byF)
                    imgOut_row[x] = iLabel;
                else
                    imgOut_row[x] = 0;
                if (x + 1<w) {
                    if (img_row[x + 1] == byF)
                        imgOut_row[x + 1] = iLabel;
                    else
                        imgOut_row[x + 1] = 0;
                    if (y + 1<h) {
                        if (img_row_fol[x] == byF)
                            imgOut_row_fol[x] = iLabel;
                        else
                            imgOut_row_fol[x] = 0;
                        if (img_row_fol[x + 1] == byF)
                            imgOut_row_fol[x + 1] = iLabel;
                        else
                            imgOut_row_fol[x + 1] = 0;
                    }
                }
                else if (y + 1<h) {
                    if (img_row_fol[x] == byF)
                        imgOut_row_fol[x] = iLabel;
                    else
                        imgOut_row_fol[x] = 0;
                }
            }
            else {
                imgOut_row[x] = 0;
                if (x + 1<w) {
                    imgOut_row[x + 1] = 0;
                    if (y + 1<h) {
                        imgOut_row_fol[x] = 0;
                        imgOut_row_fol[x + 1] = 0;
                    }
                }
                else if (y + 1<h) {
                    imgOut_row_fol[x] = 0;
                }
            }
        }
    }
    //cout << "." << endl;

    // output the number of labels
    //*numLabels = iCurLabel;
    delete[] aRTable; // add []
    delete[] aNext; // add []
    delete[] aTail; //add []
    return ++iCurLabel;
}