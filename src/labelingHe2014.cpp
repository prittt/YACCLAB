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

//#include "labelingHe2014.h"
//#include "register.h"
//
//
//using namespace cv;
//using namespace std;
//
//#define Ca 1
//#define Cb 2
//#define Cc 3
//#define Cd 4
//#define Ce 5
//#define Cf 6
//#define Cg 7
//#define Ch 8
//#define Ci 9
//#define null -1
//
//REGISTER_LABELING(CTB);
//
//void CTB::AllocateMemory()
//{
//    const size_t Plength = UPPER_BOUND_8_CONNECTIVITY;
//    P = (unsigned *)fastMalloc(sizeof(unsigned) * Plength); //array P for equivalences resolution
//    return;
//}
//
//void CTB::DeallocateMemory()
//{
//    fastFree(P);
//    return;
//}
//
//
//unsigned CTB::FirstScan()
//{
//    const int h = img_.rows;
//    const int w = img_.cols;
//
//    img_labels_ = Mat1i(img_.size(), 0);
//    P[0] = 0;	//first label is for background pixels
//    unsigned lunique = 1;
//
//    // first scan
//    for (int r = 0; r < h; r += 2)
//    {
//        int prob_fol_state = null;
//        int prev_state = null;
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(r);
//        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
//        const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(r);
//        unsigned* const imgLabels_row_prev = (unsigned *)(((char *)imgLabels_row) - img_labels_.step.p[0]);
//        unsigned* const imgLabels_row_fol = (unsigned *)(((char *)imgLabels_row) + img_labels_.step.p[0]);
//
//        for (int c = 0; c < w; c += 1)
//        {
//
//            // He et al. work with mask
//            // +--+--+--+
//            // |n1|n2|n3|
//            // +--+--+--+
//            // |n4| a|
//            // +--+--+
//            // |n5| b|
//            // +--+--+
//
//            // A bunch of defines used to check if the pixels are foreground, and current state of graph
//            // without going outside the image limits.
//
//        #define condition_a img_row[c]>0
//        #define condition_b r+1<h && img_row_fol[c]>0
//        #define condition_n1 c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
//        #define condition_n2 r-1>=0 && img_row_prev[c]>0
//        #define condition_n3 r-1>=0 && c+1<w && img_row_prev[c+1]>0
//        #define condition_n4 c-1>=0 && img_row[c-1]>0
//        #define condition_n5 c-1>=0 && r+1<h && img_row_fol[c-1]>0
//
//            switch (prev_state)
//            {
//            case(Ca):
//                //cout << "Ca" << endl;
//                // previous configuration was Ca
//                if (condition_a)
//                {
//                    // case c2 follows c1: transition a->b
//                    prev_state = Cb; // set current state as previous state for next turn
//                    if (condition_n2)
//                    {
//                        // first check on n2: it is a foreground pixel
//                        imgLabels_row[c] = imgLabels_row_prev[c];
//                        // not need to check pixel n1 and n3: if they are foreground pixel, 
//                        // they are known to be eight-connected with pixel n2 befor processing 
//                        // the current two pixel a and b; thus they should belong to the same
//                        // equivalent-label set already
//                        prob_fol_state = Cd; // probably follows state are Cd, Cg and Ca. We choose Cd as "delegate"
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            // second check on n3: it is a foreground pixel
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1];
//                            prob_fol_state = Ce;  // probably follows state are Ce, Cg and Ca. We choose Ce as "delegate"
//                            if (condition_n1)
//                            {
//                                // solve equivalence between n1 and a
//                                set_union(P, imgLabels_row_prev[c - 1], imgLabels_row[c]);
//                            }
//                        }
//                        else
//                        {
//                            // both n2 and n3 are background pixels
//                            prob_fol_state = Cf;  // probably follows state are Cf, Cg and Ca. We choose Ce as "delegate"
//                            if (condition_n1)
//                            {
//                                // third check on n1: it is a foregroun pixel
//                                imgLabels_row[c] = imgLabels_row_prev[c - 1];
//                            }
//                            else
//                            {
//                                // new label
//                                imgLabels_row[c] = lunique;
//                                P[lunique] = lunique;
//                                lunique++;
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // case c3 follows c1: transition a->c
//                        prev_state = Cc; // set previous state as current state for next turn
//                                         // new label for b
//                        imgLabels_row_fol[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                    }
//                    else
//                    {
//                        // case c1 follows c1: transition a->a
//                        // nothing to do 
//                    }
//                }
//                break;
//            case(Cb):
//                //cout << "Cb" << endl;
//                // previous configuration was Cb
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // solve equivalence between a and n3
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do 
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4 already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; //assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // all possible confiuration are Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cc):
//                //cout << "Cc" << endl;
//                // previous configuration was Cc
//                if (condition_a)
//                {
//                    // current configuration is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Ci
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1]; // assign b provisional label of n4
//                        prev_state = Ci;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do 
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cd):
//                //cout << "Cd" << endl;
//                // previous configuration was Cd
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // assign "a" provisional label of n4 alreasy done
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 alreasy done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4, already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; // assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ce):
//                //cout << "Ce" << endl;
//                // previous configuration was Ce
//                if (condition_a)
//                {
//                    // current configuration is Cd
//                    prev_state = Cd;
//                    imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                    if (condition_n2)
//                    {
//                        // assign "a" provisional label of n4 alreasy done
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 already done)
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        prev_state = Cg;
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1];//assign b provisional label of n4
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothung to do ??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cf):
//                //cout << "Cf" << endl;
//                // previous configuration was Cf
//                if (condition_a)
//                {
//                    // possible current configuration are Ce and Cf
//                    if (prob_fol_state == Ce)
//                    {
//                        // current configuration is Ce
//                        imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                        prev_state = Ce;
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Cf)
//                        {
//                            // current configuration is Cf
//                            imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                            if (condition_n3)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // nothing to do, (assign a provisional label of n4, already done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                        else
//                        {
//                            // current configuration is?? TODO posso mai entrare in questo stato?
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1];// assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothign to do??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cg):
//                //cout << "Cg" << endl;
//                if (condition_a)
//                {
//                    // current state is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        //current configuration in Ci
//                        prev_state = Ci;
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1]; // assign b provisional label of n5
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ch):
//                //cout << "Ch" << endl;
//                // previous configuration was Ch
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // assign "a" provisional label of n4 alreasy done
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 alreasy done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4, already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; // assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ci):
//                //cout << "Ci" << endl;
//                if (condition_a)
//                {
//                    // current configuration is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Ci
//                        prev_state = Ci;
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1];// assign b provisional label of n5
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(null):
//                //cout << "null" << endl;
//                // no previous configuration defined
//                if (condition_a)
//                {
//                    //a is foreground pixel
//                    prev_state = Cb;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; //assign a provisional label of n2
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            // new label for a, not need to check n1
//                            imgLabels_row[c] = lunique;
//                            P[lunique] = lunique;
//                            lunique++;
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // new label for b
//                        imgLabels_row_fol[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // nothing to do 
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            }//End switch
//        }//End columns's for
//    }//End rows's for
//
//    return lunique;
//}
//
//unsigned CTB::SecondScan(const unsigned& lunique)
//{
//    int nLabels = flattenL(P, lunique);
//
//    // second scan
//    for (int r_i = 0; r_i < img_labels_.rows; ++r_i)
//    {
//        unsigned *imgLabels_row_start = img_labels_.ptr<unsigned>(r_i);
//        unsigned *imgLabels_row_end = imgLabels_row_start + img_labels_.cols;
//        unsigned *imgLabels_row = imgLabels_row_start;
//        for (int c_i = 0; imgLabels_row != imgLabels_row_end; ++imgLabels_row, ++c_i)
//        {
//            const unsigned l = P[*imgLabels_row];
//            *imgLabels_row = l;
//        }
//    }
//
//    return nLabels;
//}
//
//unsigned CTB::PerformLabeling()
//{
//    const int h = img_.rows;
//    const int w = img_.cols;
//
//    img_labels_ = Mat1i(img_.size(), 0);
//
//    P[0] = 0;	//first label is for background pixels
//    unsigned lunique = 1;
//
//    for (int r = 0; r < h; r += 2)
//    {
//        int prob_fol_state = null;
//        int prev_state = null;
//        // Get rows pointer
//        const uchar* const img_row = img_.ptr<uchar>(r);
//        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img_.step.p[0]);
//        const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img_.step.p[0]);
//        unsigned* const imgLabels_row = img_labels_.ptr<unsigned>(r);
//        unsigned* const imgLabels_row_prev = (unsigned *)(((char *)imgLabels_row) - img_labels_.step.p[0]);
//        unsigned* const imgLabels_row_fol = (unsigned *)(((char *)imgLabels_row) + img_labels_.step.p[0]);
//
//        for (int c = 0; c < w; c += 1)
//        {
//
//            // He et al. work with mask
//            // +--+--+--+
//            // |n1|n2|n3|
//            // +--+--+--+
//            // |n4| a|
//            // +--+--+
//            // |n5| b|
//            // +--+--+
//
//            // A bunch of defines used to check if the pixels are foreground, and current state of graph
//            // without going outside the image limits.
//
//        #define condition_a img_row[c]>0
//        #define condition_b r+1<h && img_row_fol[c]>0
//        #define condition_n1 c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
//        #define condition_n2 r-1>=0 && img_row_prev[c]>0
//        #define condition_n3 r-1>=0 && c+1<w && img_row_prev[c+1]>0
//        #define condition_n4 c-1>=0 && img_row[c-1]>0
//        #define condition_n5 c-1>=0 && r+1<h && img_row_fol[c-1]>0
//
//            switch (prev_state)
//            {
//            case(Ca):
//                //cout << "Ca" << endl;
//                // previous configuration was Ca
//                if (condition_a)
//                {
//                    // case c2 follows c1: transition a->b
//                    prev_state = Cb; // set current state as previous state for next turn
//                    if (condition_n2)
//                    {
//                        // first check on n2: it is a foreground pixel
//                        imgLabels_row[c] = imgLabels_row_prev[c];
//                        // not need to check pixel n1 and n3: if they are foreground pixel, 
//                        // they are known to be eight-connected with pixel n2 befor processing 
//                        // the current two pixel a and b; thus they should belong to the same
//                        // equivalent-label set already
//                        prob_fol_state = Cd; // probably follows state are Cd, Cg and Ca. We choose Cd as "delegate"
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            // second check on n3: it is a foreground pixel
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1];
//                            prob_fol_state = Ce;  // probably follows state are Ce, Cg and Ca. We choose Ce as "delegate"
//                            if (condition_n1)
//                            {
//                                // solve equivalence between n1 and a
//                                set_union(P, imgLabels_row_prev[c - 1], imgLabels_row[c]);
//                            }
//                        }
//                        else
//                        {
//                            // both n2 and n3 are background pixels
//                            prob_fol_state = Cf;  // probably follows state are Cf, Cg and Ca. We choose Ce as "delegate"
//                            if (condition_n1)
//                            {
//                                // third check on n1: it is a foregroun pixel
//                                imgLabels_row[c] = imgLabels_row_prev[c - 1];
//                            }
//                            else
//                            {
//                                // new label
//                                imgLabels_row[c] = lunique;
//                                P[lunique] = lunique;
//                                lunique++;
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // case c3 follows c1: transition a->c
//                        prev_state = Cc; // set previous state as current state for next turn
//                                         // new label for b
//                        imgLabels_row_fol[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                    }
//                    else
//                    {
//                        // case c1 follows c1: transition a->a
//                        // nothing to do 
//                    }
//                }
//                break;
//            case(Cb):
//                //cout << "Cb" << endl;
//                // previous configuration was Cb
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // solve equivalence between a and n3
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do 
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4 already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; //assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // all possible confiuration are Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cc):
//                //cout << "Cc" << endl;
//                // previous configuration was Cc
//                if (condition_a)
//                {
//                    // current configuration is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Ci
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1]; // assign b provisional label of n4
//                        prev_state = Ci;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do 
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cd):
//                //cout << "Cd" << endl;
//                // previous configuration was Cd
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // assign "a" provisional label of n4 alreasy done
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 alreasy done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4, already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; // assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ce):
//                //cout << "Ce" << endl;
//                // previous configuration was Ce
//                if (condition_a)
//                {
//                    // current configuration is Cd
//                    prev_state = Cd;
//                    imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                    if (condition_n2)
//                    {
//                        // assign "a" provisional label of n4 alreasy done
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 already done)
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        prev_state = Cg;
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1];//assign b provisional label of n4
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothung to do ??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cf):
//                //cout << "Cf" << endl;
//                // previous configuration was Cf
//                if (condition_a)
//                {
//                    // possible current configuration are Ce and Cf
//                    if (prob_fol_state == Ce)
//                    {
//                        // current configuration is Ce
//                        imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                        prev_state = Ce;
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Cf)
//                        {
//                            // current configuration is Cf
//                            imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                            if (condition_n3)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // nothing to do, (assign a provisional label of n4, already done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                        else
//                        {
//                            // current configuration is?? TODO posso mai entrare in questo stato?
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1];// assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothign to do??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Cg):
//                //cout << "Cg" << endl;
//                if (condition_a)
//                {
//                    // current state is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        //current configuration in Ci
//                        prev_state = Ci;
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1]; // assign b provisional label of n5
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do??
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ch):
//                //cout << "Ch" << endl;
//                // previous configuration was Ch
//                if (condition_a)
//                {
//                    // all possible configuration are Cd, Ce, Cf
//                    if (prob_fol_state == Cd)
//                    {
//                        // current configuration is Cd
//                        prev_state = Cd;
//                        imgLabels_row[c] = imgLabels_row[c - 1]; // in all cases
//                        if (condition_n2)
//                        {
//                            // assign "a" provisional label of n4 alreasy done
//                            prob_fol_state = Cd;
//                        }
//                        else
//                        {
//                            if (condition_n3)
//                            {
//                                // assign "a" provisional label of n4 alreasy done
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]); // solve equivalence between a and n3
//                                prob_fol_state = Ce;
//                            }
//                            else
//                            {
//                                // both n2 and n3 background, nothing to do , (assign "a" provisional label of n4 alreasy done)
//                                prob_fol_state = Cf;
//                            }
//                        }
//                    }
//                    else
//                    {
//                        if (prob_fol_state == Ce)
//                        {
//                            // current configuration is Ce
//                            imgLabels_row[c] = imgLabels_row[c - 1];  // assign a provisional label of n4
//                            prev_state = Ce;
//                        }
//                        else
//                        {
//                            if (prob_fol_state == Cf)
//                            {
//                                // current configuration is Cf
//                                imgLabels_row[c] = imgLabels_row[c - 1]; // assign a provisional label of n4
//                                if (condition_n3)
//                                {
//                                    set_union(P, imgLabels_row[c], imgLabels_row_prev[c + 1]);
//                                    prob_fol_state = Ce;
//                                }
//                                else
//                                {
//                                    // nothing to do, (assign a provisional label of n4, already done)
//                                    prob_fol_state = Cf;
//                                }
//                            }
//                            else
//                            {
//                                // current configuration is?? TODO posso mai entrare in questo stato?
//                            }
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Cg
//                        imgLabels_row_fol[c] = imgLabels_row[c - 1]; // assign b provisional label of n4
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(Ci):
//                //cout << "Ci" << endl;
//                if (condition_a)
//                {
//                    // current configuration is Ch
//                    prev_state = Ch;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; // assign a provisional label of n2
//                        set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            set_union(P, imgLabels_row[c], imgLabels_row_fol[c - 1]); // solve equivalence between a and n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            imgLabels_row[c] = imgLabels_row_fol[c - 1]; // assign a provisional label of n5
//                            if (condition_n1)
//                            {
//                                set_union(P, imgLabels_row[c], imgLabels_row_prev[c - 1]); // solve equivalence between a and n1
//                            }
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // current configuration is Ci
//                        prev_state = Ci;
//                        imgLabels_row_fol[c] = imgLabels_row_fol[c - 1];// assign b provisional label of n5
//                    }
//                    else
//                    {
//                        // current configuration is Ca, nothing to do
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            case(null):
//                //cout << "null" << endl;
//                // no previous configuration defined
//                if (condition_a)
//                {
//                    //a is foreground pixel
//                    prev_state = Cb;
//                    if (condition_n2)
//                    {
//                        imgLabels_row[c] = imgLabels_row_prev[c]; //assign a provisional label of n2
//                        prob_fol_state = Cd;
//                    }
//                    else
//                    {
//                        if (condition_n3)
//                        {
//                            imgLabels_row[c] = imgLabels_row_prev[c + 1]; // assign a provisional label of n3
//                            prob_fol_state = Ce;
//                        }
//                        else
//                        {
//                            // new label for a, not need to check n1
//                            imgLabels_row[c] = lunique;
//                            P[lunique] = lunique;
//                            lunique++;
//                            prob_fol_state = Cf;
//                        }
//                    }
//                    if (condition_b)
//                    {
//                        // set also label of pixel b = a
//                        imgLabels_row_fol[c] = imgLabels_row[c];
//                    }
//                }
//                else
//                {
//                    if (condition_b)
//                    {
//                        // new label for b
//                        imgLabels_row_fol[c] = lunique;
//                        P[lunique] = lunique;
//                        lunique++;
//                        prev_state = Cg;
//                    }
//                    else
//                    {
//                        // nothing to do 
//                        prev_state = Ca;
//                    }
//                }
//                break;
//            }//End switch
//        }//End columns's for
//    }//End rows's for
//
//    unsigned nLabel = flattenL(P, lunique);
//
//    // second scan
//    for (int r_i = 0; r_i < img_labels_.rows; ++r_i)
//    {
//        unsigned *imgLabels_row_start = img_labels_.ptr<unsigned>(r_i);
//        unsigned *imgLabels_row_end = imgLabels_row_start + img_labels_.cols;
//        unsigned *imgLabels_row = imgLabels_row_start;
//        for (int c_i = 0; imgLabels_row != imgLabels_row_end; ++imgLabels_row, ++c_i)
//        {
//            const unsigned l = P[*imgLabels_row];
//            *imgLabels_row = l;
//        }
//    }
//
//    return nLabel;
//}
//
