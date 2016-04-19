#include "labelingLacassagne2011.h"
#include <assert.h>
#include <numeric>

using namespace std; 
using namespace cv; 

int LSL_STD(const Mat1b& img, Mat1i& labels) {
    int rows = img.rows, cols = img.cols;
    // Step 1
    Mat1i ER(rows, cols); // matrix of relative label (1 label/pixel)
    Mat1i RLC(rows, cols); // matrix of run lenghts (up to 1 run/pixel)
    vector<int> ner(rows); // number of runs
    for (int r = 0; r < rows; ++r) {
        int x0;
        int x1 = 0; // previous value of X
        int f = 0; // front detection
        int b = 0; // right border compensation
        int er = 0;
        for (int c = 0; c < cols; ++c) {
            x0 = img(r, c) > 0;
            f = x0 ^ x1;
            RLC(r, er) = c - b;
            b = b ^ f;
            er = er + f;
            ER(r, c) = er;
            x1 = x0;
        }
        x0 = 0;
        f = x0 ^ x1;
        RLC(r, er) = cols - b;
        er = er + f;
        ner[r] = er;
    }

    // Step 2
    Mat1i ERA(rows, cols, 0); // relative to absolute label mapping
    vector<int> EQ(rows*cols / 4); // equivalence table (maximum number of labels is 1 every 4 pixels, on a regular grid)
    iota(begin(EQ), end(EQ), 0);
    int nea = 0;
    for (int r = 0; r < rows; ++r) {
        for (int er = 1; er <= ner[r]; er += 2) {
            int j0 = RLC(r, er - 1);
            int j1 = RLC(r, er);
            // check extension in case of 8-connect algorithm
            if (j0 > 0)
                j0--;
            if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1" 
                j1++;
            int er0 = r == 0 ? 0 : ER(r - 1, j0);
            int er1 = r == 0 ? 0 : ER(r - 1, j1);
            // check label parity: segments are odd
            if (er0 % 2 == 0)
                er0++;
            if (er1 % 2 == 0)
                er1--;
            if (er1 >= er0) {
                assert(r > 0);
                int ea = ERA(r - 1, er0);
                int a = EQ[ea];
                for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2" 
                    int eak = ERA(r - 1, erk);
                    int ak = EQ[eak];
                    // min extraction and propagation
                    if (a < ak){
                        //EQ[ak] = a; // Added
                        //EQ[eak] = a;
                        EQ[eak] = a;
                    }
                    else {
                        //EQ[a] = ak; // Added
                        a = ak;
                        EQ[ea] = a;
                        ea = eak;
                    }
                }
                ERA(r, er) = a; // the global min
            }
            else {
                // new label
                nea++;
                ERA(r, er) = nea;
            }
        }
    }

    // Step 3
    //Mat1i EA(rows, cols);
    //for (int r = 0; r < rows; ++r) {
    //	for (int c = 0; c < cols; ++c) {
    //		EA(r, c) = ERA(r, ER(r, c));
    //	}
    //}
    // Sorry, but I really don't get why this shouldn't be included in the last step

    // Step 4
    vector<int> A(EQ.size());
    int na = 0;
    for (int e = 1; e <= nea; ++e) {
        if (EQ[e] != e)
            A[e] = A[EQ[e]];
        else {
            na++;
            A[e] = na;
        }
    }

    // Step 5
    labels = Mat1i(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            //labels(r, c) = A[EA(r, c)];
            labels(r, c) = A[ERA(r, ER(r, c))]; // This is Step 3 and 5 together
        }
    }

    return ++na;
}

int LSL_STD_OPT(const Mat1b& img, Mat1i& labels) {
	int rows = img.rows, cols = img.cols;
	// Step 1
	Mat1i ER(rows, cols);  // matrix of relative label (1 label/pixel)
	Mat1i RLC(rows, cols); // matrix of run lenghts (up to 1 run/pixel)
	vector<int> ner(rows); // number of runs
	for (int r = 0; r < rows; ++r) {
        // Get pointers to rows
        const uchar* img_r = img.ptr<uchar>(r);
        uint* ER_r = ER.ptr<uint>(r);
        uint* RLC_r = RLC.ptr<uint>(r);
		int x0;
		int x1 = 0; // previous value of X
		int f = 0;  // front detection
		int b = 0;  // right border compensation
		int er = 0;
		for (int c = 0; c < cols; ++c) {
            x0 = img_r[c] > 0;
			f = x0 ^ x1;
			RLC_r[er] = c - b;
			b = b ^ f;
			er = er + f;
			ER_r[c] = er;
			x1 = x0;
		}
		x0 = 0;
		f = x0 ^ x1;
		RLC_r[er] = cols - b;
		er = er + f;
		ner[r] = er;
	}

	// Step 2
	Mat1i ERA(rows, cols, 0); // relative to absolute label mapping (up to 1 label every 2 pixels)
	vector<int> EQ(rows*cols / 4); // equivalence table (maximum number of labels is 1 every 4 pixels, on a regular grid)
	iota(begin(EQ), end(EQ), 0);
	int nea = 0;
	for (int r = 0; r < rows; ++r) {
        // get pointers to rows
        uint* ERA_r = ERA.ptr<uint>(r);
        const uint* ERA_r_prev = (uint *)(((char *)ERA_r) - ERA.step.p[0]);;         
        const uint* RLC_r = RLC.ptr<uint>(r);
		for (int er = 1; er <= ner[r]; er += 2) {
			int j0 = RLC_r[er - 1];
			int j1 = RLC_r[er];
			// check extension in case of 8-connect algorithm
			if (j0 > 0)
				j0--;
			if (j1 < cols - 1) // WRONG in the paper! "n-1" should be "w-1" 
				j1++;
			int er0 = r == 0 ? 0 : ER(r - 1, j0);
			int er1 = r == 0 ? 0 : ER(r - 1, j1);
			// check label parity: segments are odd
			if (er0 % 2 == 0)
				er0++;
			if (er1 % 2 == 0)
				er1--;
			if (er1 >= er0) {
				assert(r > 0);
				int ea = ERA_r_prev[er0];
				int a = EQ[ea];
				for (int erk = er0 + 2; erk <= er1; erk += 2) { // WRONG in the paper! missing "step 2" 
					int eak = ERA_r_prev[erk];
					int ak = EQ[eak];
					// min extraction and propagation
					if (a < ak)
						EQ[eak] = a;
					else {
						a = ak;
						EQ[ea] = a;
						ea = eak;
					}
				}
				ERA_r[er] = a; // the global min
			}
            else {
                // new label
                nea++;
                ERA_r[er] = nea;
			}
		}
	}

	// Step 3
	//Mat1i EA(rows, cols);
	//for (int r = 0; r < rows; ++r) {
	//	for (int c = 0; c < cols; ++c) {
	//		EA(r, c) = ERA(r, ER(r, c));
	//	}
	//}
	// Sorry, but we really don't get why this shouldn't be included in the last step

	// Step 4
	vector<int> A(EQ.size());
	int na = 0;
	for (int e = 1; e <= nea; ++e) {
		if (EQ[e] != e)
			A[e] = A[EQ[e]];
		else {
			na++;
			A[e] = na;
		}
	}

	// Step 5
	labels = Mat1i(rows, cols);
	for (int r = 0; r < rows; ++r) {
        // get pointers to rows
        uint* labels_r = labels.ptr<uint>(r);
        const uint* ERA_r = ERA.ptr<uint>(r); 
        const uint* ER_r = ER.ptr<uint>(r); 
        for (int c = 0; c < cols; ++c) {
            //labels(r, c) = A[EA(r, c)];
            auto a = ER_r[c];
            auto b = ERA_r[ER_r[c]];
            labels_r[c] = A[ERA_r[ER_r[c]]]; // This is Step 3 and 5 together
		}
	}

	return ++na; // Background's label
}
