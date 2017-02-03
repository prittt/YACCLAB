// Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#include "labelingDiStefano1999.h"

using namespace cv;
using namespace std;

int DiStefano(const Mat1b &img, Mat1i &imgOut) {
	imgOut = Mat1i(img.size());

	int iNewLabel(0);
	// p q r		  p
	// s x			q x
	// lp,lq,lx: labels assigned to p,q,x
	// FIRST SCAN:
	int *aClass = new int[img.rows*img.cols / 4];
	bool *aSingle = new bool[img.rows*img.cols / 4];
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img(y, x)) {

                int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
				if (y > 0) {
					if (x > 0)
						lp = imgOut(y - 1, x - 1);
					lq = imgOut(y - 1, x);
					if (x < img.cols - 1)
						lr = imgOut(y - 1, x + 1);
				}
				if (x > 0)
					ls = imgOut(y, x - 1);

				// if everything around is background
				if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
					lx = ++iNewLabel;
					aClass[lx] = lx;
					aSingle[lx] = true;
				}
				else {
					// p
					lx = lp;
					// q
					if (lx == 0)
						lx = lq;
					// r
					if (lx > 0) {
						if (lr > 0 && aClass[lx] != aClass[lr]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[lr];
								aSingle[aClass[lr]] = false;
							}
							else if (aSingle[aClass[lr]]) {
								aClass[lr] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[lr];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = lr;
					// s
					if (lx > 0) {
						if (ls > 0 && aClass[lx] != aClass[ls]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[ls];
								aSingle[aClass[ls]] = false;
							}
							else if (aSingle[aClass[ls]]) {
								aClass[ls] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[ls];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = ls;
				}

				imgOut(y, x) = lx;
			}
			else
				imgOut(y, x) = 0;
		}
	}

	// Renumbering of labels
	int *aRenum = new int[iNewLabel + 1];
	int iCurLabel = 0;
	for (int k = 1; k <= iNewLabel; k++) {
		if (aClass[k] == k) {
			iCurLabel++;
			aRenum[k] = iCurLabel;
		}
	}
	for (int k = 1; k <= iNewLabel; k++)
		aClass[k] = aRenum[aClass[k]];

	// SECOND SCAN 
	for (int y = 0; y < imgOut.rows; y++) {
		for (int x = 0; x < imgOut.cols; x++) {
			int iLabel = imgOut(y, x);
			if (iLabel > 0)
				imgOut(y, x) = aClass[iLabel];
		}
	}

	delete[] aClass;
	delete[] aSingle;
	delete[] aRenum;
	return iCurLabel + 1;
}

int DiStefanoOPT(const Mat1b &img, Mat1i &imgOut) {
	imgOut = Mat1i(img.size());

	int iNewLabel(0);
	// p q r		  p
	// s x			q x
	// lp,lq,lx: labels assigned to p,q,x
	// FIRST SCAN:
	int *aClass = new int[img.rows*img.cols / 4];
	bool *aSingle = new bool[img.rows*img.cols / 4];
	for (int y = 0; y < img.rows; y++) {

		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(y);
		uint* const imgOut_row = imgOut.ptr<uint>(y);
		uint* const imgOut_row_prev = (uint *)(((char *)imgOut_row) - imgOut.step.p[0]);

		for (int x = 0; x < img.cols; x++) {
			if (img_row[x]) {

                int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
				if (y > 0) {
					if (x > 0)
						lp = imgOut_row_prev[x - 1];
					lq = imgOut_row_prev[x];
					if (x < img.cols - 1)
						lr = imgOut_row_prev[x + 1];
				}
				if (x > 0)
					ls = imgOut_row[x - 1];

				// if everything around is background
				if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
					lx = ++iNewLabel;
					aClass[lx] = lx;
					aSingle[lx] = true;
				}
				else {
					// p
					lx = lp;
					// q
					if (lx == 0)
						lx = lq;
					// r
					if (lx > 0) {
						if (lr > 0 && aClass[lx] != aClass[lr]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[lr];
								aSingle[aClass[lr]] = false;
							}
							else if (aSingle[aClass[lr]]) {
								aClass[lr] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[lr];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = lr;
					// s
					if (lx > 0) {
						if (ls > 0 && aClass[lx] != aClass[ls]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[ls];
								aSingle[aClass[ls]] = false;
							}
							else if (aSingle[aClass[ls]]) {
								aClass[ls] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[ls];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = ls;
				}

				imgOut_row[x] = lx;
			}
			else
				imgOut_row[x] = 0;
		}
	}

	// Renumbering of labels
	int *aRenum = new int[iNewLabel + 1];
	int iCurLabel = 0;
	for (int k = 1; k <= iNewLabel; k++) {
		if (aClass[k] == k) {
			iCurLabel++;
			aRenum[k] = iCurLabel;
		}
	}
	for (int k = 1; k <= iNewLabel; k++)
		aClass[k] = aRenum[aClass[k]];

	// SECOND SCAN 
	for (int y = 0; y < imgOut.rows; y++) {

		// Get rows pointer
		uint* const imgOut_row = imgOut.ptr<uint>(y);

		for (int x = 0; x < imgOut.cols; x++) {
			int iLabel = imgOut_row[x];
			if (iLabel > 0)
				imgOut_row[x] = aClass[iLabel];
		}
	}

	delete[] aClass;
	delete[] aSingle;
	delete[] aRenum;
	return iCurLabel + 1;
}

int DiStefanoMEM(const Mat1b &img_origin, vector<unsigned long int> &accesses){
	
	memMat<uchar> img(img_origin); 
	memMat<int> imgOut(img_origin.size());

	int iNewLabel(0);
	// p q r		  p
	// s x			q x
	// lp,lq,lx: labels assigned to p,q,x
	// FIRST SCAN:
	memVector<int> aClass(img_origin.rows*img_origin.cols / 4);
	memVector<char> aSingle(img_origin.rows*img_origin.cols / 4);
	for (int y = 0; y < img_origin.rows; y++) {
		for (int x = 0; x < img_origin.cols; x++) {
			if (img(y, x)) {

				int lp(0), lq(0), lr(0), ls(0), lx(0); // lMin(INT_MAX);
				if (y > 0) {
					if (x > 0)
						lp = imgOut(y - 1, x - 1);
					lq = imgOut(y - 1, x);
					if (x < img.cols - 1)
						lr = imgOut(y - 1, x + 1);
				}
				if (x > 0)
					ls = imgOut(y, x - 1);

				// if everything around is background
				if (lp == 0 && lq == 0 && lr == 0 && ls == 0) {
					lx = ++iNewLabel;
					aClass[lx] = lx;
					aSingle[lx] = true;
				}
				else {
					// p
					lx = lp;
					// q
					if (lx == 0)
						lx = lq;
					// r
					if (lx > 0) {
						if (lr > 0 && aClass[lx] != aClass[lr]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[lr];
								aSingle[aClass[lr]] = false;
							}
							else if (aSingle[aClass[lr]]) {
								aClass[lr] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[lr];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = lr;
					// s
					if (lx > 0) {
						if (ls > 0 && aClass[lx] != aClass[ls]) {
							if (aSingle[aClass[lx]]) {
								aClass[lx] = aClass[ls];
								aSingle[aClass[ls]] = false;
							}
							else if (aSingle[aClass[ls]]) {
								aClass[ls] = aClass[lx];
								aSingle[aClass[lx]] = false;
							}
							else {
								int iClass = aClass[ls];
								for (int k = 1; k <= iNewLabel; k++) {
									if (aClass[k] == iClass) {
										aClass[k] = aClass[lx];
									}
								}
							}
						}
					}
					else
						lx = ls;
				}

				imgOut(y, x) = lx;
			}
			else
				imgOut(y, x) = 0;
		}
	}

	// Renumbering of labels
	memVector<int> aRenum(iNewLabel + 1);
	int iCurLabel = 0;
	for (int k = 1; k <= iNewLabel; k++) {
		if (aClass[k] == k) {
			iCurLabel++;
			aRenum[k] = iCurLabel;
		}
	}
	for (int k = 1; k <= iNewLabel; k++)
		aClass[k] = aRenum[aClass[k]];

	// SECOND SCAN 
	for (int y = 0; y < imgOut.rows; y++) {
		for (int x = 0; x < imgOut.cols; x++) {
			int iLabel = imgOut(y, x);
			if (iLabel > 0)
				imgOut(y, x) = aClass[iLabel];
		}
	}

	// Store total accesses in the output vector 'accesses'
	accesses = vector<unsigned long int>((int)MD_SIZE, 0);

	accesses[MD_BINARY_MAT] = (unsigned long int)img.getTotalAcesses();
	accesses[MD_LABELED_MAT] = (unsigned long int)imgOut.getTotalAcesses();
	accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)(aClass.getTotalAcesses() + aSingle.getTotalAcesses());
	accesses[MD_OTHER] = (unsigned long int)(aRenum.getTotalAcesses());

	//a = imgOut.getImage(); 

	return iCurLabel + 1;
}