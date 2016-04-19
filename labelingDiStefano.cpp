#include "LabelingDiStefano.h"

using namespace cv;
using namespace std;

int labelingDiStefano(const Mat1b &img, Mat1i &imgOut) {
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

				int lp(0), lq(0), lr(0), ls(0), lx(0), lMin(INT_MAX);
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

int labelingDiStefanoOPT(const Mat1b &img, Mat1i &imgOut) {
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

				int lp(0), lq(0), lr(0), ls(0), lx(0), lMin(INT_MAX);
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