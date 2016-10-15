// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#include "labelingGrana2016.h"

using namespace cv;
using namespace std;


inline static
void firstScan(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

#define condition_x img(r,c)>0
#define condition_p img(r-1,c-1)>0
#define condition_q img(r-1,c)>0
#define condition_r img(r-1,c+1)>0

	{
		int r = 0;
		int c = -1;
	tree_A0: if (++c >= w) goto break_A0;
		if (condition_x) {
			// x = new label
			imgLabels(r, c) = lunique;
			P[lunique] = lunique;
			lunique++;
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	tree_B0: if (++c >= w) goto break_B0;
		if (condition_x) {
			imgLabels(r, c) = imgLabels(r , c - 1); // x = s
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	break_A0:
	break_B0 : ;
	}

	for (int r = 1; r < h; ++r) {
		// First column
		int c = 0;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels(r,c) = imgLabels(r - 1, c + 1); // x = r
					goto tree_B;
				}
				else {
					// x = new label
					imgLabels(r,c) = lunique;
					P[lunique] = lunique;
					lunique++;
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}

	tree_A: if (++c >= w - 1) goto break_A;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c + 1), (uint)imgLabels(r, c - 1)); // x = r + s
					goto tree_B;
				}
				else {
					imgLabels(r,c) = imgLabels(r,c - 1); // x = s
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_B: if (++c >= w - 1) goto break_B;
		if (condition_x) {
			imgLabels(r,c) = imgLabels(r - 1, c); // x = q
			goto tree_A;
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_C: if (++c >= w - 1) goto break_C;
		if (condition_x) {
			if (condition_r) {
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c + 1), (uint)imgLabels(r, c - 1)); // x = r + s
				goto tree_B;
			}
			else {
				imgLabels(r,c) = imgLabels(r,c - 1); // x = s
				goto tree_C;
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_D: if (++c >= w - 1) goto break_D;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r,c) = imgLabels(r-1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					if (condition_p) {
						imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c - 1), (uint)imgLabels(r - 1, c + 1)); // x = p + r
						goto tree_B;
					}
					else {
						imgLabels(r,c) = imgLabels(r - 1, c + 1); // x = r
						goto tree_B;
					}
				}
				else {
					if (condition_p) {
						imgLabels(r, c) = imgLabels(r - 1, c - 1); // x = p
						goto tree_C;
					}
					else {
						// x = new label
						imgLabels(r,c) = lunique;
						P[lunique] = lunique;
						lunique++;
						goto tree_C;
					}
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}


		// Last column
	break_A:
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
			}
			else {
				imgLabels(r,c) = imgLabels(r,c - 1); // x = s
			}
		}
		continue;
	break_B:
		if (condition_x) {
			imgLabels(r,c) = imgLabels(r - 1, c); // x = q
		}
		continue;
	break_C:
		if (condition_x) {
			imgLabels(r,c) = imgLabels(r,c - 1); // x = s
		}
		continue;
	break_D:
		if (condition_x) {
			if (condition_q) {
				imgLabels(r,c) = imgLabels(r - 1, c); // x = q
			}
			else {
				if (condition_p) {
					imgLabels(r, c) = imgLabels(r - 1, c - 1); // x = p
				}
				else {
					// x = new label
					imgLabels(r,c) = lunique;
					P[lunique] = lunique;
					lunique++;
				}
			}
		}
	}//End rows's for

#undef condition_x
#undef condition_p
#undef condition_q
#undef condition_r
}

int PRED(const Mat1b &img, Mat1i &imgLabels) {

	imgLabels = cv::Mat1i(img.size(), 0); // memset is used
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	vector<uint> P(Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

	firstScan(img, imgLabels, P.data(), lunique);

	uint nLabel = flattenL(P.data(), lunique);

	// second scan
	for (int r_i = 0; r_i < imgLabels.rows; ++r_i) {
		for (int c_i = 0; c_i < imgLabels.cols; ++c_i){
			imgLabels(r_i, c_i) = P[imgLabels(r_i, c_i)];
		}
	}

	return nLabel;
}

inline static
void firstScan_OPT(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

#define condition_x img_row[c]>0
#define condition_p img_row_prev[c-1]>0
#define condition_q img_row_prev[c]>0
#define condition_r img_row_prev[c+1]>0

	{
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(0);
		uint* const imgLabels_row = imgLabels.ptr<uint>(0);

		int c = -1;
	tree_A0: if (++c >= w) goto break_A0;
		if (condition_x) {
			// x = new label
			imgLabels_row[c] = lunique;
			P[lunique] = lunique;
			lunique++;
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	tree_B0: if (++c >= w) goto break_B0;
		if (condition_x) {
			imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	break_A0:
	break_B0: ;
	}

    for (int r = 1; r < h; ++r) {
        // Get rows pointer
        const uchar* const img_row = img.ptr<uchar>(r);
        const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
        uint* const imgLabels_row = imgLabels.ptr<uint>(r);
        uint* const imgLabels_row_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0]);

		// First column
		int c = 0;
		if (condition_x) {
			if (condition_q) {
				imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
					goto tree_B;
				}
				else {
					// x = new label
					imgLabels_row[c] = lunique;
					P[lunique] = lunique;
					lunique++;
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}

	tree_A: if (++c >= w - 1) goto break_A;
		if (condition_x) {
			if (condition_q) {
				imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
					goto tree_B;
				}
				else {
					imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_B: if (++c >= w - 1) goto break_B;
		if (condition_x) {
			imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
			goto tree_A;
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_C: if (++c >= w - 1) goto break_C;
		if (condition_x) {
			if (condition_r) {
				imgLabels_row[c] = set_union(P, imgLabels_row_prev[c + 1], imgLabels_row[c - 1]); // x = r + s
				goto tree_B;
			}
			else {
				imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
				goto tree_C;
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_D: if (++c >= w - 1) goto break_D;
		if (condition_x) {
			if (condition_q) {
				imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					if (condition_p) {
						imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]); // x = p + r
						goto tree_B;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev[c + 1]; // x = r
						goto tree_B;
					}
				}
				else {
					if (condition_p) {
						imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
						goto tree_C;
					}
					else {
						// x = new label
						imgLabels_row[c] = lunique;
						P[lunique] = lunique;
						lunique++;
						goto tree_C;
					}
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}


		// Last column
	break_A: 
		if (condition_x) {
			if (condition_q) {
				imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
			}
			else {
				imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
			}
		}
		continue;
	break_B: 
		if (condition_x) {
			imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
		}
		continue;
	break_C:
		if (condition_x) {
			imgLabels_row[c] = imgLabels_row[c - 1]; // x = s
		}
		continue;
	break_D:
		if (condition_x) {
			if (condition_q) {
				imgLabels_row[c] = imgLabels_row_prev[c]; // x = q
			}
			else {
				if (condition_p) {
					imgLabels_row[c] = imgLabels_row_prev[c - 1]; // x = p
				}
				else {
					// x = new label
					imgLabels_row[c] = lunique;
					P[lunique] = lunique;
					lunique++;
				}
			}
		}
    }//End rows's for

#undef condition_x
#undef condition_p
#undef condition_q
#undef condition_r
}

int PRED_OPT(const Mat1b &img, Mat1i &imgLabels) {
	
    imgLabels = cv::Mat1i(img.size(),0); // memset is used
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

    firstScan_OPT(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	// second scan
    for (int r_i = 0; r_i < imgLabels.rows; ++r_i) {
        uint *b = imgLabels.ptr<uint>(r_i);
		uint *e = b + imgLabels.cols;
		for (; b != e; ++b){
			*b = P[*b];
        }
    }

	fastFree(P);
	return nLabel;
}

inline static
void firstScan_MEM(memMat<uchar> &img, memMat<int> imgLabels, memVector<uint> &P, uint &lunique) {
	int w(img.cols), h(img.rows);

#define condition_x img(r,c)>0
#define condition_p img(r-1,c-1)>0
#define condition_q img(r-1,c)>0
#define condition_r img(r-1,c+1)>0

	{
		int r = 0;
		int c = -1;
	tree_A0: if (++c >= w) goto break_A0;
		if (condition_x) {
			// x = new label
			imgLabels(r, c) = lunique;
			P[lunique] = lunique;
			lunique++;
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	tree_B0: if (++c >= w) goto break_B0;
		if (condition_x) {
			imgLabels(r, c) = imgLabels(r, c - 1); // x = s
			goto tree_B0;
		}
		else {
			// nothing
			goto tree_A0;
		}
	break_A0:
	break_B0 : ;
	}

	for (int r = 1; r < h; ++r) {
		// First column
		int c = 0;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels(r, c) = imgLabels(r - 1, c + 1); // x = r
					goto tree_B;
				}
				else {
					// x = new label
					imgLabels(r, c) = lunique;
					P[lunique] = lunique;
					lunique++;
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}

	tree_A: if (++c >= w - 1) goto break_A;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c + 1), (uint)imgLabels(r, c - 1)); // x = r + s
					goto tree_B;
				}
				else {
					imgLabels(r, c) = imgLabels(r, c - 1); // x = s
					goto tree_C;
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_B: if (++c >= w - 1) goto break_B;
		if (condition_x) {
			imgLabels(r, c) = imgLabels(r - 1, c); // x = q
			goto tree_A;
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_C: if (++c >= w - 1) goto break_C;
		if (condition_x) {
			if (condition_r) {
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c + 1), (uint)imgLabels(r, c - 1)); // x = r + s
				goto tree_B;
			}
			else {
				imgLabels(r, c) = imgLabels(r, c - 1); // x = s
				goto tree_C;
			}
		}
		else {
			// nothing
			goto tree_D;
		}
	tree_D: if (++c >= w - 1) goto break_D;
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
				goto tree_A;
			}
			else {
				if (condition_r) {
					if (condition_p) {
						imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 1, c - 1), (uint)imgLabels(r - 1, c + 1)); // x = p + r
						goto tree_B;
					}
					else {
						imgLabels(r, c) = imgLabels(r - 1, c + 1); // x = r
						goto tree_B;
					}
				}
				else {
					if (condition_p) {
						imgLabels(r, c) = imgLabels(r - 1, c - 1); // x = p
						goto tree_C;
					}
					else {
						// x = new label
						imgLabels(r, c) = lunique;
						P[lunique] = lunique;
						lunique++;
						goto tree_C;
					}
				}
			}
		}
		else {
			// nothing
			goto tree_D;
		}


		// Last column
	break_A:
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
			}
			else {
				imgLabels(r, c) = imgLabels(r, c - 1); // x = s
			}
		}
		continue;
	break_B:
		if (condition_x) {
			imgLabels(r, c) = imgLabels(r - 1, c); // x = q
		}
		continue;
	break_C:
		if (condition_x) {
			imgLabels(r, c) = imgLabels(r, c - 1); // x = s
		}
		continue;
	break_D:
		if (condition_x) {
			if (condition_q) {
				imgLabels(r, c) = imgLabels(r - 1, c); // x = q
			}
			else {
				if (condition_p) {
					imgLabels(r, c) = imgLabels(r - 1, c - 1); // x = p
				}
				else {
					// x = new label
					imgLabels(r, c) = lunique;
					P[lunique] = lunique;
					lunique++;
				}
			}
		}
	}//End rows's for

#undef condition_x
#undef condition_p
#undef condition_q
#undef condition_r
}

int PRED_MEM(const Mat1b &img_origin, vector<unsigned long int> &accesses) {

	memMat<uchar> img(img_origin); 
	memMat<int> imgLabels(img_origin.size(), 0); // memset is used
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img_origin.rows*img_origin.cols / 4;
	//Tree of labels
	memVector<uint> P(Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

	firstScan_MEM(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	// second scan
	for (int r_i = 0; r_i < imgLabels.rows; ++r_i) {
		for (int c_i = 0; c_i < imgLabels.cols; ++c_i){
			imgLabels(r_i, c_i) = P[imgLabels(r_i, c_i)];
		}
	}

	// Store total accesses in the output vector 'accesses'
	accesses = vector<unsigned long int>((int)MD_SIZE, 0);

	accesses[MD_BINARY_MAT] = (unsigned long int)img.getTotalAcesses();
	accesses[MD_LABELED_MAT] = (unsigned long int)imgLabels.getTotalAcesses();
	accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)P.getTotalAcesses();

	//a = imgLabels.getImage(); 

	return nLabel;
}