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

#include "labelingWu2009.h"
#include <vector>

using namespace std;
using namespace cv; 

//Find the root of the tree of node i
template<typename LabelT>
inline static
LabelT findRoot(const LabelT *P, LabelT i){
	LabelT root = i;
	while (P[root] < root){
		root = P[root];
	}
	return root;
}

//Make all nodes in the path of node i point to root
template<typename LabelT>
inline static
void setRoot(LabelT *P, LabelT i, LabelT root){
	while (P[i] < i){
		LabelT j = P[i];
		P[i] = root;
		i = j;
	}
	P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
template<typename LabelT>
inline static
LabelT find(LabelT *P, LabelT i){
	LabelT root = findRoot(P, i);
	setRoot(P, i, root);
	return root;
}

//unite the two trees containing nodes i and j and return the new root
template<typename LabelT>
inline static
LabelT set_union(LabelT *P, LabelT i, LabelT j){
	LabelT root = findRoot(P, i);
	if (i != j){
		LabelT rootj = findRoot(P, j);
		if (root > rootj){
			root = rootj;
		}
		setRoot(P, j, root);
	}
	setRoot(P, i, root);
	return root;
}

//Flatten the Union Find tree and relabel the components
template<typename LabelT>
inline static
LabelT flattenL(LabelT *P, LabelT length){
	LabelT k = 1;
	for (LabelT i = 1; i < length; ++i){
		if (P[i] < i){
			P[i] = P[P[i]];
		}
		else{
			P[i] = k; k = k + 1;
		}
	}
	return k;
}

int SAUF_OPT(const Mat1b &img, Mat1i &imgLabels){

	const int h = img.rows;
	const int w = img.cols;
    
	imgLabels = Mat1i(img.size(),0);

	const size_t Plength = img.rows * img.cols / 4;		 // Raw superior limit for labels number
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength); //array P for equivalences resolution
	P[0] = 0;	//first label is for background pixels
	uint lunique = 1;

	// first scan 

	// Rosenfeld Mask
	// +-+-+-+
	// |p|q|r|
	// +-+-+-+
	// |s|x|
	// +-+-+

	for (int r = 0; r < h; ++r)
	{
		uchar const * const img_row = img.ptr<uchar>(r);
		uchar const * const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		uint * const  imgLabels_row = imgLabels.ptr<uint>(r);
		uint * const  imgLabels_row_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0]);
		
		for (int c = 0; c < w; ++c) {

			#define condition_p c>0 && r>0 && img_row_prev[c - 1]>0
			#define condition_q r>0 && img_row_prev[c]>0
			#define condition_r c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
			#define condition_s c > 0 && img_row[c - 1] > 0
			#define condition_x img_row[c] > 0

			if (condition_x){
				if (condition_q){
					//x <- q
					imgLabels_row[c] = imgLabels_row_prev[c];
				}
				else{
					// q = 0
					if (condition_r){
						if (condition_p){
							// x <- merge(p,r)
							imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]);
						}
						else{ 
							// p = q = 0
							if (condition_s){
								// x <- merge(s,r)
								imgLabels_row[c] = set_union(P, imgLabels_row[c - 1], imgLabels_row_prev[c + 1]);
							}
							else{ 
								// p = q = s = 0
								// x <- r
								imgLabels_row[c] = imgLabels_row_prev[c + 1];
							}
						}
					}
					else{
						// r = q = 0
						if (condition_p){
							// x <- p
							imgLabels_row[c] = imgLabels_row_prev[c - 1];
						}
						else{
							// r = q = p = 0
							if (condition_s){
								imgLabels_row[c] = imgLabels_row[c - 1];
							}
							else{
								//new label
								imgLabels_row[c] = lunique;
								P[lunique] = lunique;
								lunique = lunique + 1;
							}
						}
					}
				}
			}
			else{
				//Nothing to do, x is a background pixel
			}
		}
	}

	//second scan
	uint nLabel = flattenL(P, lunique);

	for (int r = 0; r < imgLabels.rows; ++r) {

		uint * img_row_start = imgLabels.ptr<uint>(r);
		uint * const img_row_end = img_row_start + imgLabels.cols;
		for (; img_row_start != img_row_end; ++img_row_start){
			*img_row_start = P[*img_row_start];
		}
	}

	fastFree(P);
	return nLabel;
}