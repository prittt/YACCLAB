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


#include "labelingNULL.h"

using namespace cv;
using namespace std;

int labelingNULL(const Mat1b &img, Mat1i &imgLabels) {
	imgLabels = cv::Mat1i(img.size());

	int nLabel = imgLabels.rows*imgLabels.cols;
	for (int r = 0; r < imgLabels.rows; ++r) {
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(r);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		for (int c = 0; c < imgLabels.cols; ++c) {
			if (img_row[c])
				imgLabels_row[c] = ++nLabel;
			else
				imgLabels_row[c] = --nLabel;
		}
	}

	return nLabel;
}

int labelingNULL_MEM(const Mat1b &img_origin, vector<unsigned long int> &accesses) {
	
	memMat<uchar> img(img_origin); 
	memMat<int> imgLabels(img_origin.size());

	int nLabel = imgLabels.rows*imgLabels.cols;
	for (int r = 0; r < imgLabels.rows; ++r) {

		for (int c = 0; c < imgLabels.cols; ++c) {
			if (img(r,c))
				imgLabels(r,c) = ++nLabel;
			else
				imgLabels(r,c) = --nLabel;
		}
	}

	// Store total accesses in the output vector 'accesses'
	accesses = vector<unsigned long int>((int)MD_SIZE, 0);

	accesses[MD_BINARY_MAT] = (unsigned long int)img.getTotalAcesses();
	accesses[MD_LABELED_MAT] = (unsigned long int)imgLabels.getTotalAcesses();

	return nLabel;
}