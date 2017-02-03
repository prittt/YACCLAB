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

#include "labelingZhao2010.h"

using namespace cv;
using namespace std;

inline int FindRoot(Mat1i &imgOut, int pos)
{
	while (true) {
		int tmppos = imgOut(-pos);
		if (tmppos == pos)
			break;
		pos = tmppos;
	}
	return pos;
}

inline void FindRootAndCompress(Mat1i &imgOut, int pos, int newroot)
{
	while (true) {
		int tmppos = imgOut(-pos);
		if (tmppos == newroot)
			break;
		imgOut(-pos) = newroot;
		if (tmppos == pos)
			break;
		pos = tmppos;
	}
}

int SBLA(const Mat1b &img, Mat1i &imgOut) 
{
	int N = img.cols;
	int M = img.rows;
	imgOut = Mat1i(M, N);

	// Copy the input image into the output one:
	//copy(begin(img), end(img), begin(imgOut)); This produce a warning in debug mode
	for (int r = 0; r < img.rows; ++r){
		for (int c = 0; c < img.cols; ++c){
			imgOut(r, c) = img(r, c); 
		}
	}

	// Fix for first pixel!
	int LabelNum = 0;
	bool firstpixel = imgOut(0, 0) > 0;
	if (firstpixel) {
		imgOut(0, 0) = 0;
		if (imgOut(0, 1) == 0 && imgOut(1, 0) == 0 && imgOut(1, 1) == 0)
			LabelNum = 1;
	}

	// Stripe extraction and representation
	for (int r = 0; r < M; r += 2) {
		for (int c = 0; c < N; ++c) {
			// Step 1
			int evenpix = imgOut(r, c);
			int oddpix = r + 1 < M ? imgOut(r + 1, c) : 0;

			// Step 2
			int Gp;
			if (oddpix) {
				imgOut(r + 1, c) = Gp = -((r + 1) * N + c); 
				if (evenpix)
					imgOut(r, c) = Gp;
			}
			else if (evenpix)
				imgOut(r, c) = Gp = -(r * N + c);
			else
				continue;

			// Step 3
			int stripestart = c;
			while (++c < N) {
				int evenpix = imgOut(r, c);
				int oddpix = r + 1 < M ? imgOut(r + 1, c) : 0;

				if (oddpix) {
					imgOut(r + 1, c) = Gp;
					if (evenpix)
						imgOut(r, c) = Gp;
				}
				else if (evenpix)
					imgOut(r, c) = Gp;
				else
					break;				
			}
			int stripestop = c;

			if (r == 0)
				continue;

			// Stripe union
			int lastroot = INT_MIN;
			for (int i = stripestart; i < stripestop; ++i) {
				int linepix = imgOut(r, i);
				if (!linepix)
					continue;

				int runstart = max(0, i - 1);
				do 
                    i++;
				while (i < N && imgOut(r, i));
				int runstop = min(N - 1, i);

				for (int j = runstart; j <= runstop; ++j) {
					int curpix = imgOut(r - 1, j);
					if (!curpix)
						continue;

					int newroot = FindRoot(imgOut, curpix);
					if (newroot > lastroot) {
						lastroot = newroot;
						FindRootAndCompress(imgOut, Gp, lastroot);
					}
					else if (newroot < lastroot) {
						FindRootAndCompress(imgOut, newroot, lastroot);
					}

					do 
                        ++j;
					while (j <= runstop && imgOut(r - 1, j));
				}				
			}
		}
	}

	// Label assignment
	for (int i = 0; i < M * N; ++i) {
		// FindRoot_GetLabel
		int pos = imgOut(i);
		if (pos >= 0)
			continue;

		while (pos != imgOut(-pos) && imgOut(-pos) < 0)
			pos = imgOut(-pos);
		if (imgOut(-pos) < 0)
			imgOut(-pos) = ++LabelNum;

		// Assign final label
		imgOut(i) = imgOut(-pos);
	}

	// Fix for first pixel!
	if (firstpixel) {
		if (imgOut(0, 1))
			imgOut(0, 0) = imgOut(0, 1);
		else if (imgOut(1, 0))
			imgOut(0, 0) = imgOut(1, 0);
		else if (imgOut(1, 1))
			imgOut(0, 0) = imgOut(1, 1);
		else
			imgOut(0, 0) = 1;
	}

	imgOut.rows = img.rows;
	return LabelNum + 1;
}

inline int FindRoot_OPT(int *imgOut, int pos)
{
	while (true) {
		int tmppos = imgOut[-pos];
		if (tmppos == pos)
			break;
		pos = tmppos;
	}
	return pos;
}

inline void FindRootAndCompress_OPT(int *imgOut, int pos, int newroot)
{
	while (true) {
		int tmppos = imgOut[-pos];
		if (tmppos == newroot)
			break;
		imgOut[-pos] = newroot;
		if (tmppos == pos)
			break;
		pos = tmppos;
	}
}

int SBLA_OPT(const Mat1b &img, Mat1i &imgOut)
{
	int N = img.cols;
	int M = img.rows;
	imgOut = Mat1i(M, N);

	// Fix for first pixel!
	int LabelNum = 0;
	uchar firstpixel = img(0, 0);
	if (firstpixel) {
		const_cast<Mat1b&>(img)(0, 0) = 0;
		if (img(0, 1) == 0 && img(1, 0) == 0 && img(1, 1) == 0)
			LabelNum = 1;
	}

	// Stripe extraction and representation
	uint* imgOut_row_prev = nullptr;
	int rN = 0;
	int r1N = N;
	int N2 = N * 2;
	for (int r = 0; r < M; r += 2) {
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgOut_row = imgOut.ptr<uint>(r);
		uint* const imgOut_row_fol = (uint *)(((char *)imgOut_row) + imgOut.step.p[0]);
		for (int c = 0; c < N; ++c) {
			imgOut_row[c] = img_row[c];
			if (r + 1 < M)
				imgOut_row_fol[c] = img_row_fol[c];
		}

		for (int c = 0; c < N; ++c) {
			// Step 1
			int evenpix = imgOut_row[c];
			int oddpix = r + 1 < M ? imgOut_row_fol[c] : 0;

			// Step 2
			int Gp;
			if (oddpix) {
				imgOut_row_fol[c] = Gp = -(r1N + c);
				if (evenpix)
					imgOut_row[c] = Gp;
			}
			else if (evenpix)
				imgOut_row[c] = Gp = -(rN + c);
			else
				continue;

			// Step 3
			int stripestart = c;
			while (++c < N) {
				int evenpix = imgOut_row[c];
				int oddpix = r + 1 < M ? imgOut_row_fol[c] : 0;

				if (oddpix) {
					imgOut_row_fol[c] = Gp;
					if (evenpix)
						imgOut_row[c] = Gp;
				}
				else if (evenpix)
					imgOut_row[c] = Gp;
				else
					break;
			}
			int stripestop = c;

			if (r == 0)
				continue;

			// Stripe union
			int lastroot = INT_MIN;
			for (int i = stripestart; i < stripestop; ++i) {
				int linepix = imgOut_row[i];
				if (!linepix)
					continue;

				int runstart = max(0, i - 1);
				do
					i++;
				while (i < N && imgOut_row[i]);
				int runstop = min(N - 1, i);

				for (int j = runstart; j <= runstop; ++j) {
					int curpix = imgOut_row_prev[j];
					if (!curpix)
						continue;

					int newroot = FindRoot_OPT(reinterpret_cast<int*>(imgOut.data), curpix);
					if (newroot > lastroot) {
						lastroot = newroot;
						FindRootAndCompress_OPT(reinterpret_cast<int*>(imgOut.data), Gp, lastroot);
					}
					else if (newroot < lastroot) {
						FindRootAndCompress_OPT(reinterpret_cast<int*>(imgOut.data), newroot, lastroot);
					}

					do
						++j;
					while (j <= runstop && imgOut_row_prev[j]);
				}
			}
		}
		imgOut_row_prev = imgOut_row_fol;
		rN += N2;
		r1N += N2;
	}

	// Label assignment
	int *imgOutdata = reinterpret_cast<int*>(imgOut.data);
	for (int i = 0; i < M * N; ++i) {
		// FindRoot_GetLabel
		int pos = imgOutdata[i];
		if (pos >= 0)
			continue;

		while (pos != imgOutdata[-pos] && imgOutdata[-pos] < 0)
			pos = imgOutdata[-pos];
		if (imgOutdata[-pos] < 0)
			imgOutdata[-pos] = ++LabelNum;

		// Assign final label
		imgOutdata[i] = imgOutdata[-pos];
	}

	// Fix for first pixel!
	if (firstpixel) {
		const_cast<Mat1b&>(img)(0, 0) = firstpixel;
		if (imgOut(0, 1))
			imgOut(0, 0) = imgOut(0, 1);
		else if (imgOut(1, 0))
			imgOut(0, 0) = imgOut(1, 0);
		else if (imgOut(1, 1))
			imgOut(0, 0) = imgOut(1, 1);
		else
			imgOut(0, 0) = 1;
	}

	return LabelNum + 1;
}

