#include "labelingNULL.h"

using namespace cv;
using namespace std;

static unsigned long int xnext = 1;

inline int xrand(void) // RAND_MAX assumed to be 32767
{
	xnext = xnext * 1103515245 + 12345;
	return (unsigned int)(xnext / 65536) % 32768;
}

int labelingNULL(const cv::Mat1b &img, cv::Mat1i &imgLabels) {
	imgLabels = cv::Mat1i(img.size());

	int nLabel = 0;
	for (int r = 0; r < imgLabels.rows; ++r) {
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(r);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		for (int c = 0; c < imgLabels.cols; ++c) {
			//if (img_row[c])
			//	imgLabels_row[c] = ++nLabel;
			//else
			//	imgLabels_row[c] = --nLabel;

			if (xrand()>16384)
				imgLabels_row[c] = img_row[c] + 1;
			else
				imgLabels_row[c] = img_row[c] - 1;

			//nLabel = nLabel + img_row[c] * 2 - 1;
			//imgLabels_row[c] = nLabel;
		}
	}

	return nLabel;
}
