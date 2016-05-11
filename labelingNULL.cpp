#include "labelingNULL.h"

using namespace cv;
using namespace std;

int labelingNULL(const cv::Mat1b &img, cv::Mat1i &imgLabels) {
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
