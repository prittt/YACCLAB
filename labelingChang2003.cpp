#include "labelingChang2003.h"

using namespace std; 
using namespace cv;

inline Point2i Tracer(const Mat1b &img, const unsigned char byF, Mat1i &imgOut, const Point2i &P, int iLabel, int &iPrev, bool &bIsolated) {
	int iFirst,iNext;

	// Find the direction to be analyzed
	iFirst = iNext = (iPrev+2)%8;

    Point2i crdNext;
	do {
		switch (iNext) {
			case 0: crdNext = P + Point2i(1,0); break;
            case 1: crdNext = P + Point2i(1, 1); break;
            case 2: crdNext = P + Point2i(0, 1); break;
            case 3: crdNext = P + Point2i(-1, 1); break;
            case 4: crdNext = P + Point2i(-1, 0); break;
            case 5: crdNext = P + Point2i(-1, -1); break;
            case 6: crdNext = P + Point2i(0, -1); break;
            case 7: crdNext = P + Point2i(1, -1); break;
		}
		
        if (crdNext.y >= 0 && crdNext.x >= 0 && crdNext.y < img.rows && crdNext.x < img.cols) {
			if (img(crdNext.y, crdNext.x)==byF) {
				iPrev = (iNext+4)%8;
				return crdNext;
			}
			else
				imgOut(crdNext.y, crdNext.x) = -1;
		}

		iNext = (iNext+1)%8;
	} while (iNext!=iFirst);

	bIsolated = true;
	return P;
}

inline void ContourTracing (const Mat1b &img, const unsigned char byF, Mat1i &imgOut, int x, int y, int iLabel, bool bExternal) {
	Point2i S(x,y),T,crdNextPoint,crdCurPoint;
	
    // The current point is labeled 
	// As a first step we label the point
	imgOut(S.y, S.x) = iLabel;

	bool bIsolated(false);
	int iPreviousContourPoint;
	if (bExternal)
		iPreviousContourPoint = 6;
	else 
		iPreviousContourPoint = 7;

	// First call to Tracer
	crdNextPoint = T = Tracer (img,byF,imgOut,S,iLabel,iPreviousContourPoint,bIsolated);
	if (bIsolated)
		return;

	do {
		crdCurPoint = crdNextPoint;
		imgOut(crdCurPoint.y, crdCurPoint.x) = iLabel;
		crdNextPoint = Tracer (img,byF,imgOut,crdCurPoint,iLabel,iPreviousContourPoint,bIsolated);
	} while (!(crdCurPoint==S && crdNextPoint==T));
}

int CT_OPT(const Mat1b &img, Mat1i &imgOut) {
	unsigned char byF = 1; 
    imgOut = Mat1i(img.size(), 0);
	
	int iNewLabel(0);
	for(int y=0; y<img.rows; y++) {
        const uchar* const img_row = img.ptr<uchar>(y);
        uint* const imgOut_row = imgOut.ptr<uint>(y);
		for(int x=0; x<img.cols; x++) {

			if (img_row[x] == byF) {
				// case 1
				if (imgOut_row[x]==0 && (x==0 || img_row[x-1]!=byF)) {
					iNewLabel++;
					ContourTracing (img,byF,imgOut,x,y,iNewLabel,true);
					continue;
				}
				// case 2
				else if (x<img.cols-1 && img_row[x+1]!=byF && imgOut_row[x+1]!=-1) {
					if (imgOut_row[x]==0) {
						// current pixel unlabeled
						// assing label of left pixel
						ContourTracing (img,byF,imgOut,x,y,imgOut_row[x-1],false);
					}
					else {
						ContourTracing(img,byF,imgOut,x,y,imgOut_row[x],false);	
					}
					continue;
				}
				// case 3
				else if (imgOut_row[x]==0) {
					imgOut_row[x] = imgOut_row[x-1];
				}
			}
		}
	}
	
    // Reassign to contour background value (0)
    for (int r = 0; r < imgOut.rows; ++r){
        uint* const imgOut_row = imgOut.ptr<uint>(r);
        for (int c = 0; c < imgOut.cols; ++c){
            if (imgOut_row[c] == -1)
                imgOut_row[c] = 0;
        }
    }

    return ++iNewLabel;
}
