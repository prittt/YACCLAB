// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_VOLUME_UTIL_H_
#define YACCLAB_VOLUME_UTIL_H_

#include <opencv2/core.hpp>

// Reads a 3d tensor (volume) from a directory containing the volume divided into slices, with their names listed in files.txt, and store the tensor into a cv::Mat.
cv::Mat volread(const cv::String &filename, int flags = 1);

// Writes a 3d tensor (volume) as separate slices in a directory, and lists file names in files.txt.
bool volwrite(const cv::String& filename, const cv::Mat& volume);

#endif /* YACCLAB_VOLUME_UTIL_H_ */