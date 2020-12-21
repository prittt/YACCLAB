// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "utilities.h"

#include <time.h>

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "progress_bar.h"
#include "system_info.h"
#include "volume_util.h"

#if defined YACCLAB_WITH_CUDA
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#endif

using namespace std;
using namespace cv;

#ifdef YACCLAB_APPLE
const string kTerminal = "postscript";
const string kTerminalExtension = ".ps";
#else
const string kTerminal = "pdf";
const string kTerminalExtension = ".pdf";
#endif

StreamDemultiplexer dmux::cout(std::cout);

bool CompareLengthCvString(String const& lhs, String const& rhs)
{
    return lhs.size() < rhs.size();
}

void RemoveCharacter(string& s, const char c)
{
    s.erase(std::remove(s.begin(), s.end(), c), s.end());
}

unsigned ctoi(const char& c)
{
    return ((int)c - 48);
}

string GetDatetime()
{
    time_t rawtime;
    char buffer[80];
    time(&rawtime);

    //Initialize buffer to empty string
    for (auto& c : buffer)
        c = '\0';

#if defined(YACCLAB_WINDOWS) && defined(_MSC_VER)

    struct tm timeinfo;
    localtime_s(&timeinfo, &rawtime);
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", &timeinfo);

#elif defined(YACCLAB_WINDOWS) || defined(YACCLAB_LINUX) || defined(YACCLAB_UNIX) || defined(YACCLAB_APPLE)

    struct tm* timeinfo;
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);

#endif
    string datetime(buffer);

    return datetime;
}

string GetDatetimeWithoutSpecialChars()
{
    string datetime(GetDatetime());
    std::replace(datetime.begin(), datetime.end(), ' ', '_');
    std::replace(datetime.begin(), datetime.end(), ':', '.');
    return datetime;
}

//void ColorLabels(const Mat1i& img_labels, Mat3b& img_out) {
//    img_out = Mat3b(img_labels.size());
//    for (int r = 0; r < img_labels.rows; ++r) {
//        unsigned const * const img_labels_row = img_labels.ptr<unsigned>(r);
//        Vec3b * const  img_out_row = img_out.ptr<Vec3b>(r);
//        for (int c = 0; c < img_labels.cols; ++c) {
//            img_out_row[c] = Vec3b(img_labels_row[c] * 131 % 255, img_labels_row[c] * 241 % 255, img_labels_row[c] * 251 % 255);
//        }
//    }
//}
//
//void ColorLabels(const Mat& img_labels, Mat& img_out) {
//    if (img_labels.dims == 3) {
//        img_out.create(3, img_labels.size.p, CV_8UC3);
//        for (int z = 0; z < img_labels.size[0]; z++) {
//            for (int y = 0; y < img_labels.size[1]; y++) {
//                unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(img_labels.data + z * img_labels.step[0] + y * img_labels.step[1]);
//                Vec3b * img_out_row = reinterpret_cast<Vec3b *>(img_out.data + z * img_out.step[0] + y * img_out.step[1]);
//                for (int x = 0; x < img_labels.size[2]; x++) {
//                    img_out_row[x] = Vec3b(img_labels_row[x] * 131 % 255, img_labels_row[x] * 241 % 255, img_labels_row[x] * 251 % 255);
//                }
//            }
//        }
//    }
//    else if (img_labels.dims == 2) {
//        img_out.create(img_labels.rows, img_labels.cols, CV_8UC3);
//        for (int y = 0; y < img_labels.size[0]; y++) {
//            unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(img_labels.data + y * img_labels.step[0]);
//            Vec3b * img_out_row = reinterpret_cast<Vec3b *>(img_out.data + y * img_out.step[0]);
//            for (int x = 0; x < img_labels.size[1]; x++) {
//                img_out_row[x] = Vec3b(img_labels_row[x] * 131 % 255, img_labels_row[x] * 241 % 255, img_labels_row[x] * 251 % 255);
//            }
//        }
//    }
//}
//
//void NormalizeLabels(Mat1i& img_labels) {
//    map<int, int> map_new_labels;
//    int i_max_new_label = 0;
//
//    for (int r = 0; r < img_labels.rows; ++r) {
//        unsigned * const img_labels_row = img_labels.ptr<unsigned>(r);
//        for (int c = 0; c < img_labels.cols; ++c) {
//            int iCurLabel = img_labels_row[c];
//            if (iCurLabel > 0) {
//                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
//                    map_new_labels[iCurLabel] = ++i_max_new_label;
//                }
//                img_labels_row[c] = map_new_labels.at(iCurLabel);
//            }
//        }
//    }
//}
//
//void NormalizeLabels(Mat& img_labels) {
//    map<int, int> map_new_labels;
//    int i_max_new_label = 0;
//
//    if (img_labels.dims == 3) {
//        for (int z = 0; z < img_labels.size[0]; z++) {
//            for (int y = 0; y < img_labels.size[1]; y++) {
//                unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(img_labels.data + z * img_labels.step[0] + y * img_labels.step[1]);
//                for (int x = 0; x < img_labels.size[2]; x++) {
//                    int iCurLabel = img_labels_row[x];
//                    if (iCurLabel > 0) {
//                        if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
//                            map_new_labels[iCurLabel] = ++i_max_new_label;
//                        }
//                        img_labels_row[x] = map_new_labels.at(iCurLabel);
//                    }
//                }
//            }
//        }
//    } else if (img_labels.dims == 2) {
//        for (int r = 0; r < img_labels.rows; ++r) {
//            unsigned * const img_labels_row = img_labels.ptr<unsigned>(r);
//            for (int c = 0; c < img_labels.cols; ++c) {
//                int iCurLabel = img_labels_row[c];
//                if (iCurLabel > 0) {
//                    if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
//                        map_new_labels[iCurLabel] = ++i_max_new_label;
//                    }
//                    img_labels_row[c] = map_new_labels.at(iCurLabel);
//                }
//            }
//        }
//    }
//}
//
//bool GetBinaryImage(const std::string& filename, cv::Mat& binary_mat) {
//    // Image load
//    cv::Mat image;
//	bool is_dir;
//	if (!filesystem::exists(path(filename), is_dir))
//		return false;
//	if (!is_dir) {
//		image = imread(filename, IMREAD_GRAYSCALE);   // Read the file
//	}
//	else {
//		image = volread(filename, IMREAD_GRAYSCALE);
//	}
//	// Check if image exist
//	if (image.empty()) {
//		return false;
//	}
//    // Adjust the threshold to make it binary
//    // threshold(image, binary_mat, 100, 1, THRESH_BINARY);			// da verificare che funzioni con i volumi
//    binary_mat = image;
//    return true;
//}
//
//bool GetBinaryImage(const path& p, Mat& binary_mat) {
//    return GetBinaryImage(p.string(), binary_mat);
//}
//
//bool GetBinaryImage(const std::string& filename, cv::Mat1b& binary_mat) {
//    // Image load
//    cv::Mat1b image;
//    image = imread(filename, IMREAD_GRAYSCALE);   // Read the file
//    // Check if image exist
//    if (image.empty()) {
//        return false;
//    }
//    // Adjust the threshold to make it binary
//    threshold(image, binary_mat, 100, 1, THRESH_BINARY);			
//    return true;
//}
//
//bool GetBinaryImage(const path& p, Mat1b& binary_mat) {
//    return GetBinaryImage(p.string(), binary_mat);
//}
//
//bool CompareMat(const Mat& mat_a, const Mat& mat_b)
//{
//    // Get a matrix with non-zero values at points where the
//    // two matrices have different values
//    cv::Mat diff = mat_a != mat_b;
//    // Equal if no elements disagree
//    if (cv::countNonZero(diff) != 0) {
//        volwrite("C:\\Users\\Stefano\\Desktop\\debug\\mistakes", diff);                                 ////////////////////////////////////////////////////
//    }
//    return cv::countNonZero(diff) == 0;
//}
//
//bool CompareMat(const Mat1i& mat_a, const Mat1i& mat_b)
//{
//    // Get a matrix with non-zero values at points where the
//    // two matrices have different values
//    cv::Mat1i diff = mat_a != mat_b;
//    // Equal if no elements disagree
//    return cv::countNonZero(diff) == 0;
//}
//
//void Divide(Mat& mat){
//    if (mat.dims == 3) {
//        unsigned x = mat.size[2] / 2;
//        for (unsigned y = 0; y < mat.size[1]; y++) {
//            for (unsigned z = 0; z < mat.size[0]; z++) {
//                mat.at<unsigned char>(z, y, x) = 0;
//            }
//        }
//        unsigned y = mat.size[1] / 2;
//        for (unsigned x = 0; x < mat.size[2]; x++) {
//            for (unsigned z = 0; z < mat.size[0]; z++) {
//                mat.at<unsigned char>(z, y, x) = 0;
//            }
//        }
//        unsigned z = mat.size[0] / 2;
//        for (unsigned y = 0; y < mat.size[1]; y++) {
//
//            for (unsigned x = 0; x < mat.size[2]; x++) {
//                mat.at<unsigned char>(z, y, x) = 0;
//            }
//        }
//    }
//}


bool CheckLabeledImage(const Mat1b& img, const Mat1i& labels, Mat1i& errors) {
    errors = Mat1i(img.size(), 0);
    bool correct = true;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            unsigned char val = img.at<unsigned char>(y, x);
            int label = labels.at<int>(y, x);
            if (val) {
                for (char ny = ((y > 0) ? -1 : 0); ny <= ((y + 1 < img.rows) ? 1 : 0); ny++) {
                    for (char nx = ((x > 0) ? -1 : 0); nx <= ((x + 1 < img.cols) ? 1 : 0); nx++) {
                        if (img.at<unsigned char>(y + ny, x + nx) && labels.at<int>(y + ny, x + nx) != label) {
                            errors.at<int>(y, x) = 1;
                            correct = false;
                        }
                    }
                }
            }
            else {
                if ((x > 0 && !img.at<unsigned char>(y, x - 1) && labels.at<int>(y, x - 1) != label) ||
                    (x + 1 < img.cols && !img.at<unsigned char>(y, x + 1) && labels.at<int>(y, x + 1) != label) ||
                    (y > 0 && !img.at<unsigned char>(y - 1, x) && labels.at<int>(y - 1, x) != label) ||
                    (y + 1 < img.rows && !img.at<unsigned char>(y + 1, x) && labels.at<int>(y + 1, x) != label)) {
                    errors.at<int>(y, x) = 1;
                    correct = false;
                }
            }
        }
    }
    return correct;
}

bool CheckLabeledVolume(const Mat& img, const Mat& labels, Mat& errors) {
    if (labels.dims != 3)
        throw std::runtime_error("Volume doesn't have 3 dims");
    if (labels.type() != CV_32SC1)
        throw std::runtime_error("Volume type isn't CV_32SC1");
    errors.create(3, labels.size.p, CV_32SC1);
    bool correct = true;
    for (int z = 0; z < labels.size[0]; z++) {
        for (int y = 0; y < labels.size[1]; y++) {
            for (int x = 0; x < labels.size[2]; x++) {
                unsigned char val = img.at<unsigned char>(z, y, x);
                int label = labels.at<int>(z, y, x);
                errors.at<int>(z, y, x) = 0;
                if (val) {
                    for (char nz = ((z > 0) ? -1 : 0); nz <= ((z + 1 < labels.size[0]) ? 1 : 0); nz++) {
                        for (char ny = ((y > 0) ? -1 : 0); ny <= ((y + 1 < labels.size[1]) ? 1 : 0); ny++) {
                            for (char nx = ((x > 0) ? -1 : 0); nx <= ((x + 1 < labels.size[2]) ? 1 : 0); nx++) {
                                if (img.at<unsigned char>(z + nz, y + ny, x + nx) && labels.at<int>(z + nz, y + ny, x + nx) != label) {
                                    errors.at<int>(z, y, x) = 1;
                                    correct = false;
                                }
                            }
                        }
                    }
                }
                else {
                    if ((x > 0 && !img.at<unsigned char>(z, y, x - 1) && labels.at<int>(z, y, x - 1) != label) ||
                        (x + 1 < labels.size[2] && !img.at<unsigned char>(z, y, x + 1) && labels.at<int>(z, y, x + 1) != label) ||
                        (y > 0 && !img.at<unsigned char>(z, y - 1, x) && labels.at<int>(z, y - 1, x) != label) ||
                        (y + 1 < labels.size[1] && !img.at<unsigned char>(z, y + 1, x) && labels.at<int>(z, y + 1, x) != label) ||
                        (z > 0 && !img.at<unsigned char>(z - 1, y, x) && labels.at<int>(z - 1, y, x) != label) ||
                        (z + 1 < labels.size[0] && !img.at<unsigned char>(z + 1, y, x) && labels.at<int>(z + 1, y, x) != label)) {
                        errors.at<int>(z, y, x) = 1;
                        correct = false;
                    }
                }
            }
        }
    }
    return correct;
}


void HideConsoleCursor()
{
#ifdef YACCLAB_WINDOWS
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO cursor_info;

    GetConsoleCursorInfo(out, &cursor_info);
    cursor_info.bVisible = false; // set the cursor visibility
    SetConsoleCursorInfo(out, &cursor_info);

#elif defined(YACCLAB_LINUX) || defined(YACCLAB_UNIX) || defined(YACCLAB_APPLE)
    int unused; // To avoid "return unused" and "variable unused" warnings 
    unused = system("setterm -cursor off");
#endif
    return;
}

int RedirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*)
{
    OutputBox ob;
    ob.Cerror(err_msg);
    return 0;
}

//std::string GetGnuplotTitle(const SystemInfo& s_info)
//{
//    string s = "\"{/:Bold CPU}: " + s_info.cpu() + " {/:Bold BUILD}: " + s_info.build() + " {/:Bold OS}: " + s_info.os() +
//        " {/:Bold COMPILER}: " + s_info.compiler_name() + " " + s_info.compiler_version() + "\" font ', 9'";
//    return s;
//}
//
//#if defined YACCLAB_WITH_CUDA
//string cudaBeautifyVersionNumber(int v) {
//	int minor = (v / 10) % 10;
//	int major = v / 1000;
//	return to_string(major) + '.' + to_string(minor);
//}
//
//std::string GetGnuplotTitleGpu(const SystemInfo& s_info)
//{
//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//	int runtimeVersion, driverVersion;
//	cudaRuntimeGetVersion(&runtimeVersion);
//	cudaDriverGetVersion(&driverVersion);
//
//	string s = "\"{/:Bold GPU}: " + string(prop.name) + " {/:Bold CUDA Capability}: " + to_string(prop.major) + '.' + to_string(prop.minor) + 
//		" {/:Bold Runtime}: " + cudaBeautifyVersionNumber(runtimeVersion) + " {/:Bold Driver}: " + cudaBeautifyVersionNumber(driverVersion);
//	return s;
//}
//#endif
std::string GetGnuplotTitle() {
    std::string s = "\"{/:Bold CPU}: " + SystemInfo::cpu() + " {/:Bold BUILD}: " + SystemInfo::build() + " {/:Bold OS}: " + SystemInfo::os() +
        " {/:Bold COMPILER}: " + SystemInfo::compiler_name() + " " + SystemInfo::compiler_version() + "\" font ', 9'";
    return s;
}

#if defined YACCLAB_WITH_CUDA
std::string GetGnuplotTitleGpu() {
    std::string s = "\"{/:Bold GPU}: " + CudaInfo::device_name() + " {/:Bold CUDA Capability}: " + CudaInfo::cuda_capability() +
        " {/:Bold Runtime}: " + CudaInfo::runtime_version() + " {/:Bold Driver}: " + CudaInfo::driver_version();
    return s;
}
#endif

string EscapeUnderscore(const string& s)
{
    string s_escaped;
    unsigned i = 0;
    for (const char& c : s) {
        if (c == '_' && i > 0 && s[i - 1] != '\\') {
            s_escaped += '\\';
        }
        s_escaped += c;
        ++i;
    }
    return s_escaped;
}

// Gnuplot requires double-escaped name when underscores are encountered
string DoubleEscapeUnderscore(const string& s)
{
    string s_escaped{ s };
    size_t found = s_escaped.find_first_of("_");
    while (found != std::string::npos) {
        s_escaped.insert(found, "\\\\");
        found = s_escaped.find_first_of("_", found + 3);
    }
    return s_escaped;
}