// Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

    struct tm * timeinfo;
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

void ColorLabels(const Mat1i& img_labels, Mat3b& img_out)
{
    img_out = Mat3b(img_labels.size());
    for (int r = 0; r < img_labels.rows; ++r) {
        unsigned const * const img_labels_row = img_labels.ptr<unsigned>(r);
        Vec3b * const  img_out_row = img_out.ptr<Vec3b>(r);
        for (int c = 0; c < img_labels.cols; ++c) {
            img_out_row[c] = Vec3b(img_labels_row[c] * 131 % 255, img_labels_row[c] * 241 % 255, img_labels_row[c] * 251 % 255);
        }
    }
}

void NormalizeLabels(Mat1i& img_labels)
{
    map<int, int> map_new_labels;
    int i_max_new_label = 0;

    for (int r = 0; r < img_labels.rows; ++r) {
        unsigned * const img_labels_row = img_labels.ptr<unsigned>(r);
        for (int c = 0; c < img_labels.cols; ++c) {
            int iCurLabel = img_labels_row[c];
            if (iCurLabel > 0) {
                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                    map_new_labels[iCurLabel] = ++i_max_new_label;
                }
                img_labels_row[c] = map_new_labels.at(iCurLabel);
            }
        }
    }
}

bool GetBinaryImage(const string& filename, Mat1b& binary_mat, bool inverted)
{
    // Image load
    Mat1b image;
    image = imread(filename, IMREAD_GRAYSCALE);   // Read the file
    // Check if image exist
    if (image.empty()) {
        return false;
    }

    // Adjust the threshold to make it binary
    if (inverted) {
        threshold(image, binary_mat, 100, 1, THRESH_BINARY_INV);
    }
    else {
        threshold(image, binary_mat, 100, 1, THRESH_BINARY);
    }
    return true;
}

bool GetBinaryImage(const filesystem::path& p, Mat1b& binary_mat, bool inverted)
{
    return GetBinaryImage(p.string(), binary_mat, inverted);
}

bool CompareMat(const Mat1i& mat_a, const Mat1i& mat_b)
{
    // Get a matrix with non-zero values at points where the
    // two matrices have different values
    cv::Mat diff = mat_a != mat_b;
    // Equal if no elements disagree
    return cv::countNonZero(diff) == 0;
}

void HideConsoleCursor()
{
#ifdef YACCLAB_WINDOWS
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO cursor_info;

    GetConsoleCursorInfo(out, &cursor_info);
    cursor_info.bVisible = false; // set the cursor visibility
    SetConsoleCursorInfo(out, &cursor_info);

#elif defined(YACCLAB_LINUX) || defined(YACCLAB_LINUX) || defined(YACCLAB_APPLE)
    system("setterm -cursor off");
#endif
    return;
}

int RedirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*)
{
    OutputBox ob;
    ob.Cerror(err_msg);
    return 0;
}

std::string GetGnuplotTitle(ConfigData& cfg)
{
    SystemInfo s_info(cfg);
    string s = "\"{/:Bold CPU}: " + s_info.cpu() + " {/:Bold BUILD}: " + s_info.build() + " {/:Bold OS}: " + s_info.os() +
        " {/:Bold COMPILER}: " + s_info.compiler_name() + " " + s_info.compiler_version() + "\" font ', 9'";
    return s;
}

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