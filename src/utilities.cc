#include "utilities.h"

#include <time.h>

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "system_info.h"
#include "file_manager.h"

using namespace std;
using namespace cv;

#ifdef APPLE
const string kTerminal = "postscript";
const string kTerminalExtension = ".ps";
#else
const string kTerminal = "pdf";
const string kTerminalExtension = ".pdf";
#endif

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

#ifdef WINDOWS

    struct tm timeinfo;
    localtime_s(&timeinfo, &rawtime);
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", &timeinfo);

#elif defined(LINUX) || defined(UNIX) || defined(APPLE)

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

bool GetBinaryImage(const string& filename, Mat1b& binary_mat)
{
  // Image load
    Mat image;
    image = imread(filename, IMREAD_GRAYSCALE);   // Read the file

    // Check if image exist
    if (image.empty())
        return false;

      // Adjust the threshold to actually make it binary
    threshold(image, binary_mat, 100, 1, THRESH_BINARY);

    return true;
}

bool GetBinaryImage(const filesystem::path& p, Mat1b& binary_mat)
{
    return GetBinaryImage(p.string(), binary_mat);
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
#ifdef WINDOWS
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO cursor_info;

    GetConsoleCursorInfo(out, &cursor_info);
    cursor_info.bVisible = false; // set the cursor visibility
    SetConsoleCursorInfo(out, &cursor_info);

#elif defined(LINUX) || defined(LINUX) || defined(APPLE)
    system("setterm -cursor off");
#endif
    return;
}

int RedirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*)
{
    cerror(err_msg);
    return 0;
}

#define CONSOLE_WIDTH 80

void cerror(const string& err)
{
    string status_msg = "ERROR: [";
    size_t num_spaces = max(0, int(CONSOLE_WIDTH - (status_msg.size() + err.size()) + 1) % CONSOLE_WIDTH); // 70 = output console dimension - "[ERROR]: " dimension
    cerr << status_msg << err << "]" << string(num_spaces, ' ') << endl;
    return;
}

void cmessage(const string& msg)
{
    string status_msg = "MSG: [";
    size_t num_spaces = max(0, int(CONSOLE_WIDTH - (status_msg.size() + msg.size()) + 1) % CONSOLE_WIDTH); // 70 = output console dimension - "[ERROR]: " dimension
    cerr << status_msg << msg << "]" << string(num_spaces, ' ') << endl;
    return;
}

std::string GetGnuplotTitle()
{
    SystemInfo s_info;
    string s = "\"{/:Bold CPU}: " + s_info.cpu() + " {/:Bold BUILD}: " + s_info.build() + " {/:Bold OS}: " + s_info.os() + "\" font ', 11'";
    return s;
}