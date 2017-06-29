#pragma once
#include <opencv2/opencv.hpp>
#include <time.h>
#include <map>
#include "systemInfo.h"

extern const char kPathSeparator;
extern const std::string terminal;
extern const std::string terminalExtension;

/*@brief Remove heading and trailing spaces in string

@param[in] s string to trim
@param[in] t characters to remove from s

@return string's reference trimmed
*/
std::string& trim(std::string& s, const char* t = " \t\n\r\f\v");

// This function is useful to delete eventual carriage return from a string
// and is especially designed for windows file newline format
void deleteCarriageReturn(std::string& s);

// This function take a char as input and return the corresponding int value (not ASCII one)
unsigned ctoi(const char& c);

// This function help us to manage '\' escape character
void eraseDoubleEscape(std::string& str);

// Get information about date and time
std::string getDatetime();

// Create a bunch of pseudo random colors from labels indexes and create a
// color representation for the labels
void colorLabels(const cv::Mat1i& imgLabels, cv::Mat3b& imgOut);

// This function may be useful to compare the output of different labeling procedures
// which may assign different labels to the same object. Use this to force a row major
// ordering of labels.
void normalizeLabels(cv::Mat1i& imgLabels);

// Get binary image given a image's FileName;
bool getBinaryImage(const std::string FileName, cv::Mat1b& binaryMat);

// Compare two int matrixes element by element
bool compareMat(const cv::Mat1i& mata, const cv::Mat1i& matb);


/*@brief Read bool from YAML configuration file

@param[in] nodeList FileNode that contain bool data

@return bool value of field in nodeList
*/
bool readBool(const cv::FileNode& nodeList);

// Hide blinking cursor from console
void HideConsoleCursor();

int redirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*);