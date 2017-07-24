#ifndef YACCLAB_UTILITIES_H_
#define YACCLAB_UTILITIES_H_

#include <string>
#include <opencv2/core.hpp>

#include "file_manager.h"

//extern const char kPathSeparator;
extern const std::string kTerminal;
extern const std::string kTerminalExtension;

// To compare lengths of OpenCV String 
bool CompareLengthCvString(cv::String const& lhs, cv::String const& rhs);

// This function is useful to delete eventual carriage return from a string
// and is especially designed for windows file newline format
//void DeleteCarriageReturn(std::string& s);
void RemoveCharacter(std::string& s, const char c);

// This function take a char as input and return the corresponding int value (not ASCII one)
unsigned ctoi(const char& c);

// This function help us to manage '\' escape character
//void EraseDoubleEscape(std::string& str);

/*@brief Get information about date and time

@param[in] bool if true substitute both ' ' and ':' chars with '_' and '.'

@return string value with datetime stringyfied
*/
std::string GetDatetime();
std::string GetDatetimeWithoutSpecialChars();

// Create a bunch of pseudo random colors from labels indexes and create a
// color representation for the labels
void ColorLabels(const cv::Mat1i& img_labels, cv::Mat3b& img_out);

// This function may be useful to compare the output of different labeling procedures
// which may assign different labels to the same object. Use this to force a row major
// ordering of labels.
void NormalizeLabels(cv::Mat1i& img_labels);

// Get binary image given a image's filename;
bool GetBinaryImage(const std::string& filename, cv::Mat1b& binary_mat);
bool GetBinaryImage(const filesystem::path& p, cv::Mat1b& binary_mat);


// Compare two int matrices element by element
bool CompareMat(const cv::Mat1i& mat_a, const cv::Mat1i& mat_b);

/*@brief Read bool from YAML configuration file

@param[in] node_list FileNode that contain bool data_

@return bool value of field in node_list
*/
//bool ReadBool(const cv::FileNode& node_list);

// Hide blinking cursor from console
void HideConsoleCursor();

int RedirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*);

void cerror(const std::string& err);

void cmessage(const std::string& msg);

/*
@brief Return the string title to insert in gnuplot charts

@return string which represents the title
*/
std::string GetGnuplotTitle();

#endif // !YACCLAB_UTILITIES_H_