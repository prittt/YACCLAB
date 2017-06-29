#include "utilities.h"

using namespace std;
using namespace cv;

const char kPathSeparator =
#ifdef _WIN32
'\\';
#else
'/';
#endif

#ifdef APPLE
const string terminal = "postscript";
const string terminalExtension = ".ps";
#else
const string terminal = "pdf";
const string terminalExtension = ".pdf";
#endif

string& trim(string& s, const char* t)
{
    s.erase(0, s.find_first_not_of(t));
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

void deleteCarriageReturn(string& s)
{
    size_t found;
    do
    {
        // The while cycle is probably unnecessary
        found = s.find("\r");
        if (found != string::npos)
            s.erase(found, 1);
    } while (found != string::npos);

    return;
}

unsigned ctoi(const char& c)
{
    return ((int)c - 48);
}

void eraseDoubleEscape(string& str)
{
    for (size_t i = 0; i < str.size() - 1; ++i)
    {
        if (str[i] == str[i + 1] && str[i] == '\\')
            str.erase(i, 1);
    }
}

string getDatetime()
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

    return string(buffer);
}

void colorLabels(const Mat1i& imgLabels, Mat3b& imgOut)
{
    imgOut = Mat3b(imgLabels.size());
    for (int r = 0; r < imgLabels.rows; ++r)
    {
        for (int c = 0; c < imgLabels.cols; ++c)
        {
            imgOut(r, c) = Vec3b(imgLabels(r, c) * 131 % 255, imgLabels(r, c) * 241 % 255, imgLabels(r, c) * 251 % 255);
        }
    }
}

void normalizeLabels(Mat1i& imgLabels)
{
    map<int, int> mapNewLabels;
    int iMaxNewLabel = 0;

    for (int r = 0; r < imgLabels.rows; ++r)
    {
        unsigned * const imgLabels_row = imgLabels.ptr<unsigned>(r);
        for (int c = 0; c < imgLabels.cols; ++c)
        {
            int iCurLabel = imgLabels_row[c];
            if (iCurLabel > 0)
            {
                if (mapNewLabels.find(iCurLabel) == mapNewLabels.end())
                {
                    mapNewLabels[iCurLabel] = ++iMaxNewLabel;
                }
                imgLabels_row[c] = mapNewLabels.at(iCurLabel);
            }
        }
    }
}

bool getBinaryImage(const string FileName, Mat1b& binaryMat)
{
    // Image load
    Mat image;
    image = imread(FileName, IMREAD_GRAYSCALE);   // Read the file

    // Check if image exist
    if (image.empty())
        return false;

    // Adjust the threshold to actually make it binary
    threshold(image, binaryMat, 100, 1, THRESH_BINARY);

    return true;
}

bool compareMat(const Mat1i& mata, const Mat1i& matb)
{
    // Get a matrix with non-zero values at points where the
    // two matrices have different values
    cv::Mat diff = mata != matb;
    // Equal if no elements disagree
    return cv::countNonZero(diff) == 0;
}

bool readBool(const FileNode& nodeList)
{
    bool b = false;
    if (!nodeList.empty())
    {
        //The entry is found
        string s((string)nodeList);
        transform(s.begin(), s.end(), s.begin(), ::tolower);
        istringstream(s) >> boolalpha >> b;
    }

    return b;
}

void HideConsoleCursor()
{
#ifdef WINDOWS
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO cursorInfo;

    GetConsoleCursorInfo(out, &cursorInfo);
    cursorInfo.bVisible = false; // set the cursor visibility
    SetConsoleCursorInfo(out, &cursorInfo);

#elif define(LINUX) || define(LINUX) || define(APPLE)
    system("setterm -cursor off");
#endif
    return;
}

int redirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*)
{
    cerr << "Error: [" << err_msg << "]" << endl;
    return 0;
}