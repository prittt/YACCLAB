#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <functional>
#include <cstdint>
#include <iomanip>

#include "performanceEvaluator.h"
#include "labelingAlgorithms.h"
#include "foldersManager.h"
#include "progressBar.h"
#include "memoryTester.h"
#include "systemInfo.h"
#include "latexGeneration.h"
#include "utilities.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

extern const char kPathSeparator;
extern const string terminal;
extern const string terminalExtension;

// To check the correctness of algorithms on datasets specified
void CheckAlgorithms(vector<String>& CCLAlgorithms, const vector<String>& datasets, const string& inputPath, const string& inputTxt)
{
    vector<bool> stats(CCLAlgorithms.size(), true); // true if the i-th algorithm is correct, false otherwise
    vector<string> firstFail(CCLAlgorithms.size()); // name of the file on which algorithm fails the first time
    bool stop = false; // true if all algorithms are incorrect
    bool checkPerform = false; // true if almost one check was execute

    for (unsigned i = 0; i < datasets.size(); ++i) {
        // For every dataset in check list

        cout << "Test on " << datasets[i] << " starts: " << endl;

        string isPath = inputPath + kPathSeparator + datasets[i] + kPathSeparator + inputTxt;

        // Open file
        ifstream is(isPath);
        if (!is.is_open()) {
            cout << "Unable to open " + isPath << endl;
            continue;
        }

        // To save list of filename on which CLLAlgorithms must be tested
        vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
        string filename;
        while (getline(is, filename)) {
            // To delete eventual carriage return in the file name (especially designed for windows file newline format)
            size_t found;
            do {
                // The while cycle is probably unnecessary
                found = filename.find("\r");
                if (found != string::npos)
                    filename.erase(found, 1);
            } while (found != string::npos);
            // Add purified file name in the vector
            filenames.push_back(make_pair(filename, true));
        }
        is.close();

        // Number of files
        int fileNumber = filenames.size();

        // Count number of lines to display progress bar
        unsigned currentNumber = 0;

        ProgressBar pBar(fileNumber);
        pBar.Start();

        // For every file in list
        for (unsigned file = 0; file < filenames.size() && !stop; ++file) {
            filename = filenames[file].first;

            deleteCarriageReturn(filename);

            pBar.Display(currentNumber++);

            if (!getBinaryImage(inputPath + kPathSeparator + datasets[i] + kPathSeparator + filename, labeling::aImg)) {
                cout << "Unable to check on '" + filename + "', file does not exist" << endl;
                continue;
            }

            unsigned nLabelsCorrect, nLabelsToControl;

            // SAUF is the reference (the labels are already normalized)
            auto& SAUF = LabelingMapSingleton::GetInstance().data.at("SAUF");
            SAUF->AllocateMemory();
            nLabelsCorrect = SAUF->PerformLabeling();
            SAUF->DeallocateMemory();

            Mat1i& labeledImgCorrect = SAUF->aImgLabels;
            //nLabelsCorrect = connectedComponents(binaryImg, labeledImgCorrect, 8, 4, CCL_WU);

            unsigned j = 0;
            for (const auto& algoName : CCLAlgorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data.at(algoName);
                checkPerform = true;
                if (stats[j]) {
                    try {
                        Mat1i& labeledImgToControl = algorithm->aImgLabels;

                        algorithm->AllocateMemory();
                        nLabelsToControl = algorithm->PerformLabeling();
                        algorithm->DeallocateMemory();

                        normalizeLabels(labeledImgToControl);
                        if (nLabelsCorrect != nLabelsToControl || !compareMat(labeledImgCorrect, labeledImgToControl)) {
                            stats[j] = false;
                            firstFail[j] = inputPath + kPathSeparator + datasets[i] + kPathSeparator + filename;
                            if (adjacent_find(stats.begin(), stats.end(), not_equal_to<int>()) == stats.end()) {
                                stop = true;
                                break;
                            }
                        }
                    }
                    catch (...) {
                        cerr << "errore\n" << endl;
                    }
                }
                ++j;
                // For all the Algorithms in the array
            }
        }// END WHILE (LIST OF IMAGES)
        pBar.End();
    }// END FOR (LIST OF DATASETS)

    if (checkPerform) {
        unsigned j = 0;
        for (const auto& algoName : CCLAlgorithms) {
            if (stats[j])
                cout << "\"" << algoName << "\" is correct!" << endl;
            else
                cout << "\"" << algoName << "\" is not correct, it first fails on " << firstFail[j] << endl;
            ++j;
        }
    }
    else {
        cout << "Unable to perform check, skipped" << endl;
    }
}

//This function take a Mat1d of results and save it on specified outputstream
void saveBroadOutputResults(const Mat1d& results, const string& oFilename, vector<String>& CCLAlgorithms, const bool& writeNLabels, const Mat1i& labels, const vector<pair<string, bool>>& filenames)
{
    ofstream os(oFilename);
    if (!os.is_open()) {
        cout << "Unable to save middle results" << endl;
        return;
    }

    // To set heading file format
    os << "#";
    for (const auto& algoName : CCLAlgorithms) {
        os << "\t" << algoName;
        writeNLabels ? os << "\t" << "n_label" : os << "";
    }
    os << endl;
    // To set heading file format

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algoName : CCLAlgorithms) {
                os << results(files, i) << "\t";
                writeNLabels ? os << labels(files, i) << "\t" : os << "";
                ++i;
            }
            os << endl;
        }
    }
}

string AverageTest(vector<String>& CCLAlgorithms, Mat1d& allRes, const unsigned& algPos, const string& inputPath, const string& inputFolder, const string& inputTxt, const string& gnuplotScriptExtension, string& outputPath, string& latexFolder, string& colorsFolder, const bool& saveMiddleResults, const unsigned& nTest, const string& middleFolder, const bool& writeNLabels = true, const bool& outputColors = true)
{
    string outputFolder = inputFolder,
        completeOutputPath = outputPath + kPathSeparator + outputFolder,
        gnuplotScript = inputFolder + gnuplotScriptExtension,
        outputBroadResults = inputFolder + "_results.txt",
        middleFile = inputFolder + "_run",
        outputAverageResults = inputFolder + "_average.txt",
        outputGraph = outputFolder + terminalExtension,
        outputGraphBw = outputFolder + "_bw" + terminalExtension,
        middleOutFolder = completeOutputPath + kPathSeparator + middleFolder,
        outColorFolder = outputPath + kPathSeparator + outputFolder + kPathSeparator + colorsFolder;

    unsigned numberOfDecimalDigitToDisplayInGraph = 2;

    // Creation of output path
    if (!dirExists(completeOutputPath.c_str()))
        if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
            return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);

    if (outputColors) {
        // Creation of color output path
        if (!dirExists(outColorFolder.c_str()))
            if (0 != std::system(("mkdir " + outColorFolder).c_str()))
                return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
    }

    if (saveMiddleResults) {
        if (!dirExists(middleOutFolder.c_str()))
            if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
                return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);
    }

    string isPath = inputPath + kPathSeparator + inputFolder + kPathSeparator + inputTxt,
        osPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputBroadResults,
        averageOsPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputAverageResults;

    // For AVERAGES RESULT
    ofstream averageOs(averageOsPath);
    if (!averageOs.is_open())
        return ("Averages_Test on '" + inputFolder + "': Unable to open " + averageOsPath);
    // For LIST OF INPUT IMAGES
    ifstream is(isPath);
    if (!is.is_open())
        return ("Averages_Test on '" + inputFolder + "': Unable to open " + isPath);

    // To save list of filename on which CLLAlgorithms must be tested
    vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
    string filename;
    while (getline(is, filename)) {
        // To delete eventual carriage return in the file name (especially designed for windows file newline format)
        size_t found;
        do {
            // The while cycle is probably unnecessary
            found = filename.find("\r");
            if (found != string::npos)
                filename.erase(found, 1);
        } while (found != string::npos);
        // Add purified file name in the vector
        filenames.push_back(make_pair(filename, true));
    }
    is.close();

    // Number of files
    int fileNumber = filenames.size();

    // To save middle/min and average results;
    Mat1d minRes(fileNumber, CCLAlgorithms.size(), numeric_limits<double>::max());
    Mat1d currentRes(fileNumber, CCLAlgorithms.size(), numeric_limits<double>::max());
    Mat1i labels(fileNumber, CCLAlgorithms.size(), 0);
    vector<pair<double, uint16_t>> suppAverage(CCLAlgorithms.size(), make_pair(0.0, 0));

    ProgressBar pBar(fileNumber);
    pBar.Start();

    // Test is executed nTest times
    for (unsigned test = 0; test < nTest; ++test) {
        // Count number of lines to display "progress bar"
        unsigned currentNumber = 0;

        // For every file in list
        for (unsigned file = 0; file < filenames.size(); ++file) {
            filename = filenames[file].first;

            // Display "progress bar"
            //if (currentNumber * 100 / fileNumber != (currentNumber - 1) * 100 / fileNumber)
            //{
            //    cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << fileNumber << "         \r";
            //    fflush(stdout);
            //}
            pBar.Display(currentNumber++, test + 1);

            Mat1b binaryImg;

            if (!getBinaryImage(inputPath + kPathSeparator + inputFolder + kPathSeparator + filename, labeling::aImg)) {
                if (filenames[file].second)
                    cout << "'" + filename + "' does not exist" << endl;
                filenames[file].second = false;
                continue;
            }

            unsigned i = 0;
            // For all the Algorithms in the array
            for (const auto& algoName : CCLAlgorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data.at(algoName);

                // This variables need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                unsigned nLabels;
                Mat3b imgColors;

                // Perform current algorithm on current image and save result.
                algorithm->AllocateMemory();

                algorithm->perf.start();
                nLabels = algorithm->PerformLabeling();
                algorithm->perf.stop();

                algorithm->DeallocateMemory();

                // Save number of labels (we reasonably supposed that labels's number is the same on every #test so only the first time we save it)
                if (test == 0)
                    labels(file, i) = nLabels;

                // Save time results
                currentRes(file, i) = algorithm->perf.last();
                if (algorithm->perf.last() < minRes(file, i))
                    minRes(file, i) = algorithm->perf.last();

                // If 'at_colorLabels' is enabled only the fisrt time (test == 0) the output is saved
                if (test == 0 && outputColors) {
                    // Remove gnuplot escape character from output filename
                    /*string algName = (*it).second;
                    algName.erase(std::remove(algName.begin(), algName.end(), '\\'), algName.end());
*/
                    normalizeLabels(algorithm->aImgLabels);
                    colorLabels(algorithm->aImgLabels, imgColors);
                    imwrite(outColorFolder + kPathSeparator + filename + "_" + algoName + ".png", imgColors);
                }
                ++i;
            }// END ALGORITHMS FOR
        } // END FILES FOR.

          // To display "progress bar"
        //cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << fileNumber << "         \r";
        //fflush(stdout);

        // Save middle results if necessary (flag 'averageSaveMiddleTests' enable)
        if (saveMiddleResults) {
            string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
            saveBroadOutputResults(currentRes, middleOut, CCLAlgorithms, writeNLabels, labels, filenames);
        }
    }// END TESTS FOR

    pBar.End(nTest);

    // To write in a file min results
    saveBroadOutputResults(minRes, osPath, CCLAlgorithms, writeNLabels, labels, filenames);

    // To calculate average times and write it on the specified file
    for (int r = 0; r < minRes.rows; ++r) {
        for (int c = 0; c < minRes.cols; ++c) {
            if (minRes(r, c) != numeric_limits<double>::max()) {
                suppAverage[c].first += minRes(r, c);
                suppAverage[c].second++;
            }
        }
    }

    averageOs << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << endl;
    for (unsigned i = 0; i < CCLAlgorithms.size(); ++i) {
        // For all the Algorithms in the array
        allRes(algPos, i) = suppAverage[i].first / suppAverage[i].second;
        averageOs << CCLAlgorithms[i] << "\t" << suppAverage[i].first / suppAverage[i].second << "\t";
        averageOs << std::fixed << std::setprecision(numberOfDecimalDigitToDisplayInGraph) << suppAverage[i].first / suppAverage[i].second << endl;
    }

    // GNUPLOT SCRIPT

    //Retrieve info about the current machine
    systemInfo info;

    //replace the . with _ for filenames
    pair<string, string> compiler(systemInfo::getCompiler());
    std::replace(compiler.first.begin(), compiler.first.end(), '.', '_');
    std::replace(compiler.second.begin(), compiler.second.end(), '.', '_');

    string scriptos_path = outputPath + kPathSeparator + outputFolder + kPathSeparator + gnuplotScript;
    ofstream scriptOs(scriptos_path);
    if (!scriptOs.is_open())
        return ("Averages_Test on '" + inputFolder + "': Unable to create " + scriptos_path);

    scriptOs << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
    scriptOs << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplotScript << "' if you want to run it" << endl << endl;

    scriptOs << "reset" << endl;
    scriptOs << "cd '" << completeOutputPath << "\'" << endl;
    scriptOs << "set grid ytic" << endl;
    scriptOs << "set grid" << endl << endl;

    scriptOs << "# " << outputFolder << "(COLORS)" << endl;
    scriptOs << "set output \"" + outputGraph + "\"" << endl;

    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " colors" << endl;
    scriptOs << "set terminal " << terminal << " enhanced color font ',15'" << endl << endl;

    scriptOs << "# Graph style" << endl;
    scriptOs << "set style data histogram" << endl;
    scriptOs << "set style histogram cluster gap 1" << endl;
    scriptOs << "set style fill solid 0.25 border -1" << endl;
    scriptOs << "set boxwidth 0.9" << endl << endl;

    scriptOs << "# Get stats to set labels" << endl;
    scriptOs << "stats \"" << outputAverageResults << "\" using 2 nooutput" << endl;
    scriptOs << "ymax = STATS_max + (STATS_max/100)*10" << endl;
    scriptOs << "xw = 0" << endl;
    scriptOs << "yw = (ymax)/22" << endl << endl;

    scriptOs << "# Axes labels" << endl;
    scriptOs << "set xtic rotate by -45 scale 0" << endl;
    scriptOs << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptOs << "# Axes range" << endl;
    scriptOs << "set yrange[0:ymax]" << endl;
    scriptOs << "set xrange[*:*]" << endl << endl;

    scriptOs << "# Legend" << endl;
    scriptOs << "set key off" << endl << endl;

    scriptOs << "# Plot" << endl;
    scriptOs << "plot \\" << endl;
    scriptOs << "'" + outputAverageResults + "' using 2:xtic(1), '" << outputAverageResults << "' using ($0 - xw) : ($2 + yw) : (stringcolumn(3)) with labels" << endl << endl;

    scriptOs << "# Replot in latex folder" << endl;
    scriptOs << "set title \"\"" << endl << endl;
    scriptOs << "set output \'.." << kPathSeparator << latexFolder << kPathSeparator << compiler.first + compiler.second + outputGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# " << outputFolder << "(BLACK AND WHITE)" << endl;
    scriptOs << "set output \"" + outputGraphBw + "\"" << endl;

    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " black and white" << endl;
    scriptOs << "set terminal " << terminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    scriptOs << "exit gnuplot" << endl;

    averageOs.close();
    scriptOs.close();
    // GNUPLOT SCRIPT

    if (0 != std::system(("gnuplot " + completeOutputPath + kPathSeparator + gnuplotScript).c_str()))
        return ("Averages_Test on '" + inputFolder + "': Unable to run gnuplot's script");

    //return ("Averages_Test on '" + inputFolder + "': successfuly done");
    return "";
}

string DensitySizeTest(vector<String>& CCLAlgorithms, const string& inputPath, const string& inputFolder, const string& inputTxt, const string& gnuplotScriptExtension, string& outputPath, string& latexFolder, string& colorsFolder, const bool& saveMiddleResults, const unsigned& nTest, const string& middleFolder, const bool& writeNLabels = true, const bool& outputColors = true)
{
    string outputFolder = inputFolder,
        completeOutputPath = outputPath + kPathSeparator + outputFolder,
        gnuplotScript = inputFolder + gnuplotScriptExtension,
        outputBroadResult = inputFolder + "_results.txt",
        outputSizeResult = "size.txt",
        //output_size_normalized_result = "",
        outputDensityResult = "density.txt",
        outputDensityNormalizedResult = "normalized_density.txt",
        outputSizeGraph = "size" + terminalExtension,
        outputSizeGraphBw = "size_bw" + terminalExtension,
        outputDensityGraph = "density" + terminalExtension,
        outputDensityGraphBw = "density_bw" + terminalExtension,
        outputNormalizationDensityGraph = "normalized_density" + terminalExtension,
        outputNormalizationDensityGraphBw = "normalized_density_bw" + terminalExtension,
        middleFile = inputFolder + "_run",
        middleOutFolder = completeOutputPath + kPathSeparator + middleFolder,
        outColorFolder = outputPath + kPathSeparator + outputFolder + kPathSeparator + colorsFolder,
        outputNull = inputFolder + "_NULL_results.txt";

    // Creation of output path
    if (!dirExists(completeOutputPath.c_str()))
        if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
            return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);

    if (outputColors) {
        // Creation of color output path
        if (!dirExists(outColorFolder.c_str()))
            if (0 != std::system(("mkdir " + outColorFolder).c_str()))
                return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
    }

    if (saveMiddleResults) {
        if (!dirExists(middleOutFolder.c_str()))
            if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
                return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);
    }

    string isPath = inputPath + kPathSeparator + inputFolder + kPathSeparator + inputTxt,
        osPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputBroadResult,
        densityOsPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputDensityResult,
        densityNormalizedOsPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputDensityNormalizedResult,
        sizeOsPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputSizeResult,
        //size_normalized_os_path = outputPath + kPathSeparator + outputFolder + kPathSeparator + output_size_normalized_result,
        NullPath = outputPath + kPathSeparator + outputFolder + kPathSeparator + outputNull;

    // For DENSITY RESULT
    ofstream densityOs(densityOsPath);
    if (!densityOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + densityOsPath);
    // For DENSITY NORMALIZED RESULT
    ofstream densityNormalizedOs(densityNormalizedOsPath);
    if (!densityNormalizedOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + densityNormalizedOsPath);
    // For SIZE RESULT
    ofstream sizeOs(sizeOsPath);
    if (!sizeOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + sizeOsPath);
    // For LIST OF INPUT IMAGES
    ifstream is(isPath);
    if (!is.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to open " + isPath);
    // For NULL LABELING RESULTS
    ofstream NullOs(NullPath);
    if (!NullOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + NullPath);

    // To save list of filename on which CLLAlgorithms must be tested
    vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
    string filename;
    while (getline(is, filename)) {
        // To delete eventual carriage return in the file name (especially designed for windows file newline format)
        size_t found;
        do {
            // The while cycle is probably unnecessary
            found = filename.find("\r");
            if (found != string::npos)
                filename.erase(found, 1);
        } while (found != string::npos);
        filenames.push_back(make_pair(filename, true));
    }
    is.close();

    // Number of files
    int fileNumber = filenames.size();

    // To save middle/min and average results;
    Mat1d minRes(fileNumber, CCLAlgorithms.size(), numeric_limits<double>::max());
    Mat1d currentRes(fileNumber, CCLAlgorithms.size(), numeric_limits<double>::max());
    Mat1i labels(fileNumber, CCLAlgorithms.size(), 0);
    vector<pair<double, uint16_t>> suppAverage(CCLAlgorithms.size(), make_pair(0, 0));

    // To save labeling NULL results
    vector<double> NullLabeling(fileNumber, numeric_limits<double>::max());

    // To set heading file format (SIZE RESULT, DENSITY RESULT)
    //os << "#";
    densityOs << "#Density";
    sizeOs << "#Size";
    densityNormalizedOs << "#DensityNorm";
    for (const auto& algoName : CCLAlgorithms) {
        //os << "\t" << (*it).second;
        //writeNLabels ? os << "\t" << "n_label" : os << "";
        densityOs << "\t" << algoName;
        sizeOs << "\t" << algoName;
        densityNormalizedOs << "\t" << algoName;
    }
    //os << endl;
    densityOs << endl;
    sizeOs << endl;
    densityNormalizedOs << endl;
    // To set heading file format (SIZE RESULT, DENSITY RESULT)

    uint8_t density = 9 /*[0.1,0.9]*/, size = 8 /*[32,64,128,256,512,1024,2048,4096]*/;

    vector<vector<pair<double, uint16_t>>> suppDensity(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
    vector<vector<pair<double, uint16_t>>> suppNormalizedDensity(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
    vector<vector<pair<double, uint16_t>>> suppSize(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));
    //vector<vector<pair<double, uint16_t>>> supp_normalized_size(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));

    // Note that number of random_images is less than 800, this is why the second element of the
    // pair has uint16_t data type. Extern vector represent the algorithms, inner vector represent
    // density for "suppDensity" variable and dimension for "supp_dimension" one. In particular:
    //
    //	FOR "suppDensity" VARIABLE:
    //	INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
    //	INNER_VECTPR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
    //  .. and so on;
    //
    //	SO:
    //	  suppDensity[0][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
    //	  for algorithm in position 0;
    //
    //	  suppDensity[0][1] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
    //	  for algorithm in position 0;
    //
    //	  suppDensity[1][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
    //	  for algorithm in position 1;
    //    .. and so on
    //
    //	FOR "SUP_DIMENSION VARIABLE":
    //	INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_32*32_DIMENSION, COUNT_OF_THAT_IMAGES }
    //	INNER_VECTOR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_64*64_DIMENSION, COUNT_OF_THAT_IMAGES }
    //
    //	view "suppDensity" explanation for more details;
    //
    ProgressBar pBar(fileNumber);
    pBar.Start();

    // Test is execute nTest times
    for (unsigned test = 0; test < nTest; ++test) {
        // Count number of lines to display "progress bar"
        unsigned currentNumber = 0;

        // For every file in list
        for (unsigned file = 0; file < filenames.size(); ++file) {
            filename = filenames[file].first;

            // Display "progress bar"
            //if (currentNumber * 100 / fileNumber != (currentNumber - 1) * 100 / fileNumber)
            //{
            //    cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << fileNumber << "         \r";
            //    fflush(stdout);
            //}
            pBar.Display(currentNumber++, test + 1);

            Mat1b binaryImg;

            if (!getBinaryImage(inputPath + kPathSeparator + inputFolder + kPathSeparator + filename, labeling::aImg)) {
                if (filenames[file].second)
                    cout << "'" + filename + "' does not exist" << endl;
                filenames[file].second = false;
                continue;
            }

            // One time for every test and for every image we execute the NULL labeling and get the minimum

            auto& labelingNULL = LabelingMapSingleton::GetInstance().data.at("labelingNULL");
            labelingNULL->perf.start();
            labelingNULL->PerformLabeling();
            labelingNULL->perf.stop();

            if (labelingNULL->perf.last() < NullLabeling[file]) {
                NullLabeling[file] = labelingNULL->perf.last();
            }
            // One time for every test and for every image we execute the NULL labeling and get the minimum

            unsigned i = 0;
            for (const auto& algoName : CCLAlgorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data.at(algoName);
                // For all the Algorithms in the array

                // This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                Mat1i labeledMat;
                unsigned nLabels;
                Mat3b imgColors;

                // Note that "i" represent the current algorithm's position in vectors "suppDensity" and "supp_dimension"
                algorithm->AllocateMemory();

                algorithm->perf.start();
                nLabels = algorithm->PerformLabeling();
                algorithm->perf.stop();

                algorithm->DeallocateMemory();

                if (test == 0)
                    labels(file, i) = nLabels;

                // Save time results
                currentRes(file, i) = algorithm->perf.last();
                if (algorithm->perf.last() < minRes(file, i))
                    minRes(file, i) = algorithm->perf.last();

                // If 'at_colorLabels' is enable only the fisrt time (test == 0) the output is saved
                if (test == 0 && outputColors) {
                    // Remove gnuplot escape character from output filename
                    /*string algName = (*it).second;
                    algName.erase(std::remove(algName.begin(), algName.end(), '\\'), algName.end());
*/
                    normalizeLabels(labeledMat);
                    colorLabels(labeledMat, imgColors);
                    imwrite(outColorFolder + kPathSeparator + filename + "_" + algoName + ".png", imgColors);
                }
                ++i;
            }// END ALGORTIHMS FOR
        } // END FILES FOR
          // To display "progress bar"
        /*cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << fileNumber << "         \r";
        fflush(stdout);*/

        // Save middle results if necessary (flag 'averageSaveMiddleTests' enable)
        if (saveMiddleResults) {
            string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
            saveBroadOutputResults(currentRes, middleOut, CCLAlgorithms, writeNLabels, labels, filenames);
        }
    }// END TEST FOR

    pBar.End(nTest);

    // To wirte in a file min results
    saveBroadOutputResults(minRes, osPath, CCLAlgorithms, writeNLabels, labels, filenames);

    // To sum min results, in the correct manner, before make average
    for (unsigned files = 0; files < filenames.size(); ++files) {
        // Note that files correspond to minRes rows
        for (int c = 0; c < minRes.cols; ++c) {
            // Add current time to "suppDensity" and "suppSize" in the correct position
            if (isdigit(filenames[files].first[0]) && isdigit(filenames[files].first[1]) && isdigit(filenames[files].first[2]) && filenames[files].second) {
                if (minRes(files, c) != numeric_limits<double>::max()) { // superfluous test?
                  // For density graph
                    suppDensity[c][ctoi(filenames[files].first[1])].first += minRes(files, c);
                    suppDensity[c][ctoi(filenames[files].first[1])].second++;

                    // For normalized desnity graph
                    suppNormalizedDensity[c][ctoi(filenames[files].first[1])].first += (minRes(files, c)) / (NullLabeling[files]);
                    suppNormalizedDensity[c][ctoi(filenames[files].first[1])].second++;

                    // For dimension graph
                    suppSize[c][ctoi(filenames[files].first[0])].first += minRes(files, c);
                    suppSize[c][ctoi(filenames[files].first[0])].second++;
                }
            }
            // Add current time to "suppDensity" and "suppSize" in the correct position
        }
    }
    // To sum min results, in the correct manner, before make average

    // To calculate average times
    vector<vector<long double>> densityAverage(CCLAlgorithms.size(), vector<long double>(density)), sizeAverage(CCLAlgorithms.size(), vector<long double>(size));
    vector<vector<long double>> densityNormalizedAverage(CCLAlgorithms.size(), vector<long double>(density));
    for (unsigned i = 0; i < CCLAlgorithms.size(); ++i) {
        // For all algorithms
        for (unsigned j = 0; j < densityAverage[i].size(); ++j) {
            // For all density and normalized density
            if (suppDensity[i][j].second != 0) {
                densityAverage[i][j] = suppDensity[i][j].first / suppDensity[i][j].second;
                densityNormalizedAverage[i][j] = suppNormalizedDensity[i][j].first / suppNormalizedDensity[i][j].second;
            }
            else {
                // If there is no element with this density characteristic the average value is set to zero
                densityAverage[i][j] = 0.0;
                densityNormalizedAverage[i][j] = 0.0;
            }
        }
        for (unsigned j = 0; j < sizeAverage[i].size(); ++j) {
            // For all size
            if (suppSize[i][j].second != 0)
                sizeAverage[i][j] = suppSize[i][j].first / suppSize[i][j].second;
            else
                sizeAverage[i][j] = 0.0;  // If there is no element with this size characteristic the average value is set to zero
        }
    }
    // To calculate average

    // To write density result on specified file
    for (unsigned i = 0; i < density; ++i) {
        // For every density
        if (densityAverage[0][i] == 0.0) { // Check it only for the first algorithm (it is the same for the others)
            densityOs << "#"; // It means that there is no element with this density characteristic
            densityNormalizedOs << "#"; // It means that there is no element with this density characteristic
        }
        densityOs << ((float)(i + 1) / 10) << "\t"; //Density value
        densityNormalizedOs << ((float)(i + 1) / 10) << "\t"; //Density value
        for (unsigned j = 0; j < densityAverage.size(); ++j) {
            // For every algorithm
            densityOs << densityAverage[j][i] << "\t";
            densityNormalizedOs << densityNormalizedAverage[j][i] << "\t";
        }
        densityOs << endl; // End of current line (current density)
        densityNormalizedOs << endl; // End of current line (current density)
    }
    // To write density result on specified file

    // To set sizes's label
    vector <pair<unsigned, double>> supp_size_labels(size, make_pair(0, 0));

    // To write size result on specified file
    for (unsigned i = 0; i < size; ++i) {
        // For every size
        if (sizeAverage[0][i] == 0.0) // Check it only for the first algorithm (it is the same for the others)
            sizeOs << "#"; // It means that there is no element with this size characteristic
        supp_size_labels[i].first = (int)(pow(2, i + 5));
        supp_size_labels[i].second = sizeAverage[0][i];
        sizeOs << (int)pow(supp_size_labels[i].first, 2) << "\t"; //Size value
        for (unsigned j = 0; j < sizeAverage.size(); ++j) {
            // For every algorithms
            sizeOs << sizeAverage[j][i] << "\t";
        }
        sizeOs << endl; // End of current line (current size)
    }
    // To write size result on specified file

    // To write NULL result on specified file
    for (unsigned i = 0; i < filenames.size(); ++i) {
        NullOs << filenames[i].first << "\t" << NullLabeling[i] << endl;
    }
    // To write NULL result on specified file

    // GNUPLOT SCRIPT

    //Retrieve info about the current machine
    systemInfo info;

    //replace the . with _ for filenames
    pair<string, string> compiler(systemInfo::getCompiler());
    replace(compiler.first.begin(), compiler.first.end(), '.', '_');
    replace(compiler.second.begin(), compiler.second.end(), '.', '_');

    string scriptos_path = outputPath + kPathSeparator + outputFolder + kPathSeparator + gnuplotScript;
    ofstream scriptOs(scriptos_path);
    if (!scriptOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + scriptos_path);

    scriptOs << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
    scriptOs << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplotScript << "' if you want to run it" << endl << endl;

    scriptOs << "reset" << endl;
    scriptOs << "cd '" << completeOutputPath << "\'" << endl;
    scriptOs << "set grid" << endl << endl;

    // DENSITY
    scriptOs << "# DENSITY GRAPH (COLORS)" << endl << endl;

    scriptOs << "set output \"" + outputDensityGraph + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " colors" << endl;
    scriptOs << "set terminal " << terminal << " enhanced color font ',15'" << endl << endl;

    scriptOs << "# Axes labels" << endl;
    scriptOs << "set xlabel \"Density\"" << endl;
    scriptOs << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptOs << "# Axes range" << endl;
    scriptOs << "set xrange [0:1]" << endl;
    scriptOs << "set yrange [*:*]" << endl;
    scriptOs << "set logscale y" << endl << endl;

    scriptOs << "# Legend" << endl;
    scriptOs << "set key left top nobox spacing 2 font ', 8'" << endl << endl;

    scriptOs << "# Plot" << endl;
    scriptOs << "plot \\" << endl;
    vector<String>::iterator it; // I need it after the cycle
    unsigned i = 2;
    for (it = CCLAlgorithms.begin(); it != (CCLAlgorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputDensityResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputDensityResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# Replot in latex folder" << endl;
    scriptOs << "set title \"\"" << endl << endl;
    scriptOs << "set output \'.." << kPathSeparator << latexFolder << kPathSeparator << compiler.first + compiler.second + outputDensityGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# DENSITY GRAPH (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputDensityGraphBw + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " black and white" << endl;
    scriptOs << "set terminal " << terminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    // DENSITY NORMALIZED
    scriptOs << "#NORMALIZED DENSITY GRAPH (COLORS)" << endl << endl;

    scriptOs << "set output \"" + outputNormalizationDensityGraph + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " colors" << endl;
    scriptOs << "set terminal " << terminal << " enhanced color font ',15'" << endl << endl;

    scriptOs << "# Axes labels" << endl;
    scriptOs << "set xlabel \"Density\"" << endl;
    scriptOs << "set ylabel \"Normalized Execution Time [ms]\"" << endl << endl;

    scriptOs << "# Axes range" << endl;
    scriptOs << "set xrange [0:1]" << endl;
    scriptOs << "set yrange [*:*]" << endl;
    scriptOs << "set logscale y" << endl << endl;

    scriptOs << "# Legend" << endl;
    scriptOs << "set key left top nobox spacing 2 font ', 8'" << endl << endl;

    scriptOs << "# Plot" << endl;
    scriptOs << "plot \\" << endl;
    //vector<pair<CCLPointer, string>>::iterator it; // I need it after the cycle
    //unsigned i = 2;
    i = 2;
    for (it = CCLAlgorithms.begin(); it != (CCLAlgorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputDensityNormalizedResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputDensityNormalizedResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# NORMALIZED DENSITY GRAPH (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputNormalizationDensityGraphBw + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " black and white" << endl;
    scriptOs << "set terminal " << terminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    // SIZE
    scriptOs << "# SIZE GRAPH (COLORS)" << endl << endl;

    scriptOs << "set output \"" + outputSizeGraph + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " colors" << endl;
    scriptOs << "set terminal " << terminal << " enhanced color font ',15'" << endl << endl;

    scriptOs << "# Axes labels" << endl;
    scriptOs << "set xlabel \"Pixels\"" << endl;
    scriptOs << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptOs << "# Axes range" << endl;
    scriptOs << "set format x \"10^{%L}\"" << endl;
    scriptOs << "set xrange [100:100000000]" << endl;
    scriptOs << "set yrange [*:*]" << endl;
    scriptOs << "set logscale xy 10" << endl << endl;

    scriptOs << "# Legend" << endl;
    scriptOs << "set key left top nobox spacing 2 font ', 8'" << endl;

    scriptOs << "# Plot" << endl;
    //// Set Labels
    //for (unsigned i=0; i < supp_size_labels.size(); ++i){
    //	if (supp_size_labels[i].second != 0){
    //		scriptOs << "set label " << i+1 << " \"" << supp_size_labels[i].first << "x" << supp_size_labels[i].first << "\" at " << pow(supp_size_labels[i].first,2) << "," << supp_size_labels[i].second << endl;
    //	}
    //	else{
    //		// It means that there is no element with this size characteristic so this label is not necessary
    //	}
    //}
    //// Set Labels
    scriptOs << "plot \\" << endl;
    //vector<pair<CCLPointer, string>>::iterator it; // I need it after the cycle
    //unsigned i = 2;
    i = 2;
    for (it = CCLAlgorithms.begin(); it != (CCLAlgorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputSizeResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputSizeResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# Replot in latex folder" << endl;
    scriptOs << "set title \"\"" << endl << endl;
    scriptOs << "set output \'.." << kPathSeparator << latexFolder << kPathSeparator << compiler.first + compiler.second + outputSizeGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# SIZE (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputSizeGraphBw + "\"" << endl;
    scriptOs << "set title \" {/:Bold CPU}: " + info.getCpuBrand() + " {/:Bold BUILD}: " + info.getBuild() + " {/:Bold OS}: " + info.getOs() + "\" font ', 11'" << endl << endl;

    scriptOs << "# " << terminal << " black and white" << endl;
    scriptOs << "set terminal " << terminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    scriptOs << "exit gnuplot" << endl;

    densityOs.close();
    sizeOs.close();
    scriptOs.close();
    // GNUPLOT SCRIPT

    if (0 != std::system(("gnuplot " + completeOutputPath + kPathSeparator + gnuplotScript).c_str()))
        return ("Density_Size_Test on '" + inputFolder + "': Unable to run gnuplot's script");
    //return ("Density_Size_Test on '" + outputFolder + "': successfully done");
    return ("");
}

string MemoryTest(vector<String>& CCLMemAlgorithms, Mat1d& algoAverageAccesses, const string& inputPath, const string& inputFolder, const string& inputTxt, string& outputPath)
{
    string outputFolder = inputFolder,
        completeOutputPath = outputPath + kPathSeparator + outputFolder;

    unsigned numberOfDecimalDigitToDisplayInGraph = 2;

    // Creation of output path
    if (!makeDir(completeOutputPath))
        return ("Memory_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);

    string isPath = inputPath + kPathSeparator + inputFolder + kPathSeparator + inputTxt;

    // For LIST OF INPUT IMAGES
    ifstream is(isPath);
    if (!is.is_open())
        return ("Memory_Test on '" + inputFolder + "': Unable to open " + isPath);

    // (TODO move this code into a function)
    // To save list of filename on which CLLAlgorithms must be tested
    vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
    string filename;
    while (getline(is, filename)) {
        // To delete eventual carriage return in the file name (especially designed for windows file newline format)
        size_t found;
        do {
            // The while cycle is probably unnecessary
            found = filename.find("\r");
            if (found != string::npos)
                filename.erase(found, 1);
        } while (found != string::npos);
        // Add purified file name in the vector
        filenames.push_back(make_pair(filename, true));
    }
    is.close();
    // (TODO move this code into a function)

    // Number of files
    int fileNumber = filenames.size();

    // To store average memory accesses (one column for every data structure type: col 1 -> BINARY_MAT, col 2 -> LABELED_MAT, col 3 -> EQUIVALENCE_VET, col 0 -> OTHER)
    algoAverageAccesses = Mat1d(Size(MD_SIZE, CCLMemAlgorithms.size()), 0);

    // Count number of lines to display "progress bar"
    unsigned currentNumber = 0;

    unsigned totTest = 0; // To count the real number of image on which labeling will be applied
                      // For every file in list

    for (unsigned file = 0; file < filenames.size(); ++file) {
        filename = filenames[file].first;

        // Display "progress bar"
        if (currentNumber * 100 / fileNumber != (currentNumber - 1) * 100 / fileNumber) {
            cout << currentNumber << "/" << fileNumber << "         \r";
            fflush(stdout);
        }
        currentNumber++;

        Mat1b binaryImg;

        if (!getBinaryImage(inputPath + kPathSeparator + inputFolder + kPathSeparator + filename, labeling::aImg)) {
            if (filenames[file].second)
                cout << "'" + filename + "' does not exist" << endl;
            filenames[file].second = false;
            continue;
        }

        totTest++;
        unsigned i = 0;
        // For all the Algorithms in the list
        for (const auto& algoName : CCLMemAlgorithms) {
            auto& algorithm = LabelingMapSingleton::GetInstance().data.at(algoName);

            // The following data structure is used to get the memory access matrices
            vector<unsigned long int> accessesVal; // Rows represents algorithms and columns represent data structures
            unsigned nLabels;

            nLabels = algorithm->PerformLabelingMem(accessesVal);

            // For every data structure "returned" by the algorithm
            for (size_t a = 0; a < accessesVal.size(); ++a) {
                algoAverageAccesses(i, a) += accessesVal[a];
            }
            ++i;
        }// END ALGORITHMS FOR
    } // END FILES FOR

      // To display "progress bar"
    cout << currentNumber << "/" << fileNumber << "         \r";
    fflush(stdout);

    // To calculate average memory accesses
    for (int r = 0; r < algoAverageAccesses.rows; ++r) {
        for (int c = 0; c < algoAverageAccesses.cols; ++c) {
            algoAverageAccesses(r, c) /= totTest;
        }
    }

    return ("Memory_Test on '" + inputFolder + "': successfuly done");
}

int main()
{
    cout << "NewYACCLAB\n";

    const string configFile = "config.yaml";
    // Configuration file
    FileStorage fs;

    //redirect cv exceptions
    cvRedirectError(redirectCvError);

    try {
        fs.open(configFile, FileStorage::READ);
    }
    catch (const cv::Exception&) {
        //Error redirected
        return EXIT_FAILURE;
    }

    //return EXIT_SUCCESS;

    if (!fs.isOpened()) {
        cerr << "failed to open " << configFile << endl;
        return EXIT_FAILURE;
    }

    vector<vector<String>> CCLAlgorithmss;
    read(fs["algorithms"], CCLAlgorithmss);


    return EXIT_SUCCESS;

    // Flags to customize output format. False by default.
    bool performCheck8Connectivity = readBool(fs["perform"]["check8Connectivity"]),
        performAverage = readBool(fs["perform"]["average"]),
        performDensity = readBool(fs["perform"]["density"]),
        performMemory = readBool(fs["perform"]["memory"]);

    bool averageColorLabels = readBool(fs["colorLabels"]["average"]),
        densityColorLabels = readBool(fs["colorLabels"]["density"]),
        writeNLabels = readBool(fs["writeNLabels"]),
        averageSaveMiddleTests = readBool(fs["saveMiddleTests"]["average"]),
        densitySaveMiddleTests = readBool(fs["saveMiddleTests"]["density"]);

    // Number of tests
    uint8_t averageTestsNumber = static_cast<int>(fs["testsNumber"]["average"]),
        densityTestsNumber = static_cast<int>(fs["testsNumber"]["density"]);

    string inputTxt = "files.txt",             /* Files which contain list of images's name on which CCLAlgorithms are tested */
        gnuplotScriptExtension = ".gnuplot",  /* Extension of gnuplot scripts*/
        colorsFolder = "colors",
        middleFolder = "middle_results",
        latexFile = "averageResults.tex",
        latexCharts = "averageCharts.tex",
        latexMemoryFile = "memoryAccesses.tex",
        outputPath = (string)fs["paths"]["output"], /* Folder on which result are stored */
        inputPath = (string)fs["paths"]["input"],   /* Folder on which datasets are placed */
        latexFolder = "latex";

    // List of dataset on which CCLA are checked
    vector<String> checkDatasets(fs["checkDatasets"].size());

    // List of dataset on which CCLA are memory checked
    vector<String> memoryDatasets(fs["memoryDatasets"].size());

    // Dataset to use for density tests
    vector<String> densityDatasets = { "test_random" };

    // Lists of dataset on which CCLA are tested: one list for every type of test
    vector<String> averageDatasets(fs["averageDatasets"].size());

    // Lists of 'STANDARD' algorithms to check and/or test
    vector<String> CCLAlgorithms(fs["algorithms"].size());

    // Read list of parameters from config file
    read(fs["checkDatasets"], checkDatasets);
    read(fs["memoryDatasets"], memoryDatasets);
    read(fs["averageDatasets"], averageDatasets);
    read(fs["algorithms"], CCLAlgorithms);

    // Release FileStorage
    fs.release();

    if (CCLAlgorithms.size() == 0) {
        cerr << "'Algorithms' field must not be empty" << endl;
        return 1;
    }

    // Create an output directory named with current datetime
    string datetime = getDatetime();
    outputPath += "_" + datetime;
    replace(outputPath.begin(), outputPath.end(), ' ', '_');
    replace(outputPath.begin(), outputPath.end(), ':', '.');

    // Create output directory
    if (!makeDir(outputPath))
        return 1;

    //Create a directory with all the charts
    string latexPath = outputPath + kPathSeparator + latexFolder;
    if ((performAverage || performDensity) && !makeDir(latexPath)) {
        cout << ("Cannot create the directory" + latexPath);
        return 1;
    }

    // hide cursor from console
    HideConsoleCursor();

    // Check if algorithms are correct
    if (performCheck8Connectivity) {
        //cout << "CHECK ALGORITHMS ON 8-CONNECTIVITY: " << endl;
        TitleBar tBar("CHECK ALGORITHMS ON 8-CONNECTIVITY");
        tBar.Start();
        CheckAlgorithms(CCLAlgorithms, checkDatasets, inputPath, inputTxt);
        tBar.End();
    }

    // Test Algorithms with different input type and different output format, and show execution result
    // AVERAGES TEST
    Mat1d allRes(averageDatasets.size(), CCLAlgorithms.size(), numeric_limits<double>::max()); // We need it to save average results and generate latex table

    if (performAverage) {
        //cout << endl << "AVERAGE TESTS: " << endl;
        TitleBar tBar("AVERAGE TESTS");
        tBar.Start();
        if (CCLAlgorithms.size() == 0) {
            cout << "ERROR: no algorithms, average tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < averageDatasets.size(); ++i) {
                cout << "Averages_Test on '" << averageDatasets[i] << "': starts" << endl;
                cout << AverageTest(CCLAlgorithms, allRes, i, inputPath, averageDatasets[i], inputTxt, gnuplotScriptExtension, outputPath, latexFolder, colorsFolder, averageSaveMiddleTests, averageTestsNumber, middleFolder, writeNLabels, averageColorLabels) << endl;
                //cout << "Averages_Test on '" << averageDatasets[i] << "': ends" << endl << endl;
            }
            generateLatexTable(outputPath, latexFile, allRes, averageDatasets, CCLAlgorithms);
        }
        tBar.End();
    }

    // DENSITY_SIZE_TESTS
    if (performDensity) {
        //cout << endl << "DENSITY_SIZE TESTS: " << endl;
        TitleBar tBar("DENSITY_SIZE TESTS");
        tBar.Start();
        if (CCLAlgorithms.size() == 0) {
            cout << "ERROR: no algorithms, density_size tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < densityDatasets.size(); ++i) {
                cout << "Density_Size_Test on '" << densityDatasets[i] << "': starts" << endl;
                cout << DensitySizeTest(CCLAlgorithms, inputPath, densityDatasets[i], inputTxt, gnuplotScriptExtension, outputPath, latexFolder, colorsFolder, densitySaveMiddleTests, densityTestsNumber, middleFolder, writeNLabels, densityColorLabels) << endl;
                //cout << "Density_Size_Test on '" << densityDatasets[i] << "': ends" << endl << endl;
            }
        }
        tBar.End();
    }

    // GENERATE CHARTS TO INCLUDE IN LATEX
    if (performAverage) {
        vector<String> datasetCharts = averageDatasets;
        // Include density tests if they were performed
        if (performDensity) {
            datasetCharts.push_back("density");
            datasetCharts.push_back("size");
        }
        // Generate the latex file that includes all the generated charts
        generateLatexCharts(outputPath, latexCharts, latexFolder, datasetCharts);
    }

    // MEMORY_TESTS
    if (performMemory) {
        Mat1d accesses;
        cout << endl << "MEMORY TESTS: " << endl;

        //Check which algorithms support Memory Tests
        labeling::aImg = Mat1b();
        vector<String> CCLMemAlgorithms;

        for (const auto& algoName : CCLAlgorithms) {
            auto& algorithm = LabelingMapSingleton::GetInstance().data.at(algoName);
            try {
                algorithm->PerformLabelingMem(vector<unsigned long>());
                //The algorithm is added in CCLMemAlgorithms only if it supports Memory Test
                CCLMemAlgorithms.push_back(algoName);
            }
            catch (const runtime_error& e) {
                cerr << algoName << ": " << e.what() << endl;
            }
        }

        if (CCLAlgorithms.size() == 0) {
            cout << "ERROR: no algorithms, memory tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < memoryDatasets.size(); ++i) {
                cout << endl << "Memory_Test on '" << memoryDatasets[i] << "': starts" << endl;
                cout << MemoryTest(CCLMemAlgorithms, accesses, inputPath, memoryDatasets[i], inputTxt, outputPath) << endl;
                cout << "Memory_Test on '" << memoryDatasets[i] << "': ends" << endl << endl;
                generateMemoryLatexTable(outputPath, latexMemoryFile, accesses, memoryDatasets[i], CCLMemAlgorithms);
            }
        }
    }

    return EXIT_SUCCESS;
}