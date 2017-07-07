// Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#include <cstdint>

#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

//#include "config_data.h"
#include "performance_evaluator.h"
#include "labeling_algorithms.h"
#include "file_manager.h"
#include "progress_bar.h"
#include "memory_tester.h"
#include "system_info.h"
#include "latex_generator.h"
#include "utilities.h"

using namespace std;
using namespace cv;

// To check the correctness of algorithms on datasets specified
void CheckAlgorithms(vector<String>& ccl_algorithms, const vector<String>& datasets, const string& input_path, const string& input_txt)
{
    vector<bool> stats(ccl_algorithms.size(), true); // true if the i-th algorithm is correct, false otherwise
    vector<string> firstFail(ccl_algorithms.size()); // name of the file on which algorithm fails the first time
    bool stop = false; // true if all algorithms are incorrect
    bool checkPerform = false; // true if almost one check was execute

    for (unsigned i = 0; i < datasets.size(); ++i) {
        // For every dataset in check list

        cout << "Test on " << datasets[i] << " starts: " << endl;

        string isPath = input_path + kPathSeparator + datasets[i] + kPathSeparator + input_txt;

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
        int file_number = filenames.size();

        // Count number of lines to display progress bar
        unsigned currentNumber = 0;

        ProgressBar p_bar(file_number);
        p_bar.Start();

        // For every file in list
        for (unsigned file = 0; file < filenames.size() && !stop; ++file) {
            filename = filenames[file].first;

            DeleteCarriageReturn(filename);

            p_bar.Display(currentNumber++);

            if (!GetBinaryImage(input_path + kPathSeparator + datasets[i] + kPathSeparator + filename, Labeling::img_)) {
                cout << "Unable to check on '" + filename + "', file does not exist" << endl;
                continue;
            }

            unsigned nLabelsCorrect, nLabelsToControl;

            // SAUF is the reference (the labels are already normalized)
            auto& SAUF = LabelingMapSingleton::GetInstance().data_.at("SAUF_RemSp");
            nLabelsCorrect = SAUF->PerformLabeling();

            Mat1i& labeledImgCorrect = SAUF->img_labels_;
            //nLabelsCorrect = connectedComponents(binaryImg, labeledImgCorrect, 8, 4, CCL_WU);

            unsigned j = 0;
            for (const auto& algo_name : ccl_algorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data_.at(algo_name);
                checkPerform = true;
                if (stats[j]) {
                    try {
                        Mat1i& labeledImgToControl = algorithm->img_labels_;

                        nLabelsToControl = algorithm->PerformLabeling();

                        NormalizeLabels(labeledImgToControl);
                        const auto diff = CompareMat(labeledImgCorrect, labeledImgToControl);
                        if (nLabelsCorrect != nLabelsToControl || !diff) {
                            stats[j] = false;
                            firstFail[j] = input_path + kPathSeparator + datasets[i] + kPathSeparator + filename;
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
        p_bar.End();
    }// END FOR (LIST OF DATASETS)

    if (checkPerform) {
        unsigned j = 0;
        for (const auto& algo_name : ccl_algorithms) {
            if (stats[j])
                cout << "\"" << algo_name << "\" is correct!" << endl;
            else
                cout << "\"" << algo_name << "\" is not correct, it first fails on " << firstFail[j] << endl;
            ++j;
        }
    }
    else {
        cout << "Unable to perform check, skipped" << endl;
    }
}

//This function take a Mat1d of results and save it on specified outputstream
void saveBroadOutputResults(const Mat1d& results, const string& oFilename, vector<String>& ccl_algorithms, const bool& write_n_labels, const Mat1i& labels, const vector<pair<string, bool>>& filenames)
{
    ofstream os(oFilename);
    if (!os.is_open()) {
        cout << "Unable to save middle results" << endl;
        return;
    }

    // To set heading file format
    os << "#";
    for (const auto& algo_name : ccl_algorithms) {
        os << "\t" << algo_name;
        write_n_labels ? os << "\t" << "n_label" : os << "";
    }
    os << endl;
    // To set heading file format

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algo_name : ccl_algorithms) {
                os << results(files, i) << "\t";
                write_n_labels ? os << labels(files, i) << "\t" : os << "";
                ++i;
            }
            os << endl;
        }
    }
}

string AverageTest(vector<String>& ccl_algorithms, Mat1d& all_res, const unsigned& algPos, const string& input_path, const string& inputFolder, const string& input_txt, const string& gnuplot_script_extension, string& output_path, string& latex_folder, string& colors_folder, const bool& saveMiddleResults, const unsigned& n_test, const string& middle_folder, const bool& write_n_labels = true, const bool& outputColors = true)
{
    string outputFolder = inputFolder,
        completeOutputPath = output_path + kPathSeparator + outputFolder,
        gnuplotScript = inputFolder + gnuplot_script_extension,
        outputBroadResults = inputFolder + "_results.txt",
        middleFile = inputFolder + "_run",
        outputAverageResults = inputFolder + "_average.txt",
        outputGraph = outputFolder + kTerminalExtension,
        outputGraphBw = outputFolder + "_bw" + kTerminalExtension,
        middleOutFolder = completeOutputPath + kPathSeparator + middle_folder,
        outColorFolder = output_path + kPathSeparator + outputFolder + kPathSeparator + colors_folder;

    unsigned numberOfDecimalDigitToDisplayInGraph = 2;

    // Creation of output path
    /*if (!dirExists(completeOutputPath.c_str()))
        if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
            return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);*/
    if (!MakeDir(completeOutputPath)) {
        return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);
    }

    if (outputColors) {
        // Creation of color output path
        /*if (!dirExists(outColorFolder.c_str()))
            if (0 != std::system(("mkdir " + outColorFolder).c_str()))
                return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);*/
        if (!MakeDir(outColorFolder)) {
            return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
        }
    }

    if (saveMiddleResults) {
        /*if (!dirExists(middleOutFolder.c_str()))
            if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
                return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);*/
        if (!MakeDir(middleOutFolder)) {
            return ("Averages_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);
        }
    }

    string isPath = input_path + kPathSeparator + inputFolder + kPathSeparator + input_txt,
        osPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputBroadResults,
        averageOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputAverageResults;

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
    int file_number = filenames.size();

    // To save middle/min and average results;
    Mat1d min_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
    Mat1d current_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
    Mat1i labels(file_number, ccl_algorithms.size(), 0);
    vector<pair<double, uint16_t>> supp_average(ccl_algorithms.size(), make_pair(0.0, 0));

    ProgressBar p_bar(file_number);
    p_bar.Start();

    // Test is executed n_test times
    for (unsigned test = 0; test < n_test; ++test) {
        // Count number of lines to display "progress bar"
        unsigned currentNumber = 0;

        // For every file in list
        for (unsigned file = 0; file < filenames.size(); ++file) {
            filename = filenames[file].first;

            // Display "progress bar"
            //if (currentNumber * 100 / file_number != (currentNumber - 1) * 100 / file_number)
            //{
            //    cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
            //    fflush(stdout);
            //}
            p_bar.Display(currentNumber++, test + 1);

            Mat1b binaryImg;

            if (!GetBinaryImage(input_path + kPathSeparator + inputFolder + kPathSeparator + filename, Labeling::img_)) {
                if (filenames[file].second)
                    cout << "'" + filename + "' does not exist" << endl;
                filenames[file].second = false;
                continue;
            }

            unsigned i = 0;
            // For all the Algorithms in the array
            for (const auto& algo_name : ccl_algorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data_[algo_name];

                // This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                unsigned n_labels;

                // Perform current algorithm on current image and save result.

                algorithm->perf_.start();
                n_labels = algorithm->PerformLabeling();
                algorithm->perf_.stop();

                // Save number of labels (we reasonably supposed that labels's number is the same on every #test so only the first time we save it)
                if (test == 0) {
                    labels(file, i) = n_labels;
                }

                // Save time results
                current_res(file, i) = algorithm->perf_.last();
                if (algorithm->perf_.last() < min_res(file, i)) {
                    min_res(file, i) = algorithm->perf_.last();
                }

                // If 'at_colorLabels' is enabled only the first time (test == 0) the output is saved
                if (outputColors && test == 0) {
                    // Remove gnuplot escape character from output filename
                    /*string alg_name = (*it).second;
                    alg_name.erase(std::remove(alg_name.begin(), alg_name.end(), '\\'), alg_name.end());*/
                    Mat3b imgColors;

                    NormalizeLabels(algorithm->img_labels_);
                    ColorLabels(algorithm->img_labels_, imgColors);
                    imwrite(outColorFolder + kPathSeparator + filename + "_" + algo_name + ".png", imgColors);
                }
                ++i;
            }// END ALGORITHMS FOR
        } // END FILES FOR.

          // To display "progress bar"
        //cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
        //fflush(stdout);

        // Save middle results if necessary (flag 'average_save_middle_tests' enable)
        if (saveMiddleResults) {
            string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
            saveBroadOutputResults(current_res, middleOut, ccl_algorithms, write_n_labels, labels, filenames);
        }
    }// END TESTS FOR

    p_bar.End(n_test);

    // To write in a file min results
    saveBroadOutputResults(min_res, osPath, ccl_algorithms, write_n_labels, labels, filenames);

    // To calculate average times and write it on the specified file
    for (int r = 0; r < min_res.rows; ++r) {
        for (int c = 0; c < min_res.cols; ++c) {
            if (min_res(r, c) != numeric_limits<double>::max()) {
                supp_average[c].first += min_res(r, c);
                supp_average[c].second++;
            }
        }
    }

    averageOs << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << endl;
    for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
        // For all the Algorithms in the array
        all_res(algPos, i) = supp_average[i].first / supp_average[i].second;
        averageOs << ccl_algorithms[i] << "\t" << supp_average[i].first / supp_average[i].second << "\t";
        averageOs << std::fixed << std::setprecision(numberOfDecimalDigitToDisplayInGraph) << supp_average[i].first / supp_average[i].second << endl;
    }

    // GNUPLOT SCRIPT

    //replace the . with _ for filenames
    pair<string, string> compiler(SystemInfo::GetCompiler());
    std::replace(compiler.first.begin(), compiler.first.end(), '.', '_');
    std::replace(compiler.second.begin(), compiler.second.end(), '.', '_');

    string scriptos_path = output_path + kPathSeparator + outputFolder + kPathSeparator + gnuplotScript;
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

    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " colors" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

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
    scriptOs << "set output \'.." << kPathSeparator << latex_folder << kPathSeparator << compiler.first + compiler.second + outputGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# " << outputFolder << "(BLACK AND WHITE)" << endl;
    scriptOs << "set output \"" + outputGraphBw + "\"" << endl;

    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " black and white" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    scriptOs << "exit gnuplot" << endl;

    averageOs.close();
    scriptOs.close();
    // GNUPLOT SCRIPT

    if (0 != std::system(("gnuplot \"" + completeOutputPath + kPathSeparator + gnuplotScript + "\"").c_str()))
        return ("Averages_Test on '" + inputFolder + "': Unable to run gnuplot's script");

    //return ("Averages_Test on '" + inputFolder + "': successfully done");
    return "";
}

string DensitySizeTest(vector<String>& ccl_algorithms, const string& input_path, const string& inputFolder, const string& input_txt, const string& gnuplot_script_extension, string& output_path, string& latex_folder, string& colors_folder, const bool& saveMiddleResults, const unsigned& n_test, const string& middle_folder, const bool& write_n_labels = true, const bool& outputColors = true)
{
    string outputFolder = inputFolder,
        completeOutputPath = output_path + kPathSeparator + outputFolder,
        gnuplotScript = inputFolder + gnuplot_script_extension,
        outputBroadResult = inputFolder + "_results.txt",
        outputSizeResult = "size.txt",
        //output_size_normalized_result = "",
        outputDensityResult = "density.txt",
        outputDensityNormalizedResult = "normalized_density.txt",
        outputSizeGraph = "size" + kTerminalExtension,
        outputSizeGraphBw = "size_bw" + kTerminalExtension,
        outputDensityGraph = "density" + kTerminalExtension,
        outputDensityGraphBw = "density_bw" + kTerminalExtension,
        outputNormalizationDensityGraph = "normalized_density" + kTerminalExtension,
        outputNormalizationDensityGraphBw = "normalized_density_bw" + kTerminalExtension,
        middleFile = inputFolder + "_run",
        middleOutFolder = completeOutputPath + kPathSeparator + middle_folder,
        outColorFolder = output_path + kPathSeparator + outputFolder + kPathSeparator + colors_folder,
        outputNull = inputFolder + "_NULL_results.txt";

    // Creation of output path
    //if (!dirExists(completeOutputPath.c_str()))
    //    if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
    //        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);
    if (!MakeDir(completeOutputPath)) {
        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);
    }

    if (outputColors) {
        // Creation of color output path
        //if (!dirExists(outColorFolder.c_str()))
        //    if (0 != std::system(("mkdir " + outColorFolder).c_str()))
        //        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
        if (!MakeDir(outColorFolder)) {
            return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
        }
    }

    if (saveMiddleResults) {
        /*if (!dirExists(middleOutFolder.c_str()))
            if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
                return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);*/
        if (!MakeDir(middleOutFolder)) {
            return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);
        }
    }

    string isPath = input_path + kPathSeparator + inputFolder + kPathSeparator + input_txt,
        osPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputBroadResult,
        densityOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputDensityResult,
        densityNormalizedOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputDensityNormalizedResult,
        sizeOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputSizeResult,
        //size_normalized_os_path = output_path + kPathSeparator + outputFolder + kPathSeparator + output_size_normalized_result,
        NullPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputNull;

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
    // For labeling_NULL LABELING RESULTS
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
    int file_number = filenames.size();

    // To save middle/min and average results;
    Mat1d min_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
    Mat1d current_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
    Mat1i labels(file_number, ccl_algorithms.size(), 0);
    vector<pair<double, uint16_t>> supp_average(ccl_algorithms.size(), make_pair(0, 0));

    // To save labeling labeling_NULL results
    vector<double> NullLabeling(file_number, numeric_limits<double>::max());

    // To set heading file format (SIZE RESULT, DENSITY RESULT)
    //os << "#";
    densityOs << "#Density";
    sizeOs << "#Size";
    densityNormalizedOs << "#DensityNorm";
    for (const auto& algo_name : ccl_algorithms) {
        //os << "\t" << (*it).second;
        //write_n_labels ? os << "\t" << "n_label" : os << "";
        densityOs << "\t" << algo_name;
        sizeOs << "\t" << algo_name;
        densityNormalizedOs << "\t" << algo_name;
    }
    //os << endl;
    densityOs << endl;
    sizeOs << endl;
    densityNormalizedOs << endl;
    // To set heading file format (SIZE RESULT, DENSITY RESULT)

    uint8_t density = 9 /*[0.1,0.9]*/, size = 8 /*[32,64,128,256,512,1024,2048,4096]*/;

    using vvp = vector<vector<pair<double, uint16_t>>>;
    vvp suppDensity(ccl_algorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
    vvp suppNormalizedDensity(ccl_algorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
    vvp suppSize(ccl_algorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));
    //vector<vector<pair<double, uint16_t>>> supp_normalized_size(ccl_algorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));

    // Note that number of random_images is less than 800, this is why the second element of the
    // pair has uint16_t data_ type. Extern vector represent the algorithms, inner vector represent
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
    ProgressBar p_bar(file_number);
    p_bar.Start();

    // Test is execute n_test times
    for (unsigned test = 0; test < n_test; ++test) {
        // Count number of lines to display "progress bar"
        unsigned currentNumber = 0;

        // For every file in list
        for (unsigned file = 0; file < filenames.size(); ++file) {
            filename = filenames[file].first;

            // Display "progress bar"
            //if (currentNumber * 100 / file_number != (currentNumber - 1) * 100 / file_number)
            //{
            //    cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
            //    fflush(stdout);
            //}
            p_bar.Display(currentNumber++, test + 1);

            Mat1b binaryImg;

            if (!GetBinaryImage(input_path + kPathSeparator + inputFolder + kPathSeparator + filename, Labeling::img_)) {
                if (filenames[file].second)
                    cout << "'" + filename + "' does not exist" << endl;
                filenames[file].second = false;
                continue;
            }

            // One time for every test and for every image we execute the labeling_NULL labeling and get the minimum

            auto& labeling_NULL = LabelingMapSingleton::GetInstance().data_.at("labeling_NULL");
            labeling_NULL->perf_.start();
            labeling_NULL->PerformLabeling();
            labeling_NULL->perf_.stop();

            if (labeling_NULL->perf_.last() < NullLabeling[file]) {
                NullLabeling[file] = labeling_NULL->perf_.last();
            }
            // One time for every test and for every image we execute the labeling_NULL labeling and get the minimum

            unsigned i = 0;
            for (const auto& algo_name : ccl_algorithms) {
                auto& algorithm = LabelingMapSingleton::GetInstance().data_.at(algo_name);
                // For all the Algorithms in the array

                // This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                Mat1i labeledMat;
                unsigned n_labels;
                Mat3b imgColors;

                // Note that "i" represent the current algorithm's position in vectors "suppDensity" and "supp_dimension"

                algorithm->perf_.start();
                n_labels = algorithm->PerformLabeling();
                algorithm->perf_.stop();

                if (test == 0)
                    labels(file, i) = n_labels;

                // Save time results
                current_res(file, i) = algorithm->perf_.last();
                if (algorithm->perf_.last() < min_res(file, i))
                    min_res(file, i) = algorithm->perf_.last();

                // If 'at_colorLabels' is enable only the fisrt time (test == 0) the output is saved
                if (test == 0 && outputColors) {
                    // Remove gnuplot escape character from output filename
                    /*string alg_name = (*it).second;
                    alg_name.erase(std::remove(alg_name.begin(), alg_name.end(), '\\'), alg_name.end());
*/
                    NormalizeLabels(labeledMat);
                    ColorLabels(labeledMat, imgColors);
                    imwrite(outColorFolder + kPathSeparator + filename + "_" + algo_name + ".png", imgColors);
                }
                ++i;
            }// END ALGORTIHMS FOR
        } // END FILES FOR
          // To display "progress bar"
        /*cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
        fflush(stdout);*/

        // Save middle results if necessary (flag 'average_save_middle_tests' enable)
        if (saveMiddleResults) {
            string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
            saveBroadOutputResults(current_res, middleOut, ccl_algorithms, write_n_labels, labels, filenames);
        }
    }// END TEST FOR

    p_bar.End(n_test);

    // To write in a file min results
    saveBroadOutputResults(min_res, osPath, ccl_algorithms, write_n_labels, labels, filenames);

    // To sum min results, in the correct manner, before make average
    for (unsigned files = 0; files < filenames.size(); ++files) {
        // Note that files correspond to min_res rows
        for (int c = 0; c < min_res.cols; ++c) {
            // Add current time to "suppDensity" and "suppSize" in the correct position
            if (isdigit(filenames[files].first[0]) && isdigit(filenames[files].first[1]) && isdigit(filenames[files].first[2]) && filenames[files].second) {
                if (min_res(files, c) != numeric_limits<double>::max()) { // superfluous test?
                  // For density graph
                    suppDensity[c][ctoi(filenames[files].first[1])].first += min_res(files, c);
                    suppDensity[c][ctoi(filenames[files].first[1])].second++;

                    // For normalized desnity graph
                    suppNormalizedDensity[c][ctoi(filenames[files].first[1])].first += (min_res(files, c)) / (NullLabeling[files]);
                    suppNormalizedDensity[c][ctoi(filenames[files].first[1])].second++;

                    // For dimension graph
                    suppSize[c][ctoi(filenames[files].first[0])].first += min_res(files, c);
                    suppSize[c][ctoi(filenames[files].first[0])].second++;
                }
            }
            // Add current time to "suppDensity" and "suppSize" in the correct position
        }
    }
    // To sum min results, in the correct manner, before make average

    // To calculate average times
    vector<vector<long double>> densityAverage(ccl_algorithms.size(), vector<long double>(density)), sizeAverage(ccl_algorithms.size(), vector<long double>(size));
    vector<vector<long double>> densityNormalizedAverage(ccl_algorithms.size(), vector<long double>(density));
    for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
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

    // To write labeling_NULL result on specified file
    for (unsigned i = 0; i < filenames.size(); ++i) {
        NullOs << filenames[i].first << "\t" << NullLabeling[i] << endl;
    }
    // To write labeling_NULL result on specified file

    // GNUPLOT SCRIPT

    //replace the . with _ for filenames
    pair<string, string> compiler(SystemInfo::GetCompiler());
    replace(compiler.first.begin(), compiler.first.end(), '.', '_');
    replace(compiler.second.begin(), compiler.second.end(), '.', '_');

    string scriptos_path = output_path + kPathSeparator + outputFolder + kPathSeparator + gnuplotScript;
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
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " colors" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

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
    for (it = ccl_algorithms.begin(); it != (ccl_algorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputDensityResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputDensityResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# Replot in latex folder" << endl;
    scriptOs << "set title \"\"" << endl << endl;
    scriptOs << "set output \'.." << kPathSeparator << latex_folder << kPathSeparator << compiler.first + compiler.second + outputDensityGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# DENSITY GRAPH (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputDensityGraphBw + "\"" << endl;
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " black and white" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    // DENSITY NORMALIZED
    scriptOs << "#NORMALIZED DENSITY GRAPH (COLORS)" << endl << endl;

    scriptOs << "set output \"" + outputNormalizationDensityGraph + "\"" << endl;
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " colors" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

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
    for (it = ccl_algorithms.begin(); it != (ccl_algorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputDensityNormalizedResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputDensityNormalizedResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# NORMALIZED DENSITY GRAPH (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputNormalizationDensityGraphBw + "\"" << endl;
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " black and white" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    // SIZE
    scriptOs << "# SIZE GRAPH (COLORS)" << endl << endl;

    scriptOs << "set output \"" + outputSizeGraph + "\"" << endl;
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " colors" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

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
    for (it = ccl_algorithms.begin(); it != (ccl_algorithms.end() - 1); ++it, ++i) {
        scriptOs << "\"" + outputSizeResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\" , \\" << endl;
    }
    scriptOs << "\"" + outputSizeResult + "\" using 1:" << i << " with linespoints title \"" + *it + "\"" << endl << endl;

    scriptOs << "# Replot in latex folder" << endl;
    scriptOs << "set title \"\"" << endl << endl;
    scriptOs << "set output \'.." << kPathSeparator << latex_folder << kPathSeparator << compiler.first + compiler.second + outputSizeGraph + "\'" << endl;
    scriptOs << "replot" << endl << endl;

    scriptOs << "# SIZE (BLACK AND WHITE)" << endl << endl;

    scriptOs << "set output \"" + outputSizeGraphBw + "\"" << endl;
    scriptOs << "set title " << GetGnuplotTitle() << endl << endl;

    scriptOs << "# " << kTerminal << " black and white" << endl;
    scriptOs << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

    scriptOs << "replot" << endl << endl;

    scriptOs << "exit gnuplot" << endl;

    densityOs.close();
    sizeOs.close();
    scriptOs.close();
    // GNUPLOT SCRIPT

    if (0 != std::system(("gnuplot \"" + completeOutputPath + kPathSeparator + gnuplotScript + "\"").c_str()))
        return ("Density_Size_Test on '" + inputFolder + "': Unable to run gnuplot's script");
    //return ("Density_Size_Test on '" + outputFolder + "': successfully done");
    return ("");
}

string MemoryTest(vector<String>& ccl_mem_algorithms, Mat1d& algoAverageAccesses, const string& input_path, const string& inputFolder, const string& input_txt, string& output_path)
{
    string outputFolder = inputFolder,
        completeOutputPath = output_path + kPathSeparator + outputFolder;

    unsigned numberOfDecimalDigitToDisplayInGraph = 2;

    // Creation of output path
    if (!MakeDir(completeOutputPath))
        return ("Memory_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);

    string isPath = input_path + kPathSeparator + inputFolder + kPathSeparator + input_txt;

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
    int file_number = filenames.size();

    // To store average memory accesses (one column for every data_ structure type: col 1 -> BINARY_MAT, col 2 -> LABELED_MAT, col 3 -> EQUIVALENCE_VET, col 0 -> OTHER)
    algoAverageAccesses = Mat1d(Size(MD_SIZE, ccl_mem_algorithms.size()), 0);

    // Count number of lines to display "progress bar"
    unsigned currentNumber = 0;

    unsigned totTest = 0; // To count the real number of image on which labeling will be applied
                      // For every file in list

    for (unsigned file = 0; file < filenames.size(); ++file) {
        filename = filenames[file].first;

        // Display "progress bar"
        if (currentNumber * 100 / file_number != (currentNumber - 1) * 100 / file_number) {
            cout << currentNumber << "/" << file_number << "         \r";
            fflush(stdout);
        }
        currentNumber++;

        Mat1b binaryImg;

        if (!GetBinaryImage(input_path + kPathSeparator + inputFolder + kPathSeparator + filename, Labeling::img_)) {
            if (filenames[file].second)
                cout << "'" + filename + "' does not exist" << endl;
            filenames[file].second = false;
            continue;
        }

        totTest++;
        unsigned i = 0;
        // For all the Algorithms in the list
        for (const auto& algo_name : ccl_mem_algorithms) {
            auto& algorithm = LabelingMapSingleton::GetInstance().data_.at(algo_name);

            // The following data_ structure is used to get the memory access matrices
            vector<unsigned long int> accessesVal; // Rows represents algorithms and columns represent data_ structures
            unsigned n_labels;

            n_labels = algorithm->PerformLabelingMem(accessesVal);

            // For every data_ structure "returned" by the algorithm
            for (size_t a = 0; a < accessesVal.size(); ++a) {
                algoAverageAccesses(i, a) += accessesVal[a];
            }
            ++i;
        }// END ALGORITHMS FOR
    } // END FILES FOR

      // To display "progress bar"
    cout << currentNumber << "/" << file_number << "         \r";
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

    // Redirect cv exceptions
    cvRedirectError(RedirectCvError);

    // Read configuration file
    const string config_file = "config.yaml";
    FileStorage fs;
    try {
        fs.open(config_file, FileStorage::READ);
    }
    catch (const cv::Exception&) {
        return EXIT_FAILURE; //Error redirected
    }

    if (!fs.isOpened()) {
        cerr << "Failed to open '" << config_file << "'" << endl;
        return EXIT_FAILURE;
    }

    // Flags to customize output format. False by default.
    bool perform_check_8_connectivity = ReadBool(fs["perform"]["check_8_connectivity"]),
        perform_average = ReadBool(fs["perform"]["average"]),
        perform_density = ReadBool(fs["perform"]["density"]),
        perform_memory = ReadBool(fs["perform"]["memory"]);

    bool average_color_labels = ReadBool(fs["color_labels"]["average"]),
        density_color_labels = ReadBool(fs["color_labels"]["density"]),
        write_n_labels = ReadBool(fs["write_n_labels"]),
        average_save_middle_tests = ReadBool(fs["save_middle_tests"]["average"]),
        density_save_middle_tests = ReadBool(fs["save_middle_tests"]["density"]);

    // Number of tests
    uint8_t average_tests_number = static_cast<int>(fs["tests_number"]["average"]),
        density_tests_number = static_cast<int>(fs["tests_number"]["density"]);

    string input_txt = "files.txt",             /* Files which contain list of images's name on which ccl_algorithms are tested */
        gnuplot_script_extension = ".gnuplot",  /* Extension of gnuplot scripts*/
        colors_folder = "colors",
        middle_folder = "middle_results",
        latex_file = "averageResults.tex",
        latex_charts = "averageCharts.tex",
        latex_memory_file = "memoryAccesses.tex",
        output_path = NormalizePath(fs["paths"]["output"]),  /* Folder on which result are stored */
        input_path = NormalizePath(fs["paths"]["input"]),    /* Folder on which datasets are placed */
        latex_folder = "latex";

    // List of dataset on which CCLA are checked
    vector<String> check_datasets(fs["check_datasets"].size());

    // List of dataset on which CCLA are memory checked
    vector<String> memory_datasets(fs["memory_datasets"].size());

    // Dataset to use for density tests
    vector<String> density_datasets = { "test_random" };

    // Lists of dataset on which CCLA are tested: one list for every type of test
    vector<String> average_datasets(fs["average_datasets"].size());

    // Lists of 'STANDARD' algorithms to check and/or test
    vector<String> ccl_algorithms(fs["algorithms"].size());

    // Read list of parameters from config file
    read(fs["check_datasets"], check_datasets);
    read(fs["memory_datasets"], memory_datasets);
    read(fs["average_datasets"], average_datasets);
    read(fs["algorithms"], ccl_algorithms);

    // Release FileStorage
    fs.release();

    if (ccl_algorithms.size() == 0) {
        cerr << "'Algorithms' field must not be empty" << endl;
        return 1;
    }

    // Create an output directory named with current datetime
    string datetime = GetDatetime();
    replace(datetime.begin(), datetime.end(), ' ', '_');
    replace(datetime.begin(), datetime.end(), ':', '.');
    output_path += kPathSeparator + datetime;

    // Create output directory
    if (!MakeDir(output_path))
        return 1;

    //Create a directory with all the charts
    string latex_path = output_path + kPathSeparator + latex_folder;
    if ((perform_average || perform_density) && !MakeDir(latex_path)) {
        cout << ("Cannot create the directory" + latex_path);
        return 1;
    }

    // hide cursor from console
    HideConsoleCursor();

    // Check if algorithms are correct
    if (perform_check_8_connectivity) {
        //cout << "CHECK ALGORITHMS ON 8-CONNECTIVITY: " << endl;
        TitleBar t_bar("CHECK ALGORITHMS ON 8-CONNECTIVITY");
        t_bar.Start();
        CheckAlgorithms(ccl_algorithms, check_datasets, input_path, input_txt);
        t_bar.End();
    }

    // Test Algorithms with different input type and different output format, and show execution result
    // AVERAGES TEST
    Mat1d all_res(average_datasets.size(), ccl_algorithms.size(), numeric_limits<double>::max()); // We need it to save average results and generate latex table

    if (perform_average) {
        //cout << endl << "AVERAGE TESTS: " << endl;
        TitleBar t_bar("AVERAGE TESTS");
        t_bar.Start();
        if (ccl_algorithms.size() == 0) {
            cout << "ERROR: no algorithms, average tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < average_datasets.size(); ++i) {
                cout << "Averages_Test on '" << average_datasets[i] << "': starts" << endl;
                cout << AverageTest(ccl_algorithms, all_res, i, input_path, average_datasets[i], input_txt, gnuplot_script_extension, output_path, latex_folder, colors_folder, average_save_middle_tests, average_tests_number, middle_folder, write_n_labels, average_color_labels) << endl;
                //cout << "Averages_Test on '" << average_datasets[i] << "': ends" << endl << endl;
            }
            GenerateLatexTable(output_path, latex_file, all_res, average_datasets, ccl_algorithms);
        }
        t_bar.End();
    }

    // DENSITY_SIZE_TESTS
    if (perform_density) {
        //cout << endl << "DENSITY_SIZE TESTS: " << endl;
        TitleBar t_bar("DENSITY_SIZE TESTS");
        t_bar.Start();
        if (ccl_algorithms.size() == 0) {
            cout << "ERROR: no algorithms, density_size tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < density_datasets.size(); ++i) {
                cout << "Density_Size_Test on '" << density_datasets[i] << "': starts" << endl;
                cout << DensitySizeTest(ccl_algorithms, input_path, density_datasets[i], input_txt, gnuplot_script_extension, output_path, latex_folder, colors_folder, density_save_middle_tests, density_tests_number, middle_folder, write_n_labels, density_color_labels) << endl;
                //cout << "Density_Size_Test on '" << density_datasets[i] << "': ends" << endl << endl;
            }
        }
        t_bar.End();
    }

    // GENERATE CHARTS TO INCLUDE IN LATEX
    if (perform_average) {
        vector<String> dataset_charts = average_datasets;
        // Include density tests if they were performed
        if (perform_density) {
            dataset_charts.push_back("density");
            dataset_charts.push_back("size");
        }
        // Generate the latex file that includes all the generated charts
        GenerateLatexCharts(output_path, latex_charts, latex_folder, dataset_charts);
    }

    // MEMORY_TESTS
    if (perform_memory) {
        Mat1d accesses;
        cout << endl << "MEMORY TESTS: " << endl;

        //Check which algorithms support Memory Tests
        Labeling::img_ = Mat1b();
        vector<String> ccl_mem_algorithms;

        for (const auto& algo_name : ccl_algorithms) {
            auto& algorithm = LabelingMapSingleton::GetInstance().data_.at(algo_name);
            try {
                vector<unsigned long> accesses;
                algorithm->PerformLabelingMem(accesses);
                //The algorithm is added in ccl_mem_algorithms only if it supports Memory Test
                ccl_mem_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                cerr << algo_name << ": " << e.what() << endl;
            }
        }

        if (ccl_algorithms.size() == 0) {
            cout << "ERROR: no algorithms, memory tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < memory_datasets.size(); ++i) {
                cout << endl << "Memory_Test on '" << memory_datasets[i] << "': starts" << endl;
                cout << MemoryTest(ccl_mem_algorithms, accesses, input_path, memory_datasets[i], input_txt, output_path) << endl;
                cout << "Memory_Test on '" << memory_datasets[i] << "': ends" << endl << endl;
                GenerateMemoryLatexTable(output_path, latex_memory_file, accesses, memory_datasets[i], ccl_mem_algorithms);
            }
        }
    }

    return EXIT_SUCCESS;
}