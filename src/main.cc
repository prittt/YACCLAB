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

#define _CRT_SECURE_NO_WARNINGS //To suppress 'fopen' opencv warning/bug 

#include <cstdint>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "labeling_algorithms.h"
#include "latex_generator.h"
#include "memory_tester.h"
#include "performance_evaluator.h"
#include "progress_bar.h"
#include "stream_demultiplexer.h"
#include "system_info.h"
#include "utilities.h"
#include "yacclab_tests.h"

using namespace std;
using namespace cv;

//This function take a Mat1d of results and save it on specified outputstream
void saveBroadOutputResults(const Mat1d& results, const path& oFilename, vector<String>& ccl_algorithms, const bool& write_n_labels, const Mat1i& labels, const vector<pair<string, bool>>& filenames)
{
    ofstream os(oFilename.string());
    if (!os.is_open()) {
        dmux::cout << "Unable to save middle results" << endl;
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

string DensitySizeTest(vector<String>& ccl_algorithms, const path& input_path, const string& inputFolder, const string& input_txt, const string& gnuplot_script_extension, path& output_path, string& latex_folder, string& colors_folder, const bool& saveMiddleResults, const unsigned& n_test, const string& middle_folder, const bool& write_n_labels = true, const bool& outputColors = true)
{
    string outputFolder = inputFolder,
        //completeOutputPath = output_path + kPathSeparator + outputFolder,
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
        //middleOutFolder = completeOutputPath + kPathSeparator + middle_folder,
        //outColorFolder = output_path + kPathSeparator + outputFolder + kPathSeparator + colors_folder,
        outputNull = inputFolder + "_NULL_results.txt";

    path completeOutputPath = output_path / path(outputFolder),
        middleOutFolder = completeOutputPath / path(middle_folder),
        outColorFolder = output_path / path(outputFolder) / path(colors_folder);

    // Creation of output path
    //if (!dirExists(completeOutputPath.c_str()))
    //    if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
    //        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath);
    if (!create_directories(completeOutputPath)) {
        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + completeOutputPath.string());
    }

    if (outputColors) {
        // Creation of color output path
        //if (!dirExists(outColorFolder.c_str()))
        //    if (0 != std::system(("mkdir " + outColorFolder).c_str()))
        //        return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder);
        if (!create_directories(outColorFolder)) {
            return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + outColorFolder.string());
        }
    }

    if (saveMiddleResults) {
        /*if (!dirExists(middleOutFolder.c_str()))
            if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
                return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder);*/
        if (!create_directories(middleOutFolder)) {
            return ("Density_Size_Test on '" + inputFolder + "': Unable to find/create the output path " + middleOutFolder.string());
        }
    }

    //string isPath = input_path + kPathSeparator + inputFolder + kPathSeparator + input_txt,
    //	osPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputBroadResult,
    //	densityOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputDensityResult,
    //	densityNormalizedOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputDensityNormalizedResult,
    //	sizeOsPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputSizeResult,
    //	//size_normalized_os_path = output_path + kPathSeparator + outputFolder + kPathSeparator + output_size_normalized_result,
    //	NullPath = output_path + kPathSeparator + outputFolder + kPathSeparator + outputNull;

    path isPath = input_path / path(inputFolder) / path(input_txt),
        osPath = output_path / path(outputFolder) / path(outputBroadResult),
        densityOsPath = output_path / path(outputFolder) / path(outputDensityResult),
        densityNormalizedOsPath = output_path / path(outputFolder) / path(outputDensityNormalizedResult),
        sizeOsPath = output_path / path(outputFolder) / path(outputSizeResult),
        //size_normalized_os_path = output_path / path(outputFolder) / path(output_size_normalized_result,
        NullPath = output_path / path(outputFolder) / path(outputNull),
        latex_charts_path = output_path / path(latex_folder);

    // For DENSITY RESULT
    ofstream densityOs(densityOsPath.string());
    if (!densityOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + densityOsPath.string());
    // For DENSITY NORMALIZED RESULT
    ofstream densityNormalizedOs(densityNormalizedOsPath.string());
    if (!densityNormalizedOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + densityNormalizedOsPath.string());
    // For SIZE RESULT
    ofstream sizeOs(sizeOsPath.string());
    if (!sizeOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + sizeOsPath.string());
    // For LIST OF INPUT IMAGES
    ifstream is(isPath.string());
    if (!is.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to open " + isPath.string());
    // For labeling_NULL LABELING RESULTS
    ofstream NullOs(NullPath.string());
    if (!NullOs.is_open())
        return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + NullPath.string());

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
            if (currentNumber * 100 / file_number != (currentNumber - 1) * 100 / file_number) {
                dmux::cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
                fflush(stdout);
            }
            //p_bar.Display(currentNumber++, test + 1);

            Mat1b binaryImg;

            if (!GetBinaryImage(input_path / path(inputFolder) / path(filename), Labeling::img_)) {
                if (filenames[file].second)
                    dmux::cout << "'" + filename + "' does not exist" << endl;
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
                algorithm->PerformLabeling();
                algorithm->perf_.stop();
                n_labels = algorithm->n_labels_;

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

                    path colored_file_path = outColorFolder / path(filename + "_" + algo_name + ".png");
                    imwrite(colored_file_path.string(), imgColors);
                }
                ++i;
            }// END ALGORTIHMS FOR
        } // END FILES FOR
          // To display "progress bar"
        dmux::cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
        fflush(stdout);

        // Save middle results if necessary (flag 'average_save_middle_tests' enable)
        if (saveMiddleResults) {
            //string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
            path middleOut = middleOutFolder / path(middleFile + "_" + to_string(test) + ".txt");
            saveBroadOutputResults(current_res, middleOut, ccl_algorithms, write_n_labels, labels, filenames);
        }
    }// END TEST FOR

    //p_bar.End(n_test);

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

    SystemInfo s_info;
    string compiler_name(s_info.compiler_name());
    string compiler_version(s_info.compiler_version());
    //replace the . with _ for compiler strings
    std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

    //string scriptos_path = output_path + kPathSeparator + outputFolder + kPathSeparator + gnuplotScript;
    {
        path scriptos_path = output_path / path(outputFolder) / path(gnuplotScript);
        ofstream scriptOs(scriptos_path.string());
        if (!scriptOs.is_open())
            return ("Density_Size_Test on '" + inputFolder + "': Unable to create " + scriptos_path.string());

        scriptOs << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
        scriptOs << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplotScript << "' if you want to run it" << endl << endl;

        scriptOs << "reset" << endl;
        scriptOs << "cd '" << completeOutputPath.string() << "\'" << endl;
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

        scriptOs << "set output \'" << (latex_charts_path / path(compiler_name + compiler_version + outputDensityGraph)).string() << "\'" << endl;
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

        scriptOs << "set output \'" << (latex_charts_path / path(compiler_name + compiler_version + outputSizeGraph)).string() << "\'" << endl;
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
    }
    if (0 != std::system(("gnuplot \"" + (completeOutputPath / path(gnuplotScript)).string() + "\"").c_str()))
        return ("Density_Size_Test on '" + inputFolder + "': Unable to run gnuplot's script");
    //return ("Density_Size_Test on '" + outputFolder + "': successfully done");
    return ("");
}

int main()
{
    // Redirect cv exceptions
    cvRedirectError(RedirectCvError);

    // Hide cursor from console
    HideConsoleCursor();

    // To handle filesystem errors
    error_code ec;

    // Create StreamDemultiplexer object in order
    // to print output on both stdout and log file 
    string logfile = "log.txt";
    ofstream os(logfile);
    if (os.is_open()) {
        dmux::cout.AddStream(os);
    }

    OutputBox ob_setconf("Setting Configuration Parameters"); 
    // Read yaml configuration file
    const string config_file = "config.yaml";
    FileStorage fs;
    try {
        fs.open(config_file, FileStorage::READ);
    }
    catch (const cv::Exception&) {
        exit(EXIT_FAILURE);  // Error redirected, 
                             // OpenCV redirected function will 
                             // print the error on stdout
    }

    if (!fs.isOpened()) {
        ob_setconf.Cerror("Failed to open '" + config_file + "'");
        // EXIT_FAILURE
    }

    // Load configuration data from yaml
    ConfigData cfg(fs);

    // Release FileStorage
    fs.release();

    /*************************************************************************/
    /*  Configuration parameters check                                       */
    /*************************************************************************/

    // Check if all the specified algorithms exist
    for (auto& algo_name : cfg.ccl_algorithms) {
        if (!LabelingMapSingleton::Exists(algo_name)) {
            ob_setconf.Cwarning("Unable to find the algorithm '" + algo_name + "'");
        }
        else {
            cfg.ccl_existing_algorithms.push_back(algo_name);
        }
    }

    if (cfg.ccl_existing_algorithms.size() == 0) {
        ob_setconf.Cerror("There are no valid values in the 'algorithms' list");
    }

    // Check if labeling methods of the specified algorithms exist
    Labeling::img_ = Mat1b(1, 1, static_cast<uchar>(0));
    for (const auto& algo_name : cfg.ccl_existing_algorithms) {
        const auto& algorithm = LabelingMapSingleton::GetLabeling(algo_name);
        if (cfg.perform_average || (cfg.perform_correctness && cfg.perform_check_8connectivity_std)) {
            try {
                algorithm->PerformLabeling();
                cfg.ccl_average_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
        if (cfg.perform_average_ws || (cfg.perform_correctness && cfg.perform_check_8connectivity_ws)) {
            try {
                algorithm->PerformLabelingWithSteps();
                cfg.ccl_average_ws_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
        if (cfg.perform_memory || (cfg.perform_correctness && cfg.perform_check_8connectivity_mem)) {
            try {
                algorithm->PerformLabelingMem(vector<unsigned long int>{});
                cfg.ccl_mem_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
    }

    if ((cfg.perform_average || (cfg.perform_correctness && cfg.perform_check_8connectivity_std)) && cfg.ccl_average_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabeling()' method, related tests will be skipped");
        cfg.perform_average = false; 
        cfg.perform_check_8connectivity_std = false;
    }

    if ((cfg.perform_average_ws || (cfg.perform_correctness && cfg.perform_check_8connectivity_ws)) && cfg.ccl_average_ws_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
        cfg.perform_average_ws = false;
        cfg.perform_check_8connectivity_ws = false;
    }

    if ((cfg.perform_memory || (cfg.perform_correctness && cfg.perform_check_8connectivity_mem)) && cfg.ccl_mem_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingMem()' method, related tests will be skipped");
        cfg.perform_memory = false;
        cfg.perform_check_8connectivity_mem = false;
    }

    if (cfg.perform_average && (cfg.average_tests_number < 1 || cfg.average_tests_number > 999)) {
        ob_setconf.Cwarning("'Average tests' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_average = false;
    }

    if (cfg.perform_density && (cfg.density_tests_number < 1 || cfg.density_tests_number > 999)) {
        ob_setconf.Cwarning("'Density tests' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_density = false;
    }

    if (cfg.perform_average_ws && (cfg.average_ws_tests_number < 1 || cfg.average_ws_tests_number > 999)) {
        ob_setconf.Cwarning("'Average tests with steps' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_average_ws = false;
    }

    if ((cfg.perform_correctness) && cfg.check_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'correctness tests', skipped");
        cfg.perform_correctness = false;
    }

    if ((cfg.perform_average) && cfg.average_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'average tests', skipped");
        cfg.perform_average = false;
    }

    if ((cfg.perform_memory) && cfg.memory_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'memory tests', skipped");
        cfg.perform_memory = false;
    }

    if (!cfg.perform_average && !cfg.perform_correctness && 
        !cfg.perform_density && !cfg.perform_memory && 
        !cfg.perform_average_ws) {
        ob_setconf.Cerror("There are no tests to perform");
    }

    // Check datasets
    vector<String> ds(cfg.memory_datasets);
    ds.insert(ds.end(), cfg.average_datasets.begin(), cfg.average_datasets.end());
    ds.insert(ds.end(), cfg.check_datasets.begin(), cfg.check_datasets.end());
    std::sort(ds.begin(), ds.end());
    ds.erase(unique(ds.begin(), ds.end()), ds.end());

    // Check if all the datasets files.txt exist
    for (auto& x : ds) {
        path p = cfg.input_path / path(x) / path(cfg.input_txt);
        if (!exists(p, ec)) {
            ob_setconf.Cwarning("There is no dataset (no files.txt available) " + p.string() + ", skipped");
        }
    }

    // Set and create current output directory
    if (!create_directories(cfg.output_path, ec)) {
        ob_setconf.Cerror("Unable to create output directory '" + cfg.output_path.string() + "' - " + ec.message());
    }

    // Create the directory for latex reports
    if (!create_directories(cfg.latex_path, ec)) {
        ob_setconf.Cerror("Unable to create output directory '" + cfg.latex_path.string() + "' - " + ec.message());
    }

    ob_setconf.Cmessage("Setting Configuration Parameters DONE"); 
    ob_setconf.CloseBox();

    YacclabTests yt(cfg);

    // Correctness test
    if (cfg.perform_correctness) {
        if (cfg.perform_check_8connectivity_std) {
            yt.CheckPerformLabeling();
        }

        if (cfg.perform_check_8connectivity_ws) {
            yt.CheckPerformLabelingWithSteps();
        }
        
        if (cfg.perform_check_8connectivity_mem) {
            yt.CheckPerformLabelingMem();
        }
    }

    // Average tests
    if (cfg.perform_average) {
        yt.AverageTest();
    }

    // Average with steps tests
    if (cfg.perform_average_ws) {
        yt.AverageTestWithSteps();
    }

    // Density tests


    // Granularity tests


    // Memory tests
    if (cfg.perform_memory) {
        yt.MemoryTest();
    }

    // DENSITY_SIZE_TESTS
    if (cfg.perform_density) {
        //dmux::cout << endl << "DENSITY_SIZE TESTS: " << endl;
        //TitleBar::Display("DENSITY_SIZE TESTS");
        if (cfg.ccl_existing_algorithms.size() == 0) {
            dmux::cout << "ERROR: no algorithms, density_size tests skipped" << endl;
        }
        else {
            for (unsigned i = 0; i < cfg.density_datasets.size(); ++i) {
                dmux::cout << "Density_Size_Test on '" << cfg.density_datasets[i] << "': starts" << endl;
                dmux::cout << DensitySizeTest(cfg.ccl_existing_algorithms, cfg.input_path, cfg.density_datasets[i], cfg.input_txt, cfg.gnuplot_script_extension, cfg.output_path, cfg.latex_path.stem().string(), cfg.colors_folder, cfg.density_save_middle_tests, cfg.density_tests_number, cfg.middle_folder, cfg.write_n_labels, cfg.density_color_labels) << endl;
                //dmux::cout << "Density_Size_Test on '" << density_datasets[i] << "': ends" << endl << endl;
            }
        }
    }

    yt.LatexGenerator();

    // Copy log file into output folder
    dmux::cout.flush();
    copy(path(logfile), cfg.output_path / path(logfile), ec);
 
    return EXIT_SUCCESS;
}