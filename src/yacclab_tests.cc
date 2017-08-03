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

#include "yacclab_tests.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <cstdint>

#include <opencv2/imgproc.hpp>

#include "labeling_algorithms.h"
#include "latex_generator.h"
#include "progress_bar.h"
#include "utilities.h"

using namespace std;
using namespace cv;

// Load a list of image names from a specified file (files_path) and store them into a vector of
// pairs (filenames). Each pairs contains the name of the file (first) and a bool (second)
// representing file state.
bool YacclabTests::LoadFileList(vector<pair<string, bool>>& filenames, const path& files_path)
{
    // Open files_path (files.txt)
    ifstream is(files_path.string());
    if (!is.is_open()) {
        return false;
    }

    string cur_filename;
    while (getline(is, cur_filename)) {
        // To delete possible carriage return in the file name
        // (especially designed for windows file newline format)
        RemoveCharacter(cur_filename, '\r');
        filenames.push_back(make_pair(cur_filename, true));
    }

    is.close();
    return true;
}

// This function take a Mat1d of results and save it in the  specified output-stream
void YacclabTests::SaveBroadOutputResults(map<String, Mat1d>& results, const string& o_filename, const Mat1i& labels, const vector<pair<string, bool>>& filenames)
{
    ofstream os(o_filename);
    if (!os.is_open()) {
        cerror("Unable to save middle results in '" + o_filename + "'");
        return;
    }

    // To set heading file format
    os << "#" << "\t";
    for (const auto& algo_name : cfg_.ccl_algorithms) {
        const auto& algo = LabelingMapSingleton::GetLabeling(algo_name);

        // Calculate the max of the columns to find unused steps
        Mat1d results_reduced(1, results.at(algo_name).cols);
        cv::reduce(results.at(algo_name), results_reduced, 0, CV_REDUCE_MAX);

        for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
            StepType step = static_cast<StepType>(step_number);
            double column_value(results_reduced(0, step_number));
            if (column_value != numeric_limits<double>::max()) {
                os << algo_name + "_" << Step(step) << "\t";
            }
        }
        cfg_.write_n_labels ? os << algo_name + "_n_labels" << "\t" : os << "";
    }
    os << endl;

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algo_name : cfg_.ccl_algorithms) {
                const auto& algo = LabelingMapSingleton::GetLabeling(algo_name);

                for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
                    if (results.at(algo_name)(files, step_number) != numeric_limits<double>::max()) {
                        os << results.at(algo_name)(files, step_number) << "\t";
                    }
                    else {
                        // Step not held, skipped
                        //os << 0 << "\t";
                    }
                }
                cfg_.write_n_labels ? os << labels(files, i) << "\t" : os << "";
                ++i;
            }
            os << endl;
        }
    }
}

void YacclabTests::SaveBroadOutputResults(const Mat1d& results, const string& o_filename, const Mat1i& labels, const vector<pair<string, bool>>& filenames)
{
    ofstream os(o_filename);
    if (!os.is_open()) {
        dmux::cout << "Unable to save middle results" << endl;
        return;
    }

    // To set heading file format
    os << "#";
    for (const auto& algo_name : cfg_.ccl_algorithms) {
        os << "\t" << algo_name;
        cfg_.write_n_labels ? os << "\t" << "n_label" : os << "";
    }
    os << endl;
    // To set heading file format

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algo_name : cfg_.ccl_algorithms) {
                os << results(files, i) << "\t";
                cfg_.write_n_labels ? os << labels(files, i) << "\t" : os << "";
                ++i;
            }
            os << endl;
        }
    }
}



void YacclabTests::AverageTest()
{
    // Initialize output message box
    OutputBox ob("Average Tests");

    string complete_results_suffix = "_results.txt",
        middle_results_suffix = "_run",
        average_results_suffix = "_average.txt";

    // Initialize results container
    average_results_ = cv::Mat1d(cfg_.average_datasets.size(), cfg_.ccl_average_algorithms.size(), std::numeric_limits<double>::max());

    for (unsigned d = 0; d < cfg_.average_datasets.size(); ++d) { // For every dataset in the average list
        String dataset_name(cfg_.average_datasets[d]),
            output_average_results = dataset_name + average_results_suffix,
            output_graph = dataset_name + kTerminalExtension,
            output_graph_bw = dataset_name + "_bw" + kTerminalExtension;

        path dataset_path(cfg_.input_path / path(dataset_name)),
            is_path = dataset_path / path(cfg_.input_txt), // files.txt path
            current_output_path(cfg_.output_path / path(cfg_.average_folder) / path(dataset_name)),
            output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
            output_colored_images_path = current_output_path / path(cfg_.colors_folder),
            output_middle_results_path = current_output_path / path(cfg_.middle_folder),
            average_os_path = current_output_path / path(output_average_results);

        if (!create_directories(current_output_path)) {
            cerror("Averages Test on '" + dataset_name + "': Unable to find/create the output path " + current_output_path.string());
        }

        if (cfg_.average_color_labels) {
            if (!create_directories(output_colored_images_path)) {
                cerror("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + output_colored_images_path.string());
            }
        }

        if (cfg_.average_ws_save_middle_tests) {
            if (!create_directories(output_middle_results_path)) {
                cerror("Averages Test on '" + dataset_name + "': Unable to find/create the output path " + output_middle_results_path.string());
            }
        }

        // For AVERAGES
        ofstream average_os(average_os_path.string());
        if (!average_os.is_open()) {
            cerror("Averages Test on '" + dataset_name + "': Unable to open " + average_os_path.string());
        }

        // To save list of filename on which CLLAlgorithms must be tested
        vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
        if (!LoadFileList(filenames, is_path)) {
            ob.Cerror("Unable to open '" + is_path.string() + "'", dataset_name);
            continue;
        }

        // Number of files
        int filenames_size = filenames.size();

        // To save middle/min and average results;
        Mat1d min_res(filenames_size, cfg_.ccl_average_algorithms.size(), numeric_limits<double>::max());
        Mat1d current_res(filenames_size, cfg_.ccl_average_algorithms.size(), numeric_limits<double>::max());
        Mat1i labels(filenames_size, cfg_.ccl_average_algorithms.size(), 0);
        vector<pair<double, uint16_t>> supp_average(cfg_.ccl_average_algorithms.size(), make_pair(0.0, 0));

        // Start output message box
        ob.StartRepeatedBox(dataset_name, filenames_size, cfg_.average_tests_number);

        // Test is executed n_test times
        for (unsigned test = 0; test < cfg_.average_tests_number; ++test) {
            // For every file in list
            for (unsigned file = 0; file < filenames.size(); ++file) {
                // Display output message box
                ob.UpdateRepeatedBox(file);

                string filename = filenames[file].first;
                path filename_path = dataset_path / path(filename);

                // Read and load image
                if (!GetBinaryImage(filename_path, Labeling::img_)) {
                    ob.Cmessage("Unable to open '" + filename + "'");
                    continue;
                }

                unsigned i = 0;
                // For all the Algorithms in the array
                for (const auto& algo_name : cfg_.ccl_average_algorithms) {
                    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
                    unsigned n_labels;

                    try {
                        // Perform current algorithm on current image and save result.
                        algorithm->perf_.start();
                        algorithm->PerformLabeling();
                        algorithm->perf_.stop();

                        // This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                        n_labels = algorithm->n_labels_;
                    }
                    catch (const runtime_error&) {
                        ob.Cmessage("'PerformLabeling()' method not implemented in '" + algo_name + "'");
                        continue;
                    }

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
                    if (cfg_.average_color_labels && test == 0) {
                        // Remove gnuplot escape character from output filename
                        Mat3b imgColors;

                        NormalizeLabels(algorithm->img_labels_);
                        ColorLabels(algorithm->img_labels_, imgColors);
                        String colored_image = (output_colored_images_path / path(filename + "_" + algo_name + ".png")).string();
                        imwrite(colored_image, imgColors);
                    }
                    ++i;
                } // END ALGORITHMS FOR
            } // END FILES FOR.
            ob.StopRepeatedBox();

            // Save middle results if necessary (flag 'average_save_middle_tests' enabled)
            if (cfg_.average_save_middle_tests) {
                string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + to_string(test) + ".txt")).string();
                SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames);
            }
        } // END TEST FOR

        // To write in a file min results
        SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames);

        // To calculate average times and write it on the specified file
        for (int r = 0; r < min_res.rows; ++r) {
            for (int c = 0; c < min_res.cols; ++c) {
                if (min_res(r, c) != numeric_limits<double>::max()) {
                    supp_average[c].first += min_res(r, c);
                    supp_average[c].second++;
                }
            }
        }

        average_os << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << endl;
        for (unsigned i = 0; i < cfg_.ccl_average_algorithms.size(); ++i) {
            // For all the Algorithms in the array
            const auto& algo_name = cfg_.ccl_average_algorithms[i];

            // Gnuplot requires double-escaped name in presence of underscores
            {
                string algo_name_double_escaped = algo_name;
                std::size_t found = algo_name_double_escaped.find_first_of("_");
                while (found != std::string::npos) {
                    algo_name_double_escaped.insert(found, "\\\\");
                    found = algo_name_double_escaped.find_first_of("_", found + 3);
                }
                average_os << algo_name_double_escaped << "\t";
            }

            // Save all the results
            average_results_(d, i) = supp_average[i].first / supp_average[i].second;
            average_os << std::fixed << std::setprecision(8) << supp_average[i].first / supp_average[i].second << "\t";
            // TODO numberOfDecimalDigitToDisplayInGraph in config?
            average_os << std::fixed << std::setprecision(2) << supp_average[i].first / supp_average[i].second << endl;
        }

        { // GNUPLOT SCRIPT
            SystemInfo s_info;
            string compiler_name(s_info.compiler_name());
            string compiler_version(s_info.compiler_version());
            //replace the . with _ for compiler strings
            std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

            path script_os_path = current_output_path / path(dataset_name + cfg_.gnuplot_script_extension);

            ofstream script_os(script_os_path.string());
            if (!script_os.is_open()) {
                cerror("Averages Test With Steps on '" + dataset_name + "': Unable to create " + script_os_path.string());
            }

            script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
            script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + cfg_.gnuplot_script_extension << "' if you want to run it" << endl << endl;

            script_os << "reset" << endl;
            script_os << "cd '" << current_output_path.string() << "\'" << endl;
            script_os << "set grid ytic" << endl;
            script_os << "set grid" << endl << endl;

            script_os << "# " << dataset_name << "(COLORS)" << endl;
            script_os << "set output \"" + output_graph + "\"" << endl;

            script_os << "set title " << GetGnuplotTitle() << endl << endl;

            script_os << "# " << kTerminal << " colors" << endl;
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

            script_os << "# Graph style" << endl;
            script_os << "set style data histogram" << endl;
            script_os << "set style histogram cluster gap 1" << endl;
            script_os << "set style fill solid 0.25 border -1" << endl;
            script_os << "set boxwidth 0.9" << endl << endl;

            script_os << "# Get stats to set labels" << endl;
            script_os << "stats \"" << output_average_results << "\" using 2 nooutput" << endl;
            script_os << "ymax = STATS_max + (STATS_max/100)*10" << endl;
            script_os << "xw = 0" << endl;
            script_os << "yw = (ymax)/22.0" << endl << endl;

            script_os << "# Axes labels" << endl;
            script_os << "set xtic rotate by -45 scale 0" << endl;
            script_os << "set ylabel \"Execution Time [ms]\"" << endl << endl;

            script_os << "# Axes range" << endl;
            script_os << "set yrange[0:ymax]" << endl;
            script_os << "set xrange[*:*]" << endl << endl;

            script_os << "# Legend" << endl;
            script_os << "set key off" << endl << endl;

            script_os << "# Plot" << endl;
            script_os << "plot \\" << endl;

            script_os << "'" + output_average_results + "' using 2:xtic(1), '" << output_average_results << "' using ($0 - xw) : ($2 + yw) : (stringcolumn(3)) with labels" << endl << endl;

            script_os << "# Replot in latex folder" << endl;
            script_os << "set title \"\"" << endl << endl;
            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + "_" + output_graph)).string() << "\'" << endl;
            script_os << "replot" << endl << endl;

            script_os << "# " << dataset_name << "(BLACK AND WHITE)" << endl;
            script_os << "set output \"" + output_graph_bw + "\"" << endl;

            script_os << "set title " << GetGnuplotTitle() << endl << endl;

            script_os << "# " << kTerminal << " black and white" << endl;
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

            script_os << "replot" << endl << endl;

            script_os << "exit gnuplot" << endl;

            average_os.close();
            script_os.close();
        } // GNUPLOT SCRIPT

        if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + cfg_.gnuplot_script_extension)).string() + "\"").c_str())) {
            cerror("Averages Test on '" + dataset_name + "': Unable to run gnuplot's script");
        }
    } // END DATASET FOR
}

void YacclabTests::AverageTestWithSteps()
{
    // Initialize output message box
    OutputBox ob("Average Tests With Steps");

    string complete_results_suffix = "_results.txt",
        middle_results_suffix = "_run",
        average_results_suffix = "_average.txt";

    for (unsigned d = 0; d < cfg_.average_datasets_ws.size(); ++d) { // For every dataset in the average list
        String dataset_name(cfg_.average_datasets_ws[d]),
            output_average_results = dataset_name + average_results_suffix,
            output_graph = dataset_name + kTerminalExtension,
            output_graph_bw = dataset_name + "_bw" + kTerminalExtension;

        path dataset_path(cfg_.input_path / path(dataset_name)),
            is_path = dataset_path / path(cfg_.input_txt), // files.txt path
            current_output_path(cfg_.output_path / path(cfg_.average_ws_folder) / path(dataset_name)),
            output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
            output_middle_results_path = current_output_path / path(cfg_.middle_folder),
            average_os_path = current_output_path / path(output_average_results);

        if (!create_directories(current_output_path)) {
            cerror("Averages Test With Steps on '" + dataset_name + "': Unable to find/create the output path " + current_output_path.string());
        }

        if (cfg_.average_ws_save_middle_tests) {
            if (!create_directories(output_middle_results_path)) {
                cerror("Averages Test With Steps on '" + dataset_name + "': Unable to find/create the output path " + output_middle_results_path.string());
            }
        }

        // For AVERAGES
        ofstream average_os(average_os_path.string());
        if (!average_os.is_open()) {
            cerror("Averages Test With Steps on '" + dataset_name + "': Unable to open " + average_os_path.string());
        }

        // Initialize results container
        average_ws_results_[dataset_name] = Mat1d(cfg_.ccl_average_ws_algorithms.size(), StepType::ST_SIZE, numeric_limits<double>::max());

        // To save list of filename on which CLLAlgorithms must be tested
        vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
        if (!LoadFileList(filenames, is_path)) {
            ob.Cerror("Unable to open '" + is_path.string() + "'", dataset_name);
            continue;
        }

        // Number of files
        int filenames_size = filenames.size();

        // To save middle/min and average results;
        map<String, Mat1d> current_res;
        map<String, Mat1d> min_res;
        Mat1i labels(filenames_size, cfg_.ccl_average_ws_algorithms.size(), 0);

        // If true the i-th step is used by all the algorithms
        vector<bool> steps_presence(StepType::ST_SIZE, false);

        for (const auto& algo_name : cfg_.ccl_average_ws_algorithms) {
            current_res[algo_name] = Mat1d(filenames_size, StepType::ST_SIZE, numeric_limits<double>::max());
            min_res[algo_name] = Mat1d(filenames_size, StepType::ST_SIZE, numeric_limits<double>::max());
        }

        // Start output message box
        ob.StartRepeatedBox(dataset_name, filenames_size, cfg_.average_ws_tests_number);

        // Test is executed n_test times
        for (unsigned test = 0; test < cfg_.average_ws_tests_number; ++test) {
            // For every file in list
            for (unsigned file = 0; file < filenames.size(); ++file) {
                // Display output message box
                ob.UpdateRepeatedBox(file);

                string filename = filenames[file].first;
                path filename_path = dataset_path / path(filename);

                // Read and load image
                if (!GetBinaryImage(filename_path, Labeling::img_)) {
                    ob.Cmessage("Unable to open '" + filename + "'");
                    continue;
                }

                unsigned i = 0;
                // For all the Algorithms in the array
                for (const auto& algo_name : cfg_.ccl_average_ws_algorithms) {
                    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
                    unsigned n_labels;
                    try {
                        // Perform current algorithm on current image and save result.
                        algorithm->PerformLabelingWithSteps();

                        // This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
                        n_labels = algorithm->n_labels_;
                    }
                    catch (const runtime_error&) {
                        ob.Cmessage("'PerformLabelingWithSteps()' method not implemented in '" + algo_name + "'");
                        continue;
                    }

                    // Save number of labels (we reasonably supposed that labels's number is the same on every #test so only the first time we save it)
                    if (test == 0) {
                        labels(file, i) = n_labels;
                    }

                    // Save time results of all the steps
                    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
                        string step = Step(static_cast<StepType>(step_number));

                        // Find if the current algorithm has the current step
                        if (algorithm->perf_.find(step)) {
                            current_res[algo_name](file, step_number) = algorithm->perf_.get(step);
                            if (algorithm->perf_.get(step) < min_res[algo_name](file, step_number)) {
                                min_res[algo_name](file, step_number) = algorithm->perf_.get(step);
                            }
                        }
                    }
                    ++i;
                } // END ALGORITHMS FOR
            } // END FILES FOR.
            ob.StopRepeatedBox();

            // Save middle results if necessary (flag 'average_save_middle_tests' enabled)
            if (cfg_.average_ws_save_middle_tests) {
                string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + to_string(test) + ".txt")).string();
                SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames);
            }
        }// END TESTS FOR

        // To write in a file min results
        SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames);

        // Write heading string in output stream
        average_os << "#Algorithm" << "\t";
        for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
            StepType step = static_cast<StepType>(step_number);
            average_os << Step(step) << "\t";
        }
        average_os << "Total" << endl;

        double max_value(0.0);
        // To calculate average times and write it on the specified file
        for (unsigned a = 0; a < cfg_.ccl_average_ws_algorithms.size(); ++a) {
            const auto& algo_name(cfg_.ccl_average_ws_algorithms[a]);

            vector<pair<double, uint16_t>> supp_average(StepType::ST_SIZE, make_pair(0.0, 0));

            for (int r = 0; r < min_res.at(algo_name).rows; ++r) {
                for (int c = 0; c < min_res.at(algo_name).cols; ++c) {
                    if (min_res.at(algo_name)(r, c) != numeric_limits<double>::max()) {
                        supp_average[c].first += min_res.at(algo_name)(r, c);
                        supp_average[c].second++;
                    }
                }
            }

            // Gnuplot requires double-escaped name in presence of underscores
            {
                string algo_name_double_escaped = algo_name;
                std::size_t found = algo_name_double_escaped.find_first_of("_");
                while (found != std::string::npos) {
                    algo_name_double_escaped.insert(found, "\\\\");
                    found = algo_name_double_escaped.find_first_of("_", found + 3);
                }
                average_os << algo_name_double_escaped << "\t";
            }
            double cu_sum{ 0.0 };

            // Matrix reduce done, save the results into the average file
            for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
                StepType step = static_cast<StepType>(step_number);
                double avg{ 0.0 };
                if (supp_average[step_number].first > 0.0 && supp_average[step_number].second > 0) {
                    steps_presence[step_number] = true;
                    avg = supp_average[step_number].first / supp_average[step_number].second;
                    cu_sum += avg;
                }
                else {
                    // The current step is not threated by the current algorithm, write 0
                }
                average_ws_results_[dataset_name](a, step_number) = avg;
                average_os << std::fixed << std::setprecision(6) << avg << "\t";
            }

            // Keep in memory the max cumulative time measured
            if (cu_sum > max_value) {
                max_value = cu_sum;
            }

            // Write the total time at the end of the line
            average_os << cu_sum << endl;
            supp_average.clear();
        }

        // GNUPLOT SCRIPT
        {
            SystemInfo s_info;
            string compiler_name(s_info.compiler_name());
            string compiler_version(s_info.compiler_version());
            //replace the . with _ for compiler strings
            std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

            path script_os_path = current_output_path / path(dataset_name + cfg_.gnuplot_script_extension);

            ofstream script_os(script_os_path.string());
            if (!script_os.is_open()) {
                cerror("Averages Test With Steps on '" + dataset_name + "': Unable to create " + script_os_path.string());
            }

            script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
            script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + cfg_.gnuplot_script_extension << "' if you want to run it" << endl << endl;

            script_os << "reset" << endl;
            script_os << "cd '" << current_output_path.string() << "\'" << endl;
            script_os << "set grid ytic" << endl;
            script_os << "set grid" << endl << endl;

            script_os << "# " << dataset_name << "(COLORS)" << endl;
            script_os << "set output \"" + output_graph + "\"" << endl;

            script_os << "set title " << GetGnuplotTitle() << endl << endl;

            script_os << "# " << kTerminal << " colors" << endl;
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;

            script_os << "# Graph style" << endl;
            script_os << "set style data histogram" << endl;
            script_os << "set style histogram cluster gap 1" << endl;
            script_os << "set style histogram rowstacked" << endl;
            script_os << "set style fill solid 0.25 border -1" << endl;
            script_os << "set boxwidth 0.5" << endl << endl;

            script_os << "# Get stats to set labels" << endl;
            script_os << "stats \"" << output_average_results << "\" using 4 nooutput" << endl;
            //script_os << "ymax = STATS_max + (STATS_max/100)*10" << endl;
            script_os << "ymax = " << max_value << "+" << max_value << "/10.0" << endl;
            script_os << "xw = 0" << endl;
            script_os << "yw = (ymax)/22.0" << endl << endl;

            script_os << "# Axes labels" << endl;
            script_os << "set xtic rotate by -45 scale 0" << endl;
            script_os << "set ylabel \"Execution Time [ms]\"" << endl << endl;

            script_os << "# Axes range" << endl;
            script_os << "set yrange[0:ymax]" << endl;
            script_os << "set xrange[*:*]" << endl << endl;

            script_os << "# Legend" << endl;
            script_os << "set key left top nobox font ', 8'" << endl << endl;

            script_os << "# Plot" << endl;
            script_os << "plot \\" << endl;

            script_os << "'" + output_average_results + "' using 2:xtic(1) title '" << Step(static_cast<StepType>(0)) << "', \\" << endl;
            unsigned i = 3;
            // Start from the second step
            for (int step_number = 1; step_number != StepType::ST_SIZE; ++step_number, ++i) {
                StepType step = static_cast<StepType>(step_number);
                // Add the step only if it used by almost an algorithm
                if (steps_presence[step_number]) {
                    script_os << "'' u " << i << " title '" << Step(step) << "', \\" << endl;
                }
            }
            const unsigned start_n = 2;
            i = 2;
            for (; i <= StepType::ST_SIZE + 1; ++i) {
                script_os << "'' u ($0) : ((";
                for (unsigned j = i; j >= start_n; --j) {
                    script_os << "$" << j;
                    if (j > start_n) {
                        script_os << "+";
                    }
                }
                script_os << ") - ($" << i << "/2)) : ($" << i << "!=0.0 ? sprintf(\"%6.3f\",$" << i << "):'') w labels font 'Tahoma, 11' title '', \\" << endl;
            }
            script_os << "'' u ($0) : ($" << i << " + yw) : ($" << i << "!=0.0 ? sprintf(\"%6.3f\",$" << i << "):'') w labels font 'Tahoma' title '', \\" << endl;

            script_os << "# Replot in latex folder" << endl;
            script_os << "set title \"\"" << endl << endl;
            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + "_with_steps_" + output_graph)).string() << "\'" << endl;
            script_os << "replot" << endl << endl;

            script_os << "# " << dataset_name << "(BLACK AND WHITE)" << endl;
            script_os << "set output \"" + output_graph_bw + "\"" << endl;

            script_os << "set title " << GetGnuplotTitle() << endl << endl;

            script_os << "# " << kTerminal << " black and white" << endl;
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;

            script_os << "replot" << endl << endl;

            script_os << "exit gnuplot" << endl;

            average_os.close();
            script_os.close();
            // GNUPLOT SCRIPT
        }
        if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + cfg_.gnuplot_script_extension)).string() + "\"").c_str())) {
            cerror("Averages Test With Steps on '" + dataset_name + "': Unable to run gnuplot's script");
        }
    }
}

void YacclabTests::LatexGenerator()
{
    path latex = cfg_.latex_path / path(cfg_.latex_file);
    ofstream os(latex.string());
    if (!os.is_open()) {
        dmux::cout << "Unable to open/create " + latex.string() << endl;
        return;
    }

    // fixed number of decimal values
    os << fixed;
    os << setprecision(3);

    // Document begin
    os << "%These file is generated by YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << endl << endl;
    os << "\\documentclass{article}" << endl << endl;

    os << "\\usepackage{siunitx}" << endl;
    os << "\\usepackage{graphicx}" << endl;
    os << "\\usepackage{subcaption}" << endl << endl;
    os << "\\title{{ \\huge\\bfseries YACCLAB TESTS}}" << endl;
    os << "\\author{}" << endl << endl;
    os << "\\begin{document}" << endl << endl;
    os << "\\maketitle" << endl << endl;

    // Section average results table ------------------------------------------------------------------------------------------
    if (cfg_.perform_average) {
        os << "\\section{Average Table Results}" << endl << endl;

        os << "\\begin{table}[tbh]" << endl << endl;
        os << "\t\\centering" << endl;
        os << "\t\\caption{Average Results in ms (Lower is better)}" << endl;
        os << "\t\\label{tab:table1}" << endl;
        os << "\t\\begin{tabular}{|l|";
        for (unsigned i = 0; i < cfg_.ccl_algorithms.size(); ++i)
            os << "S[table-format=2.3]|";
        os << "}" << endl;
        os << "\t\\hline" << endl;
        os << "\t";
        for (unsigned i = 0; i < cfg_.ccl_algorithms.size(); ++i) {
            //RemoveCharacter(datasets_name, '\\');
            //datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
            os << " & {" << EscapeLatexUnderscore(cfg_.ccl_algorithms[i]) << "}"; //Header
        }
        os << "\\\\" << endl;
        os << "\t\\hline" << endl;
        for (unsigned i = 0; i < cfg_.average_datasets.size(); ++i) {
            os << "\t" << cfg_.average_datasets[i];
            for (int j = 0; j < average_results_.cols; ++j) {
                os << " & ";
                if (average_results_(i, j) != numeric_limits<double>::max())
                    os << average_results_(i, j); //Data
            }
            os << "\\\\" << endl;
        }
        os << "\t\\hline" << endl;
        os << "\t\\end{tabular}" << endl << endl;
        os << "\\end{table}" << endl;
    }

    { // CHARTS SECTION ------------------------------------------------------------------------------------------
        SystemInfo s_info;
        string info_to_latex = s_info.build() + "_" + s_info.compiler_name() + s_info.compiler_version() + "_" + s_info.os();
        std::replace(info_to_latex.begin(), info_to_latex.end(), ' ', '_');
        info_to_latex = EscapeLatexUnderscore(info_to_latex);

        string chart_size{ "0.45" }, chart_width{ "1" };
        // Get information about date and time
        string datetime = GetDatetime();

        string compiler_name(s_info.compiler_name());
        string compiler_version(s_info.compiler_version());
        //replace the . with _ for compiler strings
        std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

        // SECTION AVERAGE CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_average) {
            os << "\\section{Average Charts}" << endl << endl;
            os << "\\begin{figure}[tbh]" << endl << endl;
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << endl;
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << endl;
            os << "\t\\centering" << endl;

            for (unsigned i = 0; i < cfg_.average_datasets.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << endl;
                os << "\t\t\\caption{" << cfg_.average_datasets[i] + "}" << endl;
                os << "\t\t\\centering" << endl;
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + cfg_.average_datasets[i] + ".pdf}" << endl;
                os << "\t\\end{subfigure}" << endl << endl;
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << endl << endl;
            os << "\\end{figure}" << endl << endl;
        }

        // SECTION AVERAGE WITH STEPS CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_average_ws) {
            string average_ws_suffix{ "_with_steps_" };

            os << "\\section{Average With Steps Charts}" << endl << endl;
            os << "\\begin{figure}[tbh]" << endl << endl;
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << endl;
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << endl;
            os << "\t\\centering" << endl;
            for (unsigned i = 0; i < cfg_.average_datasets_ws.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << endl;
                os << "\t\t\\caption{" << cfg_.average_datasets_ws[i] + "}" << endl;
                os << "\t\t\\centering" << endl;
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + average_ws_suffix + cfg_.average_datasets_ws[i] + ".pdf}" << endl;
                os << "\t\\end{subfigure}" << endl << endl;
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << endl << endl;
            os << "\\end{figure}" << endl << endl;
        }

        // SECTION DENSITY CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_density) {
            vector<String> density_datasets{ "density", "size" };

            os << "\\section{Density Charts}" << endl << endl;
            os << "\\begin{figure}[tbh]" << endl << endl;
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << endl;
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << endl;
            os << "\t\\centering" << endl;

            for (unsigned i = 0; i < density_datasets.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << endl;
                os << "\t\t\\caption{" << density_datasets[i] + "}" << endl;
                os << "\t\t\\centering" << endl;
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + density_datasets[i] + ".pdf}" << endl;
                os << "\t\\end{subfigure}" << endl << endl;
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << endl << endl;
            os << "\\end{figure}" << endl << endl;
        }
    } // END CHARTS SECTION

    // SECTION MEMORY RESULT TABLE ---------------------------------------------------------------------------
    if (cfg_.perform_memory) {
        os << "\\section{Memory Accesses tests}" << endl << endl;

        for (const auto& dataset : memory_accesses_) {
            const auto& dataset_name = dataset.first;
            const auto& accesses = dataset.second;

            os << "\\begin{table}[tbh]" << endl << endl;
            os << "\t\\centering" << endl;
            os << "\t\\caption{Analysis of memory accesses required by connected components computation for '" << dataset_name << "' dataset. The numbers are given in millions of accesses}" << endl;
            os << "\t\\label{tab:table1}" << endl;
            os << "\t\\begin{tabular}{|l|";
            for (int i = 0; i < accesses.cols + 1; ++i)
                os << "S[table-format=2.3]|";
            os << "}" << endl;
            os << "\t\\hline" << endl;
            os << "\t";

            // Header
            os << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
            os << "\\\\" << endl;
            os << "\t\\hline" << endl;

            for (unsigned i = 0; i < cfg_.ccl_mem_algorithms.size(); ++i) {
                // For every algorithm
                const String& alg_name = cfg_.ccl_mem_algorithms[i];
                //RemoveCharacter(alg_name, '\\');
                os << "\t{" << alg_name << "}";

                double tot = 0;

                for (int s = 0; s < accesses.cols; ++s) {
                    // For every data_ structure
                    if (accesses(i, s) != 0)
                        os << "\t& " << (accesses(i, s) / 1000000);
                    else
                        os << "\t& ";

                    tot += (accesses(i, s) / 1000000);
                }
                // Total Accesses
                os << "\t& " << tot;

                // EndLine
                os << "\t\\\\" << endl;
            }

            // EndTable
            os << "\t\\hline" << endl;
            os << "\t\\end{tabular}" << endl << endl;
            os << "\\end{table}" << endl;
        }
    }

    os << "\\end{document}";
    os.close();
}