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
#include "memory_tester.h"
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
bool YacclabTests::SaveBroadOutputResults(map<String, Mat1d>& results, const string& o_filename, const Mat1i& labels, const vector<pair<string, bool>>& filenames, const std::vector<cv::String>& ccl_algorithms)
{
    ofstream os(o_filename);
    if (!os.is_open()) {
        return false;
    }

    // To set heading file format
    os << "#" << "\t";
    for (const auto& algo_name : ccl_algorithms) {
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
    os << '\n';

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algo_name : ccl_algorithms) {
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
            os << '\n';
        }
    }

    return true;
}

bool YacclabTests::SaveBroadOutputResults(const Mat1d& results, const string& o_filename, const Mat1i& labels, const vector<pair<string, bool>>& filenames, const std::vector<cv::String>& ccl_algorithms)
{
    ofstream os(o_filename);
    if (!os.is_open()) {
        dmux::cout << "Unable to save middle results" << '\n';
        return false;
    }

    // To set heading file format
    os << "#";
    for (const auto& algo_name : ccl_algorithms) {
        os << "\t" << algo_name;
        cfg_.write_n_labels ? os << "\t" << "n_label" : os << "";
    }
    os << '\n';
    // To set heading file format

    for (unsigned files = 0; files < filenames.size(); ++files) {
        if (filenames[files].second) {
            os << filenames[files].first << "\t";
            unsigned i = 0;
            for (const auto& algo_name : ccl_algorithms) {
                os << results(files, i) << "\t";
                cfg_.write_n_labels ? os << labels(files, i) << "\t" : os << "";
                ++i;
            }
            os << '\n';
        }
    }
    return true;
}

// To calculate average times and write it on the specified file
void YacclabTests::SaveAverageWithStepsResults(const string& os_name, const String& dataset_name, bool rounded)
{
    ofstream os(os_name);
    if (!os.is_open()) {
        dmux::cout << "Unable to save average results" << '\n';
        return;
    }

    // Write heading string in output stream
    os << "#Algorithm" << "\t";
    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
        StepType step = static_cast<StepType>(step_number);
        os << Step(step) << "\t";
    }
    os << "Total" << '\n';

    const auto& results = average_ws_results_.at(dataset_name);

    for (int r = 0; r < results.rows; ++r) {
        const auto& algo_name = cfg_.ccl_average_ws_algorithms[r];
        double cumulative_sum{ 0.0 };

        // Gnuplot requires double-escaped name when underscores are encountered
        //{
        //    string algo_name_double_escaped{ algo_name };
        //    std::size_t found = algo_name_double_escaped.find_first_of("_");
        //    while (found != std::string::npos) {
        //        algo_name_double_escaped.insert(found, "\\\\");
        //        found = algo_name_double_escaped.find_first_of("_", found + 3);
        //    }
        //    os << algo_name_double_escaped << "\t";
        //}
        os << DoubleEscapeUnderscore(string(algo_name)) << '\t';

        for (int c = 0; c < results.cols; ++c) {
            if (rounded) {
                cumulative_sum += floor(results(r, c) * 100.00 + 0.5) / 100.00;
                os << std::fixed << std::setprecision(2) << results(r, c) << "\t";
            }
            else {
                cumulative_sum += results(r, c);
                os << std::fixed << std::setprecision(8) << results(r, c) << "\t";
            }
        }
        // Write cumulative_sum as total
        if (rounded) {
            os << std::fixed << std::setprecision(2) << cumulative_sum;
        }
        else {
            os << std::fixed << std::setprecision(8) << cumulative_sum;
        }
        os << '\n';
    }
    os.close();
}

void YacclabTests::AverageTest()
{
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
            ob.Cwarning("Average Test on '" + dataset_name + "': unable to find/create the output path " + current_output_path.string());
            continue;
        }

        if (cfg_.average_color_labels) {
            if (!create_directories(output_colored_images_path)) {
                ob.Cwarning("Average Test on '" + dataset_name + "': unable to find/create the output path " + output_colored_images_path.string());
            }
        }

        if (cfg_.average_ws_save_middle_tests) {
            if (!create_directories(output_middle_results_path)) {
                ob.Cwarning("Average Test on '" + dataset_name + "': unable to find/create the output path " + output_middle_results_path.string());
            }
        }

        // For AVERAGE
        ofstream average_os(average_os_path.string());
        if (!average_os.is_open()) {
            ob.Cmessage("Average Test on '" + dataset_name + "': Unable to open " + average_os_path.string());
            // TODO
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

        map<String, size_t> algo_pos;
        for (size_t i = 0; i < cfg_.ccl_average_algorithms.size(); ++i)
            algo_pos[cfg_.ccl_average_algorithms[i]] = i;
        auto shuffled_ccl_average_algorithms = cfg_.ccl_average_algorithms;

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

                random_shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms));

                // For all the Algorithms in the array
                for (const auto& algo_name : shuffled_ccl_average_algorithms) {
                    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
                    unsigned n_labels;
                    unsigned i = algo_pos[algo_name];

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
                    algorithm->FreeLabelingData();
                } // END ALGORITHMS FOR
            } // END FILES FOR.
            ob.StopRepeatedBox();

            // Save middle results if necessary (flag 'average_save_middle_tests' enabled)
            if (cfg_.average_save_middle_tests) {
                string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + to_string(test) + ".txt")).string();
                if (!SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames, cfg_.ccl_average_algorithms)) {
                    ob.Cwarning("Unable to save middle results for Average Tests");
                }
            }
        } // END TEST FOR

        // To write in a file min results
        if (!SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames, cfg_.ccl_average_algorithms)) {
            ob.Cwarning("Unable to save min results for Average Tests");
        }

        // To calculate average times and write it on the specified file
        for (int r = 0; r < min_res.rows; ++r) {
            for (int c = 0; c < min_res.cols; ++c) {
                if (min_res(r, c) != numeric_limits<double>::max()) {
                    supp_average[c].first += min_res(r, c);
                    supp_average[c].second++;
                }
            }
        }

        average_os << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << '\n';
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
            average_os << std::fixed << std::setprecision(2) << supp_average[i].first / supp_average[i].second << '\n';
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
                ob.Cmessage("Average Test With Steps on '" + dataset_name + "': Unable to create " + script_os_path.string());
                // TODO
            }

            script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
            script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

            script_os << "reset" << '\n';
            script_os << "cd '" << current_output_path.string() << "\'" << '\n';
            script_os << "set grid ytic" << '\n';
            script_os << "set grid" << '\n' << '\n';

            script_os << "# " << dataset_name << "(COLORS)" << '\n';
            script_os << "set output \"" + output_graph + "\"" << '\n';

            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " colors" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

            script_os << "# Graph style" << '\n';
            script_os << "set style data histogram" << '\n';
            script_os << "set style histogram cluster gap 1" << '\n';
            script_os << "set style fill solid 0.25 border -1" << '\n';
            script_os << "set boxwidth 0.9" << '\n' << '\n';

            script_os << "# Get stats to set labels" << '\n';
            script_os << "stats \"" << output_average_results << "\" using 3 nooutput" << '\n';
            script_os << "ymax = STATS_max + (STATS_max/100)*10" << '\n';
            script_os << "xw = 0" << '\n';
            script_os << "yw = (ymax)/22.0" << '\n' << '\n';

            script_os << "# Axes labels" << '\n';
            script_os << "set xtic rotate by -45 scale 0" << '\n';
            script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

            script_os << "# Axes range" << '\n';
            script_os << "set yrange[0:ymax]" << '\n';
            script_os << "set xrange[*:*]" << '\n' << '\n';

            script_os << "# Legend" << '\n';
            script_os << "set key off" << '\n' << '\n';

            script_os << "# Plot" << '\n';
            script_os << "plot \\" << '\n';

            script_os << "'" + output_average_results + "' using 3:xtic(1), '" << output_average_results << "' using ($0 - xw) : ($3 + yw) : (stringcolumn(3)) with labels" << '\n' << '\n';

            script_os << "# Replot in latex folder" << '\n';
            script_os << "set title \"\"" << '\n' << '\n';
            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + "_" + output_graph)).string() << "\'" << '\n';
            script_os << "replot" << '\n' << '\n';

            script_os << "# " << dataset_name << "(BLACK AND WHITE)" << '\n';
            script_os << "set output \"" + output_graph_bw + "\"" << '\n';

            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " black and white" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

            script_os << "replot" << '\n' << '\n';

            script_os << "exit gnuplot" << '\n';

            average_os.close();
            script_os.close();
        } // GNUPLOT SCRIPT

        if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + cfg_.gnuplot_script_extension)).string() + "\"").c_str())) {
            ob.Cmessage("Average Test on '" + dataset_name + "': Unable to run gnuplot's script");
            // TODO
        }
    } // END DATASET FOR
}

void YacclabTests::AverageTestWithSteps()
{
    // Initialize output message box
    OutputBox ob("Average Tests With Steps");

    string complete_results_suffix = "_results.txt",
        middle_results_suffix = "_run",
        average_results_suffix = "_average.txt",
        average_results_rounded_suffix = "_average_rounded.txt";

    for (unsigned d = 0; d < cfg_.average_datasets_ws.size(); ++d) { // For every dataset in the average list
        String dataset_name(cfg_.average_datasets_ws[d]),
            output_average_results = dataset_name + average_results_suffix,
            output_average_results_rounded = dataset_name + average_results_rounded_suffix,
            output_graph = dataset_name + kTerminalExtension,
            output_graph_bw = dataset_name + "_bw" + kTerminalExtension;

        path dataset_path(cfg_.input_path / path(dataset_name)),
            is_path = dataset_path / path(cfg_.input_txt), // files.txt path
            current_output_path(cfg_.output_path / path(cfg_.average_ws_folder) / path(dataset_name)),
            output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
            output_middle_results_path = current_output_path / path(cfg_.middle_folder),
            average_os_path = current_output_path / path(output_average_results),
            average_rounded_os_path = current_output_path / path(output_average_results_rounded);

        if (!create_directories(current_output_path)) {
            ob.Cerror("Average Test With Steps on '" + dataset_name + "': unable to find/create the output path " + current_output_path.string());
        }

        if (cfg_.average_ws_save_middle_tests) {
            if (!create_directories(output_middle_results_path)) {
                ob.Cerror("Average Test With Steps on '" + dataset_name + "': Unable to find/create the output path " + output_middle_results_path.string());
            }
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

        for (const auto& algo_name : cfg_.ccl_average_ws_algorithms) {
            current_res[algo_name] = Mat1d(filenames_size, StepType::ST_SIZE, numeric_limits<double>::max());
            min_res[algo_name] = Mat1d(filenames_size, StepType::ST_SIZE, numeric_limits<double>::max());
        }

        // Start output message box
        ob.StartRepeatedBox(dataset_name, filenames_size, cfg_.average_ws_tests_number);

        map<String, size_t> algo_pos;
        for (size_t i = 0; i < cfg_.ccl_average_ws_algorithms.size(); ++i)
            algo_pos[cfg_.ccl_average_ws_algorithms[i]] = i;
        auto shuffled_ccl_average_ws_algorithms = cfg_.ccl_average_ws_algorithms;

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

                random_shuffle(begin(shuffled_ccl_average_ws_algorithms), end(shuffled_ccl_average_ws_algorithms));

                // For all the Algorithms in the array
                for (const auto& algo_name : shuffled_ccl_average_ws_algorithms) {
                    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
                    unsigned n_labels;
                    unsigned i = algo_pos[algo_name];

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
                    algorithm->FreeLabelingData();
                } // END ALGORITHMS FOR
            } // END FILES FOR.
            ob.StopRepeatedBox();

            // Save middle results if necessary (flag 'average_save_middle_tests' enabled)
            if (cfg_.average_ws_save_middle_tests) {
                string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + to_string(test) + ".txt")).string();
                SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames, cfg_.ccl_average_ws_algorithms);
            }
        }// END TESTS FOR

        // To write in a file min results
        SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames, cfg_.ccl_average_ws_algorithms);

        // If true the i-th step is used by all the algorithms
        vector<bool> steps_presence(StepType::ST_SIZE, false);

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

            // Matrix reduce done, save the results into the average file
            for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
                StepType step = static_cast<StepType>(step_number);
                double avg{ 0.0 };
                if (supp_average[step_number].first > 0.0 && supp_average[step_number].second > 0) {
                    steps_presence[step_number] = true;
                    avg = supp_average[step_number].first / supp_average[step_number].second;
                }
                else {
                    // The current step is not threated by the current algorithm, write 0
                }
                average_ws_results_[dataset_name](a, step_number) = avg;
            }
        }

        // Write the results stored in average_ws_results_ in file
        SaveAverageWithStepsResults(average_os_path.string(), dataset_name, false);
        SaveAverageWithStepsResults(average_rounded_os_path.string(), dataset_name, true);

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
                ob.Cmessage("Average Test With Steps on '" + dataset_name + "': Unable to create " + script_os_path.string());
                // TODO
            }

            script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
            script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

            script_os << "reset" << '\n';
            script_os << "cd '" << current_output_path.string() << "\'" << '\n';
            script_os << "set grid ytic" << '\n';
            script_os << "set grid" << '\n' << '\n';

            script_os << "# " << dataset_name << "(COLORS)" << '\n';
            script_os << "set output \"" + output_graph + "\"" << '\n';

            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " colors" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

            script_os << "# Graph style" << '\n';
            script_os << "set style data histogram" << '\n';
            script_os << "set style histogram cluster gap 1" << '\n';
            script_os << "set style histogram rowstacked" << '\n';
            script_os << "set style fill solid 0.25 border -1" << '\n';
            script_os << "set boxwidth 0.6" << '\n' << '\n';

            script_os << "# Get stats to set labels" << '\n';
            script_os << "stats \"" << output_average_results_rounded << "\" using 6 nooutput" << '\n';
            script_os << "ymax = STATS_max + (STATS_max/100)*10" << '\n';
            script_os << "xw = 0" << '\n';
            script_os << "yw = (ymax)/22.0" << '\n' << '\n';

            script_os << "# Axes labels" << '\n';
            script_os << "set xtic rotate by -45 scale 0" << '\n';
            script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

            script_os << "# Axes range" << '\n';
            script_os << "set yrange[0:ymax]" << '\n';
            script_os << "set xrange[*:*]" << '\n' << '\n';

            script_os << "# Legend" << '\n';
            script_os << "set key outside left font ', 8'" << '\n' << '\n';

            script_os << "# Plot" << '\n';
            script_os << "plot \\" << '\n';

            script_os << "'" + output_average_results_rounded + "' using 2:xtic(1) title '" << Step(static_cast<StepType>(0)) << "', \\" << '\n';
            unsigned i = 3;
            // Start from the second step
            for (int step_number = 1; step_number != StepType::ST_SIZE; ++step_number, ++i) {
                StepType step = static_cast<StepType>(step_number);
                // Add the step only if it used by almost an algorithm
                if (steps_presence[step_number]) {
                    script_os << "'' u " << i << " title '" << Step(step) << "', \\" << '\n';
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
                script_os << ") - ($" << i << "/2)) : ($" << i << "!=0.0 ? sprintf(\"%4.2f\",$" << i << "):'') w labels font 'Tahoma, 8' title '', \\" << '\n';
            }
            script_os << "'' u ($0) : ($" << i << " + yw) : ($" << i << "!=0.0 ? sprintf(\"%4.2f\",$" << i << "):'') w labels font 'Tahoma' title '', \\" << '\n';

            script_os << "# Replot in latex folder" << '\n';
            script_os << "set title \"\"" << '\n' << '\n';
            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + "_with_steps_" + output_graph)).string() << "\'" << '\n';
            script_os << "replot" << '\n' << '\n';

            script_os << "# " << dataset_name << "(BLACK AND WHITE)" << '\n';
            script_os << "set output \"" + output_graph_bw + "\"" << '\n';

            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " black and white" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

            script_os << "replot" << '\n' << '\n';

            script_os << "exit gnuplot" << '\n';

            script_os.close();
        } // End GNUPLOT SCRIPT

        if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + cfg_.gnuplot_script_extension)).string() + "\"").c_str())) {
            ob.Cmessage("Average Test With Steps on '" + dataset_name + "': Unable to run gnuplot's script");
            // TODO
        }
    }
}

void YacclabTests::DensityTest()
{
    OutputBox ob("Density Tests");

    string complete_results_suffix = "_results.txt",
        middle_results_suffix = "_run",
        density_results_suffix = "_density.txt",
        normalized_density_results_suffix = "_normalized_density.txt",
        size_results_suffix = "_size.txt",
        null_results_suffix = "_null_results.txt";

    // Initialize results container
    density_results_ = cv::Mat1d(cfg_.density_datasets.size(), cfg_.ccl_average_algorithms.size(), std::numeric_limits<double>::max());

    for (unsigned d = 0; d < cfg_.density_datasets.size(); ++d) { // For every dataset in the density list
        String dataset_name(cfg_.density_datasets[d]),
            output_density_results = dataset_name + density_results_suffix,
            output_size_results = dataset_name + size_results_suffix,
            output_density_graph = dataset_name + "_density" + kTerminalExtension,
            output_density_bw_graph = dataset_name + "_density_bw" + kTerminalExtension,
            output_size_graph = dataset_name + "_size" + kTerminalExtension,
            output_size_bw_graph = dataset_name + "_size_bw" + kTerminalExtension,
            output_null = dataset_name + null_results_suffix;

        path dataset_path(cfg_.input_path / path(dataset_name)),
            is_path = dataset_path / path(cfg_.input_txt), // files.txt path
            current_output_path(cfg_.output_path / path(cfg_.density_folder) / path(dataset_name)),
            output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
            output_colored_images_path = current_output_path / path(cfg_.colors_folder),
            output_middle_results_path = current_output_path / path(cfg_.middle_folder),
            density_os_path = current_output_path / path(output_density_results),
            size_os_path = current_output_path / path(output_size_results),
            null_os_path = current_output_path / path(output_null);

        if (!create_directories(current_output_path)) {
            ob.Cwarning("unable to find/create the output path '" + current_output_path.string() + ", skipped", dataset_name);
            continue;
        }

        if (cfg_.density_color_labels) {
            if (!create_directories(output_colored_images_path)) {
                ob.Cmessage("Density Test on '" + dataset_name + "': Unable to find/create the output path " + output_colored_images_path.string());
                // TODO
            }
        }

        if (cfg_.density_save_middle_tests) {
            if (!create_directories(output_middle_results_path)) {
                ob.Cmessage("Density Test on '" + dataset_name + "': Unable to find/create the output path " + output_middle_results_path.string());
                // TODO
            }
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

        /*
        Note that number of random_images is less than 800, this is why the second element of the
        pair has uint16_t data type. Extern vector represent the algorithms, inner vector represent
        density for "supp_density" variable and dimension for "supp_dimension" one. In particular:

        FOR "supp_density" VARIABLE:
        INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
        INNER_VECTPR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
        .. and so on;

        SO:
          supp_density[0][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
          for algorithm in position 0;

          supp_density[0][1] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
          for algorithm in position 0;

          supp_density[1][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
          for algorithm in position 1;
        .. and so on

        FOR "SUP_DIMENSION VARIABLE":
        INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_32*32_DIMENSION, COUNT_OF_THAT_IMAGES }
        INNER_VECTOR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_64*64_DIMENSION, COUNT_OF_THAT_IMAGES }

        view "supp_density" explanation for more details;
       */

        uint8_t density = 9 /*[0.1,0.9]*/, size = 8 /*[32,64,128,256,512,1024,2048,4096]*/;

        using vvp = vector<vector<pair<double, uint16_t>>>;
        vvp supp_density(cfg_.ccl_average_algorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
        vvp supp_size(cfg_.ccl_average_algorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));

        // Start output message box
        ob.StartRepeatedBox(dataset_name, filenames_size, cfg_.density_tests_number);

        map<String, size_t> algo_pos;
        for (size_t i = 0; i < cfg_.ccl_average_algorithms.size(); ++i)
            algo_pos[cfg_.ccl_average_algorithms[i]] = i;
        auto shuffled_ccl_average_algorithms = cfg_.ccl_average_algorithms;

        // Test is executed n_test times
        for (unsigned test = 0; test < cfg_.density_tests_number; ++test) {
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

                random_shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms));

                for (const auto& algo_name : shuffled_ccl_average_algorithms) {
                    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
                    unsigned n_labels;
                    unsigned i = algo_pos[algo_name];

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
                    if (cfg_.density_color_labels && test == 0) {
                        // Remove gnuplot escape character from output filename
                        Mat3b imgColors;

                        NormalizeLabels(algorithm->img_labels_);
                        ColorLabels(algorithm->img_labels_, imgColors);
                        String colored_image = (output_colored_images_path / path(filename + "_" + algo_name + ".png")).string();
                        imwrite(colored_image, imgColors);
                    }
                    algorithm->FreeLabelingData();
                } // END ALGORITHMS FOR
            } // END FILES FOR.
            ob.StopRepeatedBox();

            // Save middle results if necessary (flag 'density_save_middle_tests' enabled)
            if (cfg_.density_save_middle_tests) {
                string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + to_string(test) + ".txt")).string();
                SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames, cfg_.ccl_average_algorithms);
            }
        } // END TEST FOR

        // To write in a file min results
        SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames, cfg_.ccl_average_algorithms);

        // To sum min results, in the correct manner, before make average
        for (unsigned files = 0; files < filenames.size(); ++files) {
            // Note that files correspond to min_res rows
            for (int c = 0; c < min_res.cols; ++c) {
                // Add current time to "supp_density" and "supp_size" in the correct position
                if (isdigit(filenames[files].first[0]) && isdigit(filenames[files].first[1]) && isdigit(filenames[files].first[2]) && filenames[files].second) {
                    // superfluous test?
                    if (min_res(files, c) != numeric_limits<double>::max()) {
                        // For density graph
                        supp_density[c][ctoi(filenames[files].first[1])].first += min_res(files, c);
                        supp_density[c][ctoi(filenames[files].first[1])].second++;

                        // For dimension graph
                        supp_size[c][ctoi(filenames[files].first[0])].first += min_res(files, c);
                        supp_size[c][ctoi(filenames[files].first[0])].second++;
                    }
                }
                // Add current time to "supp_density" and "supp_size" in the correct position
            }
        }

        // To calculate average times
        vector<vector<long double>> density_average(cfg_.ccl_average_algorithms.size(), vector<long double>(density));
        vector<vector<long double>> size_average(cfg_.ccl_average_algorithms.size(), vector<long double>(size));

        for (unsigned i = 0; i < cfg_.ccl_average_algorithms.size(); ++i) {
            // For all algorithms
            for (unsigned j = 0; j < density_average[i].size(); ++j) {
                // For all density and normalized density
                if (supp_density[i][j].second != 0) {
                    density_average[i][j] = supp_density[i][j].first / supp_density[i][j].second;
                }
                else {
                    // If there is no element with this density characteristic the average value is set to zero
                    density_average[i][j] = 0.0;
                }
            }
            for (unsigned j = 0; j < size_average[i].size(); ++j) {
                // For all size
                if (supp_size[i][j].second != 0)
                    size_average[i][j] = supp_size[i][j].first / supp_size[i][j].second;
                else
                    size_average[i][j] = 0.0;  // If there is no element with this size characteristic the average value is set to zero
            }
        }

        // For DENSITY RESULT
        ofstream density_os(density_os_path.string());
        if (!density_os.is_open()) {
            ob.Cmessage("Density Test on '" + dataset_name + "': Unable to open " + density_os_path.string());
            // TODO
        }

        // For SIZE RESULT
        ofstream size_os(size_os_path.string());
        if (!size_os.is_open()) {
            ob.Cmessage("Density Test on '" + dataset_name + "': Unable to open " + size_os_path.string());
            // TODO
        }

        // To write density result on specified file
        for (unsigned i = 0; i < density; ++i) {
            // For every density
            if (density_average[0][i] == 0.0) { // Check it only for the first algorithm (it is the same for the others)
                density_os << "#"; // It means that there is no element with this density characteristic
            }
            density_os << ((float)(i + 1) / 10) << "\t"; //Density value
            for (unsigned j = 0; j < density_average.size(); ++j) {
                // For every algorithm
                density_os << density_average[j][i] << "\t";
            }
            density_os << '\n'; // End of current line (current density)
        }
        // To set sizes's label
        vector <pair<unsigned, double>> supp_size_labels(size, make_pair(0, 0));

        // To write size result on specified file
        for (unsigned i = 0; i < size; ++i) {
            // For every size
            if (size_average[0][i] == 0.0) // Check it only for the first algorithm (it is the same for the others)
                size_os << "#"; // It means that there is no element with this size characteristic
            supp_size_labels[i].first = (int)(pow(2, i + 5));
            supp_size_labels[i].second = size_average[0][i];
            size_os << (int)pow(supp_size_labels[i].first, 2) << "\t"; //Size value
            for (unsigned j = 0; j < size_average.size(); ++j) {
                // For every algorithms
                size_os << size_average[j][i] << "\t";
            }
            size_os << '\n'; // End of current line (current size)
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
                ob.Cmessage("Density Test With Steps on '" + dataset_name + "': Unable to create " + script_os_path.string());
                // TODO
            }

            script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
            script_os << "# comment fifth line, open gnuplot's terminal, move to script's path and launch 'load ";
            script_os << dataset_name + cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

            script_os << "reset" << '\n';
            script_os << "cd '" << current_output_path.string() << "\'" << '\n';
            script_os << "set grid" << '\n' << '\n';

            // DENSITY
            script_os << "# DENSITY GRAPH (COLORS)" << '\n' << '\n';

            script_os << "set output \"" + output_density_graph + "\"" << '\n';
            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " colors" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

            script_os << "# Axes labels" << '\n';
            script_os << "set xlabel \"Density\"" << '\n';
            script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

            script_os << "# Get stats to set labels" << '\n';
            script_os << "stats[1:8] '" + output_density_results + "' matrix name 'density' noout" << '\n';
            script_os << "stats[1:8] '" + output_size_results + "' matrix name 'size' noout" << '\n';
            script_os << " ymax = density_max + (density_max / 100) * 10" << '\n';
            script_os << " ymin = density_min - (density_min / 100) * 10" << '\n';

            script_os << "# Axes range" << '\n';
            script_os << "set xrange [0:1]" << '\n';
            script_os << "set yrange [ymin:ymax]" << '\n';
            script_os << "set logscale y" << '\n' << '\n';

            script_os << "# Legend" << '\n';
            script_os << "set key right outside nobox spacing 2 font ', 8'" << '\n' << '\n';

            script_os << "# Plot" << '\n';
            script_os << "plot \\" << '\n';
            vector<String>::iterator it; // I need it after the cycle
            unsigned i = 2;
            for (it = cfg_.ccl_average_algorithms.begin(); it != (cfg_.ccl_average_algorithms.end() - 1); ++it, ++i) {
                script_os << "\"" + output_density_results + "\" using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(string(*it)) + "\" , \\" << '\n';
            }
            script_os << "\"" + output_density_results + "\" using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(string(*it)) + "\"" << '\n' << '\n';

            script_os << "# Replot in latex folder" << '\n';
            script_os << "set title \"\"" << '\n' << '\n';

            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + output_density_graph)).string() << "\'" << '\n';
            script_os << "replot" << '\n' << '\n';

            script_os << "# DENSITY GRAPH (BLACK AND WHITE)" << '\n' << '\n';

            script_os << "set output \"" + output_density_bw_graph + "\"" << '\n';
            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " black and white" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

            script_os << "replot" << '\n' << '\n';

            // SIZE
            script_os << "# SIZE GRAPH (COLORS)" << '\n' << '\n';

            script_os << "set output \"" + output_size_graph + "\"" << '\n';
            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " colors" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

            script_os << "# Axes labels" << '\n';
            script_os << "set xlabel \"Pixels\"" << '\n';
            script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

            script_os << "# Get stats to set labels" << '\n';
            script_os << " ymax = size_max + (size_max/100)*30" << '\n';
            script_os << " ymin = size_min - (size_min/100)*30" << '\n';

            script_os << "# Axes range" << '\n';
            script_os << "set format x \"10^{%L}\"" << '\n';
            script_os << "set xrange [100:100000000]" << '\n';
            script_os << "set yrange [*:*]" << '\n';
            script_os << "set logscale xy 10" << '\n' << '\n';

            script_os << "# Legend" << '\n';
            script_os << "set key right outside top nobox spacing 2 font ', 8'" << '\n';

            script_os << "# Plot" << '\n';
            //// Set Labels
            //for (unsigned i=0; i < supp_size_labels.size(); ++i){
            //	if (supp_size_labels[i].second != 0){
            //		script_os << "set label " << i+1 << " \"" << supp_size_labels[i].first << "x" << supp_size_labels[i].first << "\" at " << pow(supp_size_labels[i].first,2) << "," << supp_size_labels[i].second << '\n';
            //	}
            //	else{
            //		// It means that there is no element with this size characteristic so this label is not necessary
            //	}
            //}
            //// Set Labels
            script_os << "plot \\" << '\n';
            //vector<pair<CCLPointer, string>>::iterator it; // I need it after the cycle
            //unsigned i = 2;
            i = 2;
            for (it = cfg_.ccl_average_algorithms.begin(); it != (cfg_.ccl_average_algorithms.end() - 1); ++it, ++i) {
                script_os << "\"" + output_size_results + "\" using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(string(*it)) + "\" , \\" << '\n';
            }
            script_os << "\"" + output_size_results + "\" using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(string(*it)) + "\"" << '\n' << '\n';

            script_os << "# Replot in latex folder" << '\n';
            script_os << "set title \"\"" << '\n';
            script_os << "set output \'" << (cfg_.latex_path / path(compiler_name + compiler_version + output_size_graph)).string() << "\'" << '\n';
            script_os << "replot" << '\n' << '\n';

            script_os << "# SIZE (BLACK AND WHITE)" << '\n' << '\n';

            script_os << "set output \"" + output_size_bw_graph + "\"" << '\n';
            script_os << "set title " << GetGnuplotTitle() << '\n' << '\n';

            script_os << "# " << kTerminal << " black and white" << '\n';
            script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

            script_os << "replot" << '\n' << '\n';

            script_os << "exit gnuplot" << '\n';

            density_os.close();
            size_os.close();
            script_os.close();
            // GNUPLOT SCRIPT
        }

        if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + cfg_.gnuplot_script_extension)).string() + "\"").c_str())) {
            ob.Cmessage("Density Test on '" + dataset_name + "': Unable to run gnuplot's script");
            // TODO
        }
    }
}

void YacclabTests::MemoryTest()
{
    // Initialize output message box
    OutputBox ob("Memory Tests");

    path current_output_path(cfg_.output_path / path(cfg_.memory_folder));

    String output_file((current_output_path.string() / path(cfg_.memory_file)).string());

    if (!create_directories(current_output_path)) {
        ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + "', memory tests skipped");
        return;
    }

    // To write MEMORY results
    ofstream os(output_file);
    if (!os.is_open()) {
        ob.Cwarning("Unable to open '" + output_file + "', memory tests skipped");
        return;
    }
    os << "#Average number of accesses" << '\n';

    for (unsigned d = 0; d < cfg_.memory_datasets.size(); ++d) { // For every dataset in the average list
        String dataset_name(cfg_.memory_datasets[d]);

        path dataset_path(cfg_.input_path / path(dataset_name)),
            is_path = dataset_path / path(cfg_.input_txt); // files.txt path

        // To save list of filename on which CLLAlgorithms must be tested
        vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
        if (!LoadFileList(filenames, is_path)) {
            ob.Cwarning("Unable to open '" + is_path.string() + "', skipped", dataset_name);
            continue;
        }

        // Number of files
        int filenames_size = filenames.size();

        unsigned tot_test = 0; // To count the real number of image on which labeling will be applied for every file in list

        // Initialize results container
        // To store average memory accesses (one column for every data_ structure type: col 1 -> BINARY_MAT, col 2 -> LABELED_MAT, col 3 -> EQUIVALENCE_VET, col 0 -> OTHER)
        memory_accesses_[dataset_name] = Mat1d(Size(MD_SIZE, cfg_.ccl_mem_algorithms.size()), 0.0);

        // Start output message box
        ob.StartUnitaryBox(dataset_name, filenames_size);

        // For every file in list
        for (unsigned file = 0; file < filenames.size(); ++file) {
            // Display output message box
            ob.UpdateUnitaryBox(file);

            string filename = filenames[file].first;
            path filename_path = dataset_path / path(filename);

            // Read and load image
            if (!GetBinaryImage(filename_path, Labeling::img_)) {
                ob.Cwarning("Unable to open '" + filename + "'");
                continue;
            }

            ++tot_test;

            // For all the Algorithms in the array
            for (size_t i = 0; i < cfg_.ccl_mem_algorithms.size(); ++i) {
                Labeling *algorithm = LabelingMapSingleton::GetLabeling(cfg_.ccl_mem_algorithms[i]);
                // The following data_ structure is used to get the memory access matrices
                vector<unsigned long int> accesses; // Rows represents algorithms and columns represent data_ structures

                algorithm->PerformLabelingMem(accesses);

                // For every data_ structure "returned" by the algorithm
                for (size_t a = 0; a < accesses.size(); ++a) {
                    memory_accesses_[dataset_name](i, a) += accesses[a];
                }
                algorithm->FreeLabelingData();
            }
        }
        ob.StopUnitaryBox();

        // To calculate average memory accesses
        for (int r = 0; r < memory_accesses_[dataset_name].rows; ++r) {
            for (int c = 0; c < memory_accesses_[dataset_name].cols; ++c) {
                memory_accesses_[dataset_name](r, c) /= tot_test;
            }
        }

        os << "#" << dataset_name << '\n';
        os << "Algorithm\tBinary Image\tLabel Image\tEquivalence Vector/s\tOther\tTotal Accesses" << '\n';

        for (size_t a = 0; a < cfg_.ccl_mem_algorithms.size(); ++a) {
            double total_accesses{ 0.0 };
            os << cfg_.ccl_mem_algorithms[a] << '\t';
            for (int col = 0; col < memory_accesses_[dataset_name].cols; ++col) {
                os << std::fixed << std::setprecision(0) << memory_accesses_[dataset_name](a, col);
                os << '\t';
                total_accesses += memory_accesses_[dataset_name](a, col);
            }

            os << total_accesses;
            os << '\n';
        }

        os << '\n' << '\n';;
    }

    os.close();
}

void YacclabTests::LatexGenerator()
{
    OutputBox ob("Generation of Latex file/s");
    path latex = cfg_.latex_path / path(cfg_.latex_file);
    ofstream os(latex.string());
    if (!os.is_open()) {
        ob.Cwarning("Unable to open/create '" + latex.string() + "', generation skipped");
        return;
    }

    // fixed number of decimal values
    os << fixed;
    os << setprecision(3);

    // Document begin
    os << "%These file is generated by YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << '\n' << '\n';
    os << "\\documentclass{article}" << '\n' << '\n';

    os << "\\usepackage{siunitx}" << '\n';
    os << "\\usepackage{graphicx}" << '\n';
    os << "\\usepackage{subcaption}" << '\n';
    os << "\\usepackage[top = 1in, bottom = 1in, left = 1in, right = 1in]{geometry}" << '\n' << '\n';

    os << "\\title{{ \\huge\\bfseries YACCLAB TESTS}}" << '\n';
    os << "\\date{" + GetDatetime() + "}" << '\n';
    os << "\\author{}" << '\n' << '\n';
    os << "\\begin{document}" << '\n' << '\n';
    os << "\\maketitle" << '\n' << '\n';

    // Section average results table ------------------------------------------------------------------------------------------
    if (cfg_.perform_average) {
        os << "\\section{Average Table Results}" << '\n' << '\n';

        os << "\\begin{table}[tbh]" << '\n' << '\n';
        os << "\t\\centering" << '\n';
        os << "\t\\caption{Average Results in ms (Lower is better)}" << '\n';
        os << "\t\\label{tab:table1}" << '\n';
        os << "\t\\begin{tabular}{|l|";
        for (unsigned i = 0; i < cfg_.ccl_average_algorithms.size(); ++i)
            os << "S[table-format=2.3]|";
        os << "}" << '\n';
        os << "\t\\hline" << '\n';
        os << "\t";
        for (unsigned i = 0; i < cfg_.ccl_average_algorithms.size(); ++i) {
            //RemoveCharacter(datasets_name, '\\');
            //datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
            os << " & {" << EscapeUnderscore(cfg_.ccl_average_algorithms[i]) << "}"; //Header
        }
        os << "\\\\" << '\n';
        os << "\t\\hline" << '\n';
        for (unsigned i = 0; i < cfg_.average_datasets.size(); ++i) {
            os << "\t" << cfg_.average_datasets[i];
            for (int j = 0; j < average_results_.cols; ++j) {
                os << " & ";
                if (average_results_(i, j) != numeric_limits<double>::max())
                    os << average_results_(i, j); //Data
            }
            os << "\\\\" << '\n';
        }
        os << "\t\\hline" << '\n';
        os << "\t\\end{tabular}" << '\n' << '\n';
        os << "\\end{table}" << '\n';
    }

    { // CHARTS SECTION ------------------------------------------------------------------------------------------
        SystemInfo s_info;
        string info_to_latex = s_info.build() + "_" + s_info.compiler_name() + s_info.compiler_version() + "_" + s_info.os();
        std::replace(info_to_latex.begin(), info_to_latex.end(), ' ', '_');
        info_to_latex = EscapeUnderscore(info_to_latex);

        string chart_size{ "0.45" }, chart_width{ "1" };
        // Get information about date and time
        string datetime = GetDatetime();

        string compiler_name(s_info.compiler_name());
        string compiler_version(s_info.compiler_version());
        //replace the . with _ for compiler strings
        std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

        // SECTION AVERAGE CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_average) {
            os << "\\section{Average Charts}" << '\n' << '\n';
            os << "\\begin{figure}[tbh]" << '\n' << '\n';
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << '\n';
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
            os << "\t\\centering" << '\n';

            for (unsigned i = 0; i < cfg_.average_datasets.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
                os << "\t\t\\caption{" << cfg_.average_datasets[i] + "}" << '\n';
                os << "\t\t\\centering" << '\n';
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + cfg_.average_datasets[i] + ".pdf}" << '\n';
                os << "\t\\end{subfigure}" << '\n' << '\n';
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
            os << "\\end{figure}" << '\n' << '\n';
        }

        // SECTION AVERAGE WITH STEPS CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_average_ws) {
            string average_ws_suffix{ "_with_steps_" };

            os << "\\section{Average With Steps Charts}" << '\n' << '\n';
            os << "\\begin{figure}[tbh]" << '\n' << '\n';
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << '\n';
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
            os << "\t\\centering" << '\n';
            for (unsigned i = 0; i < cfg_.average_datasets_ws.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
                os << "\t\t\\caption{" << cfg_.average_datasets_ws[i] + "}" << '\n';
                os << "\t\t\\centering" << '\n';
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + average_ws_suffix + cfg_.average_datasets_ws[i] + ".pdf}" << '\n';
                os << "\t\\end{subfigure}" << '\n' << '\n';
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
            os << "\\end{figure}" << '\n' << '\n';
        }

        // SECTION DENSITY CHARTS  ---------------------------------------------------------------------------
        if (cfg_.perform_density) {
            vector<String> density_datasets{ "density", "size" };

            os << "\\section{Density Charts}" << '\n' << '\n';
            os << "\\begin{figure}[tbh]" << '\n' << '\n';
            // \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
            os << "\t\\newcommand{\\machineName}{";
            os << info_to_latex << "}" << '\n';
            // \newcommand{\compilerName}{MSVC15_0}
            os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
            os << "\t\\centering" << '\n';

            for (unsigned i = 0; i < density_datasets.size(); ++i) {
                os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
                os << "\t\t\\caption{" << density_datasets[i] + "}" << '\n';
                os << "\t\t\\centering" << '\n';
                os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + density_datasets[i] + ".pdf}" << '\n';
                os << "\t\\end{subfigure}" << '\n' << '\n';
            }
            os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
            os << "\\end{figure}" << '\n' << '\n';
        }
    } // END CHARTS SECTION

    // SECTION MEMORY RESULT TABLE ---------------------------------------------------------------------------
    if (cfg_.perform_memory) {
        os << "\\section{Memory Accesses tests}" << '\n' << '\n';
        os << "Analysis of memory accesses required by connected components computation. The numbers are given in millions of accesses." << '\n';

        for (const auto& dataset : memory_accesses_) {
            const auto& dataset_name = dataset.first;
            const auto& accesses = dataset.second;

            os << "\\begin{table}[tbh]" << '\n' << '\n';
            os << "\t\\centering" << '\n';
            os << "\t\\caption{Memory accesses on ``" << dataset_name << "'' dataset " << '\n';
            os << "\t\\label{tab:table_" << dataset_name << "}" << '\n';
            os << "\t\\begin{tabular}{|l|";
            for (int i = 0; i < accesses.cols + 1; ++i)
                os << "S[table-format=2.3]|";
            os << "}" << '\n';
            os << "\t\\hline" << '\n';
            os << "\t";

            // Header
            os << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
            os << "\\\\" << '\n';
            os << "\t\\hline" << '\n';

            for (unsigned i = 0; i < cfg_.ccl_mem_algorithms.size(); ++i) {
                // For every algorithm escape the underscore
                const String& alg_name = EscapeUnderscore(cfg_.ccl_mem_algorithms[i]);
                //RemoveCharacter(alg_name, '\\');
                os << "\t{" << alg_name << "}";

                double tot = 0;

                for (int s = 0; s < accesses.cols; ++s) {
                    // For every data_ structure

                    os << "\t& " << (accesses(i, s) / 1000000);

                    tot += (accesses(i, s) / 1000000);
                }
                // Total Accesses
                os << "\t& " << tot;

                // EndLine
                os << "\t\\\\" << '\n';
            }

            // EndTable
            os << "\t\\hline" << '\n';
            os << "\t\\end{tabular}" << '\n' << '\n';
            os << "\\end{table}" << '\n';
        }
    }

    os << "\\end{document}";
    os.close();
}