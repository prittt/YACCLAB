#include "yacclab_tests.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>	

#include <opencv2/imgproc.hpp>

#include "labeling_algorithms.h"
#include "utilities.h"
#include "progress_bar.h"

using namespace std;
using namespace cv;

// Load a list of image names from a specified file (files_path) and store them into a vector of 
// pairs (filenames). Each pairs contains the name of the file (first) and a bool (second) 
// representing file state.
bool YacclabTests::LoadFileList(vector<pair<string, bool>>& filenames, const path& files_path) {

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

void YacclabTests::CheckPerformLabeling() {}

void YacclabTests::CheckPerformLabelingWithSteps() {}

void YacclabTests::CheckPerformLabelingMem() {}

void YacclabTests::CheckAlgorithms() {

	// Initialize output message box
	OutputBox ob("Checking Algorithms on 8-Connectivity");

	vector<bool> stats(cfg_.ccl_algorithms.size(), true);  // True if the i-th algorithm is correct, false otherwise
	vector<string> first_fail(cfg_.ccl_algorithms.size());  // Name of the file on which algorithm fails the first time
	bool stop = false; // True if all algorithms are not correct
	
	bool correctness_performed = false; // True if at least one check was executed

	for (unsigned i = 0; i < cfg_.check_datasets.size(); ++i) { // For every dataset in the check_datasets list	
		String dataset_name(cfg_.check_datasets[i]); 
		path dataset_path(cfg_.input_path / path(dataset_name));
		path is_path = dataset_path / path(cfg_.input_txt); // files.txt path

		// Load list of images on which ccl_algorithms must be tested
		vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
		if (!LoadFileList(filenames, is_path)) {
			ob.Cerror("Unable to open '" + cfg_.input_txt + "'", dataset_name); 
			continue;
		}

		// Number of files
		unsigned filenames_size = filenames.size();
		// Start output message box
		ob.StartUnitaryBox(dataset_name, filenames_size);

		for (unsigned file = 0; file < filenames_size && !stop; ++file) { // For each file in list
			
			// Display output message box
			ob.UpdateUnitaryBox(file);			
			
			string filename = filenames[file].first;
			path filename_path = dataset_path / path(filename);

			// Read and load image
			if (!GetBinaryImage(filename_path, Labeling::img_)) {
				ob.Cmessage("Unable to open '" + filename + "'");
				continue;
			}

			unsigned n_labels_correct, n_labels_to_control;

			// SAUF is the reference (the labels are already normalized)
			//auto& sauf = LabelingMapSingleton::GetInstance().data_.at("SAUF_UFPC");
			//sauf->PerformLabeling();
			//n_labels_correct = sauf->n_labels_;
			//Mat1i& labeled_img_correct = sauf->img_labels_;

			// TODO: remove OpenCV connectedComponents and use SAUF above
			Mat1i labeled_img_correct;
			n_labels_correct = connectedComponents(Labeling::img_, labeled_img_correct, 8, 4, CCL_WU);

			unsigned j = 0;
			for (const auto& algo_name : cfg_.ccl_algorithms) {
				// Retrieve the algorithm
				Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
				correctness_performed = true;

				// Perform labeling on current algorithm if it has no previously failed
				if (stats[j]) {
					Mat1i& labeled_img_to_control = algorithm->img_labels_;

					try {
						algorithm->PerformLabeling();
						n_labels_to_control = algorithm->n_labels_;

						NormalizeLabels(labeled_img_to_control);
						const auto diff = CompareMat(labeled_img_correct, labeled_img_to_control);
						if (n_labels_correct != n_labels_to_control || !diff) {
							stats[j] = false;
							first_fail[j] = (path(dataset_name) / path(filename)).string() ;

							// Stop check test if all the algorithms fail
							if (adjacent_find(stats.begin(), stats.end(), not_equal_to<int>()) == stats.end()) {
								stop = true;
								break;
							}
						}
					}
					catch (...) {
						ob.Cmessage("'PerformLabeling()' method not implemented in '" + algo_name + "'");
					}
				}
				++j;
			} // For all the Algorithms in the array
		}// END WHILE (LIST OF IMAGES)
		ob.StopUnitaryBox();
	}// END FOR (LIST OF DATASETS)
	
	// To display report of correctness test
	vector<string> messages(cfg_.ccl_algorithms.size());
	unsigned longest_name = max_element(cfg_.ccl_algorithms.begin(), cfg_.ccl_algorithms.end(), CompareLengthCvString)->length();

	unsigned j = 0;
	for (const auto& algo_name : cfg_.ccl_algorithms) {
		messages[j] =  "'" + algo_name + "'" + string(longest_name - algo_name.size(), '-');
		if (stats[j]) {
			if (correctness_performed) {
				messages[j] += "-> correct!";
			}
			else {
				messages[j] += "-> NOT tested!";
			}
		}
		else {
			messages[j] += "-> NOT correct, it first fails on '" + first_fail[j] + "'";
		}
		++j;
	}
	ob.DisplayReport("Report", messages);
}


//void YacclabTests::AverageTest() {
//	if (cfg_.perform_average) {
//
//		string complete_results_suffix = "_results.txt",
//			middle_results_suffix = "_run",
//			average_results_suffix = "_average.txt";
//
//		for (unsigned d = 0; d < cfg_.average_datasets.size(); ++d) { // For every dataset in the average list	
//
//			String dataset_name(cfg_.average_datasets[d]);
//			path dataset_path(cfg_.input_path / path(dataset_name)),
//				is_path = dataset_path / path(cfg_.input_txt),
//				output_path = cfg_.output_path / path(dataset_name),
//				output_colored_images = output_path / path(cfg_.colors_folder),
//				output_middle_results = output_path / path(cfg_.middle_folder);
//
//			if (!create_directories(output_path)) {
//				cerror("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + output_path.string());
//			}
//
//			if (cfg_.average_color_labels) {
//				if (!create_directories(output_colored_images)) {
//					cerror("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + output_colored_images.string());
//				}
//			}
//
//			if (cfg_.average_save_middle_tests) {
//				if (!create_directories(output_middle_results)) {
//					cerror("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + output_middle_results.string());
//				}
//			}
//
//			// For AVERAGES RESULT
//			String average_file_stream((output_path / path(dataset_name + "_" + average_results_suffix)).string());
//			ofstream average_os(average_file_stream);
//			if (!average_os.is_open())
//				cerror("Averages_Test on '" + dataset_name + "': Unable to open " + average_file_stream);
//
//			// To save list of filename on which CLLAlgorithms must be tested
//			vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
//			if (!LoadFileList(filenames, is_path)) {
//				continue;
//			}
//
//			// Number of files
//			int file_number = filenames.size();
//
//			// To save middle/min and average results;
//			Mat1d min_res(file_number, cfg_.ccl_algorithms.size(), numeric_limits<double>::max());
//			Mat1d current_res(file_number, cfg_.ccl_algorithms.size(), numeric_limits<double>::max());
//			Mat1i labels(file_number, cfg_.ccl_algorithms.size(), 0);
//			vector<pair<double, uint16_t>> supp_average(cfg_.ccl_algorithms.size(), make_pair(0.0, 0));
//
//			ProgressBar p_bar(filenames.size());
//			p_bar.Start();
//
//			// Test is executed n_test times
//			for (unsigned test = 0; test < cfg_.average_tests_number; ++test) {
//				// Count number of lines to display "progress bar"
//				unsigned current_number = 0;
//
//				// For every file in list
//				for (unsigned file = 0; file < filenames.size(); ++file) {
//					string filename = filenames[file].first;
//					p_bar.Display(current_number++, test + 1);
//
//					Mat1b binaryImg;
//					path filename_path = dataset_path / path(filename);
//
//					// Load current image file in img_
//					if (!GetBinaryImage(filename_path, Labeling::img_)) {
//						if (filenames[file].second)
//							cout << "'" + filename + "' does not exist" << endl;
//						filenames[file].second = false;
//						continue;
//					}
//
//					unsigned i = 0;
//					// For all the Algorithms in the array
//					for (const auto& algo_name : cfg_.ccl_algorithms) {
//						auto& algorithm = LabelingMapSingleton::GetInstance().data_[algo_name];
//
//						// This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
//						unsigned n_labels;
//
//						// Perform current algorithm on current image and save result.
//
//						algorithm->perf_.start();
//						algorithm->PerformLabeling();
//						algorithm->perf_.stop();
//						n_labels = algorithm->n_labels_;
//
//						// Save number of labels (we reasonably supposed that labels's number is the same on every #test so only the first time we save it)
//						if (test == 0) {
//							labels(file, i) = n_labels;
//						}
//
//						// Save time results
//						current_res(file, i) = algorithm->perf_.last();
//						if (algorithm->perf_.last() < min_res(file, i)) {
//							min_res(file, i) = algorithm->perf_.last();
//						}
//
//						// If 'at_colorLabels' is enabled only the first time (test == 0) the output is saved
//						if (cfg_.average_color_labels && test == 0) {
//							// Remove gnuplot escape character from output filename
//							/*string alg_name = (*it).second;
//							alg_name.erase(std::remove(alg_name.begin(), alg_name.end(), '\\'), alg_name.end());*/
//							Mat3b imgColors;
//
//							NormalizeLabels(algorithm->img_labels_);
//							ColorLabels(algorithm->img_labels_, imgColors);
//							String colored_image = (output_colored_images / path(filename + "_" + algo_name + ".png")).string();
//							imwrite(colored_image, imgColors);
//						}
//						++i;
//					}// END ALGORITHMS FOR
//				} // END FILES FOR.
//				
//				p_bar.End(cfg_.average_tests_number);
//
//				// To write in a file min results
//				saveBroadOutputResults(min_res, osPath, ccl_algorithms, write_n_labels, labels, filenames);
//
//				// To calculate average times and write it on the specified file
//				for (int r = 0; r < min_res.rows; ++r) {
//					for (int c = 0; c < min_res.cols; ++c) {
//						if (min_res(r, c) != numeric_limits<double>::max()) {
//							supp_average[c].first += min_res(r, c);
//							supp_average[c].second++;
//						}
//					}
//				}
//
//				averageOs << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << endl;
//				for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
//					// For all the Algorithms in the array
//					all_res(algPos, i) = supp_average[i].first / supp_average[i].second;
//					averageOs << ccl_algorithms[i] << "\t" << supp_average[i].first / supp_average[i].second << "\t";
//					averageOs << std::fixed << std::setprecision(numberOfDecimalDigitToDisplayInGraph) << supp_average[i].first / supp_average[i].second << endl;
//				}
//
//			}
//		}
//
//
//		string /*dataset_name = dataset_name,*/
//			//completeOutputPath = output_path + kPathSeparator + dataset_name,
//			gnuplotScript = dataset_name + gnuplot_script_extension,
//			outputBroadResults = dataset_name + "_results.txt",
//			middleFile = dataset_name + "_run",
//			outputAverageResults = dataset_name + "_average.txt",
//			outputGraph = dataset_name + kTerminalExtension,
//			outputGraphBw = dataset_name + "_bw" + kTerminalExtension;
//		//middleOutFolder = completeOutputPath + kPathSeparator + middle_folder,
//		//outColorFolder = output_path + kPathSeparator + dataset_name + kPathSeparator + colors_folder;
//
//		path completeOutputPath = output_path / path(dataset_name),
//			middleOutFolder = completeOutputPath / path(middle_folder),
//			outColorFolder = output_path / dataset_name / path(colors_folder),
//			latex_charts_path = output_path / path(latex_folder);
//
//		unsigned numberOfDecimalDigitToDisplayInGraph = 2;
//
//		// Creation of output path
//		/*if (!dirExists(completeOutputPath.c_str()))
//		if (0 != std::system(("mkdir " + completeOutputPath).c_str()))
//		return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + completeOutputPath);*/
//		if (!create_directories(completeOutputPath)) {
//			return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + completeOutputPath.string());
//		}
//
//		if (outputColors) {
//			// Creation of color output path
//			/*if (!dirExists(outColorFolder.c_str()))
//			if (0 != std::system(("mkdir " + outColorFolder).c_str()))
//			return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + outColorFolder);*/
//			if (!create_directories(outColorFolder)) {
//				return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + outColorFolder.string());
//			}
//		}
//
//		if (saveMiddleResults) {
//			/*if (!dirExists(middleOutFolder.c_str()))
//			if (0 != std::system(("mkdir " + middleOutFolder).c_str()))
//			return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + middleOutFolder);*/
//			if (!create_directories(middleOutFolder)) {
//				return ("Averages_Test on '" + dataset_name + "': Unable to find/create the output path " + middleOutFolder.string());
//			}
//		}
//
//		/*string isPath = input_path + kPathSeparator + dataset_name + kPathSeparator + input_txt,
//		osPath = output_path + kPathSeparator + dataset_name + kPathSeparator + outputBroadResults,
//		averageOsPath = output_path + kPathSeparator + dataset_name + kPathSeparator + outputAverageResults;*/
//		path isPath = input_path / path(dataset_name) / path(input_txt),
//			osPath = output_path / path(dataset_name) / path(outputBroadResults),
//			averageOsPath = output_path / path(dataset_name) / path(outputAverageResults);
//
//		// For AVERAGES RESULT
//		ofstream averageOs(averageOsPath.string());
//		if (!averageOs.is_open())
//			return ("Averages_Test on '" + dataset_name + "': Unable to open " + averageOsPath.string());
//		// For LIST OF INPUT IMAGES
//		ifstream is(isPath.string());
//		if (!is.is_open())
//			return ("Averages_Test on '" + dataset_name + "': Unable to open " + isPath.string());
//
//		// To save list of filename on which CLLAlgorithms must be tested
//		vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
//		string filename;
//		while (getline(is, filename)) {
//			// To delete eventual carriage return in the file name (especially designed for windows file newline format)
//			size_t found;
//			do {
//				// The while cycle is probably unnecessary
//				found = filename.find("\r");
//				if (found != string::npos)
//					filename.erase(found, 1);
//			} while (found != string::npos);
//			// Add purified file name in the vector
//			filenames.push_back(make_pair(filename, true));
//		}
//		is.close();
//
//		// Number of files
//		int file_number = filenames.size();
//
//		// To save middle/min and average results;
//		Mat1d min_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
//		Mat1d current_res(file_number, ccl_algorithms.size(), numeric_limits<double>::max());
//		Mat1i labels(file_number, ccl_algorithms.size(), 0);
//		vector<pair<double, uint16_t>> supp_average(ccl_algorithms.size(), make_pair(0.0, 0));
//
//		ProgressBar p_bar(file_number);
//		p_bar.Start();
//
//		// Test is executed n_test times
//		for (unsigned test = 0; test < n_test; ++test) {
//			// Count number of lines to display "progress bar"
//			unsigned currentNumber = 0;
//
//			// For every file in list
//			for (unsigned file = 0; file < filenames.size(); ++file) {
//				filename = filenames[file].first;
//
//				// Display "progress bar"
//				//if (currentNumber * 100 / file_number != (currentNumber - 1) * 100 / file_number)
//				//{
//				//    cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
//				//    fflush(stdout);
//				//}
//				p_bar.Display(currentNumber++, test + 1);
//
//				Mat1b binaryImg;
//				path filename_path = input_path / path(dataset_name) / path(filename);
//
//				if (!GetBinaryImage(filename_path, Labeling::img_)) {
//					if (filenames[file].second)
//						cout << "'" + filename + "' does not exist" << endl;
//					filenames[file].second = false;
//					continue;
//				}
//
//				unsigned i = 0;
//				// For all the Algorithms in the array
//				for (const auto& algo_name : ccl_algorithms) {
//					auto& algorithm = LabelingMapSingleton::GetInstance().data_[algo_name];
//
//					// This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
//					unsigned n_labels;
//
//					// Perform current algorithm on current image and save result.
//
//					algorithm->perf_.start();
//					algorithm->PerformLabeling();
//					algorithm->perf_.stop();
//					n_labels = algorithm->n_labels_;
//
//					// Save number of labels (we reasonably supposed that labels's number is the same on every #test so only the first time we save it)
//					if (test == 0) {
//						labels(file, i) = n_labels;
//					}
//
//					// Save time results
//					current_res(file, i) = algorithm->perf_.last();
//					if (algorithm->perf_.last() < min_res(file, i)) {
//						min_res(file, i) = algorithm->perf_.last();
//					}
//
//					// If 'at_colorLabels' is enabled only the first time (test == 0) the output is saved
//					if (outputColors && test == 0) {
//						// Remove gnuplot escape character from output filename
//						/*string alg_name = (*it).second;
//						alg_name.erase(std::remove(alg_name.begin(), alg_name.end(), '\\'), alg_name.end());*/
//						Mat3b imgColors;
//
//						NormalizeLabels(algorithm->img_labels_);
//						ColorLabels(algorithm->img_labels_, imgColors);
//						path colored_file_path = outColorFolder / path(filename + "_" + algo_name + ".png");
//						imwrite(colored_file_path.string(), imgColors);
//					}
//					++i;
//				}// END ALGORITHMS FOR
//			} // END FILES FOR.
//
//			  // To display "progress bar"
//			  //cout << "Test #" << (test + 1) << ": " << currentNumber << "/" << file_number << "         \r";
//			  //fflush(stdout);
//
//			  // Save middle results if necessary (flag 'average_save_middle_tests' enable)
//			if (saveMiddleResults) {
//				//string middleOut = middleOutFolder + kPathSeparator + middleFile + "_" + to_string(test) + ".txt";
//				path middleOut = middleOutFolder / path(middleFile + "_" + to_string(test) + ".txt");
//				saveBroadOutputResults(current_res, middleOut, ccl_algorithms, write_n_labels, labels, filenames);
//			}
//		}// END TESTS FOR
//
//		p_bar.End(n_test);
//
//		// To write in a file min results
//		saveBroadOutputResults(min_res, osPath, ccl_algorithms, write_n_labels, labels, filenames);
//
//		// To calculate average times and write it on the specified file
//		for (int r = 0; r < min_res.rows; ++r) {
//			for (int c = 0; c < min_res.cols; ++c) {
//				if (min_res(r, c) != numeric_limits<double>::max()) {
//					supp_average[c].first += min_res(r, c);
//					supp_average[c].second++;
//				}
//			}
//		}
//
//		averageOs << "#Algorithm" << "\t" << "Average" << "\t" << "Round Average for Graphs" << endl;
//		for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
//			// For all the Algorithms in the array
//			all_res(algPos, i) = supp_average[i].first / supp_average[i].second;
//			averageOs << ccl_algorithms[i] << "\t" << supp_average[i].first / supp_average[i].second << "\t";
//			averageOs << std::fixed << std::setprecision(numberOfDecimalDigitToDisplayInGraph) << supp_average[i].first / supp_average[i].second << endl;
//		}
//
//		// GNUPLOT SCRIPT
//
//		SystemInfo s_info;
//		string compiler_name(s_info.compiler_name());
//		string compiler_version(s_info.compiler_version());
//		//replace the . with _ for compiler strings
//		std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');
//
//		//string scriptos_path = output_path + kPathSeparator + dataset_name + kPathSeparator + gnuplotScript;
//		path scriptos_path = output_path / path(dataset_name) / path(gnuplotScript);
//
//		ofstream scriptOs(scriptos_path.string());
//		if (!scriptOs.is_open())
//			return ("Averages_Test on '" + dataset_name + "': Unable to create " + scriptos_path.string());
//
//		scriptOs << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
//		scriptOs << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplotScript << "' if you want to run it" << endl << endl;
//
//		scriptOs << "reset" << endl;
//		scriptOs << "cd '" << completeOutputPath.string() << "\'" << endl;
//		scriptOs << "set grid ytic" << endl;
//		scriptOs << "set grid" << endl << endl;
//
//		scriptOs << "# " << dataset_name << "(COLORS)" << endl;
//		scriptOs << "set output \"" + outputGraph + "\"" << endl;
//
//		scriptOs << "set title " << GetGnuplotTitle() << endl << endl;
//
//		scriptOs << "# " << kTerminal << " colors" << endl;
//		scriptOs << "set terminal " << kTerminal << " enhanced color font ',15'" << endl << endl;
//
//		scriptOs << "# Graph style" << endl;
//		scriptOs << "set style data histogram" << endl;
//		scriptOs << "set style histogram cluster gap 1" << endl;
//		scriptOs << "set style fill solid 0.25 border -1" << endl;
//		scriptOs << "set boxwidth 0.9" << endl << endl;
//
//		scriptOs << "# Get stats to set labels" << endl;
//		scriptOs << "stats \"" << outputAverageResults << "\" using 2 nooutput" << endl;
//		scriptOs << "ymax = STATS_max + (STATS_max/100)*10" << endl;
//		scriptOs << "xw = 0" << endl;
//		scriptOs << "yw = (ymax)/22" << endl << endl;
//
//		scriptOs << "# Axes labels" << endl;
//		scriptOs << "set xtic rotate by -45 scale 0" << endl;
//		scriptOs << "set ylabel \"Execution Time [ms]\"" << endl << endl;
//
//		scriptOs << "# Axes range" << endl;
//		scriptOs << "set yrange[0:ymax]" << endl;
//		scriptOs << "set xrange[*:*]" << endl << endl;
//
//		scriptOs << "# Legend" << endl;
//		scriptOs << "set key off" << endl << endl;
//
//		scriptOs << "# Plot" << endl;
//		scriptOs << "plot \\" << endl;
//		scriptOs << "'" + outputAverageResults + "' using 2:xtic(1), '" << outputAverageResults << "' using ($0 - xw) : ($2 + yw) : (stringcolumn(3)) with labels" << endl << endl;
//
//		scriptOs << "# Replot in latex folder" << endl;
//		scriptOs << "set title \"\"" << endl << endl;
//		scriptOs << "set output \'" << (latex_charts_path / path(compiler_name + compiler_version + outputGraph)).string() << "\'" << endl;
//		scriptOs << "replot" << endl << endl;
//
//		scriptOs << "# " << dataset_name << "(BLACK AND WHITE)" << endl;
//		scriptOs << "set output \"" + outputGraphBw + "\"" << endl;
//
//		scriptOs << "set title " << GetGnuplotTitle() << endl << endl;
//
//		scriptOs << "# " << kTerminal << " black and white" << endl;
//		scriptOs << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << endl << endl;
//
//		scriptOs << "replot" << endl << endl;
//
//		scriptOs << "exit gnuplot" << endl;
//
//		averageOs.close();
//		scriptOs.close();
//		// GNUPLOT SCRIPT
//
//		if (0 != std::system(("gnuplot \"" + (completeOutputPath / path(gnuplotScript)).string() + "\"").c_str()))
//			return ("Averages_Test on '" + dataset_name + "': Unable to run gnuplot's script");
//
//		//return ("Averages_Test on '" + dataset_name + "': successfully done");
//		return "";
//
//
//	}