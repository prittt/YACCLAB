#include "yacclab_tests.h"

#include <fstream>
#include <iostream>	

#include <opencv2/imgproc.hpp>

#include <functional>
#include "labeling_algorithms.h"
#include "utilities.h"
#include "progress_bar.h"

using namespace std;
using namespace cv;

bool YacclabTests::LoadFileList(vector<pair<string, bool>>& filenames, const path& files_path) {

	// Open files_path (files.txt) 
	ifstream is(files_path.string());
	if (!is.is_open()) {
		cmessage("Unable to open '" + files_path.string() + "', " + files_path.stem().string());
		return false;
	}

	string cur_filename;
	while (getline(is, cur_filename)) {
		// To delete possible carriage return in the file name (especially designed for windows file newline format)
		RemoveCharacter(cur_filename, '\r');
		filenames.push_back(make_pair(cur_filename, true));
	}

	is.close();

	return true;
}


void YacclabTests::CheckAlgorithms() {

	if (cfg_.perform_check_8connectivity) {
		TitleBar t_bar("CHECK ALGORITHMS ON 8-CONNECTIVITY");
		t_bar.Start();

		vector<bool> stats(cfg_.ccl_algorithms.size(), true);  // true if the i-th algorithm is correct, false otherwise
		vector<string> firstFail(cfg_.ccl_algorithms.size());  // name of the file on which algorithm fails the first time
		bool stop = false; // true if all algorithms are incorrect
		bool checkPerform = false; // true if almost one check was executed

		for (unsigned i = 0; i < cfg_.check_datasets.size(); ++i) { // For every dataset in the check list	

			path dataset_path(cfg_.input_path / path(cfg_.check_datasets[i]));
			path isPath = dataset_path / path(cfg_.input_txt);
			
			// To save list of filename on which CLLAlgorithms must be tested
			vector<pair<string, bool>> filenames;  // first: filename, second: state of filename (find or not)
			if (!LoadFileList(filenames, isPath)) {
				cerror
				continue;
			}

			// Count number of lines to display progress bar
			unsigned currentNumber = 0;

			ProgressBar p_bar(filenames.size());
			p_bar.Start();

			// For every file in list
			for (unsigned file = 0; file < filenames.size() && !stop; ++file) {
				string filename = filenames[file].first;

				RemoveCharacter(filename, '\r');

				p_bar.Display(currentNumber++);

				path filename_path = dataset_path / path(filename);

				if (!GetBinaryImage(filename_path, Labeling::img_)) {
					cmessage("Unable to open '" + filename + "', file does not exist");
					continue;
				}

				unsigned nLabelsCorrect, nLabelsToControl;

				// SAUF is the reference (the labels are already normalized)
				auto& sauf = LabelingMapSingleton::GetInstance().data_.at("SAUF_RemSp");
				nLabelsCorrect = sauf->PerformLabeling();

				Mat1i& labeledImgCorrect = sauf->img_labels_;
				//nLabelsCorrect = connectedComponents(binaryImg, labeledImgCorrect, 8, 4, CCL_WU);

				unsigned j = 0;
				for (const auto& algo_name : cfg_.ccl_algorithms) {
					auto& algorithm = LabelingMapSingleton::GetInstance().data_.at(algo_name);
					checkPerform = true;
					if (stats[j]) {
						try {
							Mat1i& labeledImgToControl = algorithm->img_labels_;

							nLabelsToControl = 0;// algorithm->PerformLabeling();

							NormalizeLabels(labeledImgToControl);
							const auto diff = CompareMat(labeledImgCorrect, labeledImgToControl);
							if (nLabelsCorrect != nLabelsToControl || !diff) {
								stats[j] = false;
								firstFail[j] = filename_path.string();

								// Stop check test if all the algorithms fail
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
				} // For all the Algorithms in the array
			}// END WHILE (LIST OF IMAGES)
			p_bar.End();
		}// END FOR (LIST OF DATASETS)

		if (checkPerform) {
			unsigned j = 0;
			for (const auto& algo_name : cfg_.ccl_algorithms) {
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
		t_bar.End();
	}
}