// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#ifndef YACCLAB_TESTS_PERFORMER_H_
#define YACCLAB_TESTS_PERFORMER_H_

#include <string>
#include <map>
#include <memory>

#include <opencv2/imgproc.hpp>

#include "file_manager.h"
#include "system_info.h"
#include "utilities.h"
#include "config_data.h"
#include "labeling_algorithms.h"
#include "progress_bar.h"


using namespace filesystem;

class YacclabTests {

private:
	OutputBox ob_;
	std::error_code &ec_;
	ModeConfig &mode_cfg_;
	GlobalConfig &glob_cfg_;
	const path output_path;

	cv::Mat1d average_results_;
	cv::Mat1d density_results_;
	std::map<std::string, cv::Mat> granularity_results_;
	std::map<std::string, cv::Mat1d> average_ws_results_; // String for dataset_name, Mat1d for steps results
	std::map<std::string, cv::Mat1d> memory_accesses_; // String for dataset_name, Mat1d for memory accesses

public:
	YacclabTests(ModeConfig &mode_cfg, GlobalConfig &glob_cfg, std::error_code &ec) : 
		ec_(ec), mode_cfg_(mode_cfg), glob_cfg_(glob_cfg), output_path(glob_cfg.glob_output_path / mode_cfg.mode_output_path) {}

	void CheckPerformLabeling() {
		std::string title = "Checking Correctness of 'PerformLabeling()'";
		CheckAlgorithms(title, mode_cfg_.ccl_average_algorithms, &Labeling::PerformLabeling);
	}
	void CheckPerformLabelingWithSteps() {
		std::string title = "Checking Correctness of 'PerformLabelingWithSteps()'";
		CheckAlgorithms(title, mode_cfg_.ccl_average_ws_algorithms, &Labeling::PerformLabelingWithSteps);
	}
	void CheckPerformLabelingMem() {
		std::string title = "Checking Correctness of 'PerformLabelingMem()'";
		std::vector<uint64_t> unused;
		CheckAlgorithms(title, mode_cfg_.ccl_mem_algorithms, &Labeling::PerformLabelingMem, unused);
	}

    void InitialOperations();

	void AverageTest();
	void AverageTestWithSteps();
	void DensityTest();
	void GranularityTest();
	void MemoryTest();
	void LatexGenerator();

    ~YacclabTests() {
        LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_existing_algorithms[0])->GetInput()->Release();
    }

private:

    void CheckAlgorithmsExistence();
    void CheckMethodsExistence();
    void CheckDatasets();
    void CreateDirectories();

	bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const path& files_path);
	bool CheckFileList(const path& base_path, std::vector<std::pair<std::string, bool>>& filenames);
	bool SaveBroadOutputResults(std::map<std::string, cv::Mat1d>& results, const std::string& o_filename, const cv::Mat1i& labels,
		const std::vector<std::pair<std::string, bool>>& filenames, const std::vector<std::string>& ccl_algorithms);
	bool SaveBroadOutputResults(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels,
		const std::vector<std::pair<std::string, bool>>& filenames, const std::vector<std::string>& ccl_algorithms);
	void SaveAverageWithStepsResults(const std::string& os_name, const std::string& dataset_name, bool rounded);

	template <typename FnP, typename... Args>
	void CheckAlgorithms(const std::string& title, const std::vector<std::string>& ccl_algorithms, const FnP func, Args&&... args) {

		OutputBox ob(title);

		std::vector<bool> stats(ccl_algorithms.size(), true);  // True if the i-th algorithm is correct, false otherwise
		std::vector<std::string> first_fail(ccl_algorithms.size());  // Name of the file on which algorithm fails the first time
		bool stop = false; // True if all the algorithms are not correct

		std::string correct_algo_name;
		try {
			correct_algo_name = LabelingMapSingleton::GetLabeling(ccl_algorithms[0])->CheckAlg();
		}
		catch (std::out_of_range) {
			ob.Cwarning("No correct algorithm is available, correctness test skipped.");
			return;
		}
		Labeling* correct_algo = LabelingMapSingleton::GetLabeling(correct_algo_name);

		for (unsigned i = 0; i < mode_cfg_.check_datasets.size(); ++i) { // For every dataset in the check_datasets list
			std::string dataset_name(mode_cfg_.check_datasets[i]);
			path dataset_path(glob_cfg_.input_path / path(dataset_name));
			path is_path = dataset_path / path(glob_cfg_.input_txt); // files.txt path

			// Load list of images on which ccl_algorithms must be tested
			std::vector<std::pair<std::string, bool>> filenames; // first: filename, second: state of filename (find or not)
			if (!LoadFileList(filenames, is_path)) {
				ob.Cwarning("Unable to open '" + is_path.string() + "'", dataset_name);
				continue;
			}

			// Number of files
			size_t filenames_size = filenames.size();
			ob.StartUnitaryBox(dataset_name, filenames_size);

			for (unsigned file = 0; file < filenames_size && !stop; ++file) { // For each file in list
				ob.UpdateUnitaryBox(file);

				std::string filename = filenames[file].first;
				path filename_path = dataset_path / path(filename);

				// Load image
				if (!correct_algo->GetInput()->ReadBinary(filename_path.string())) {
					ob.Cmessage("Unable to open '" + filename + "'");
					continue;
				}

				// These variables aren't necessary
				// unsigned n_labels_correct, n_labels_to_control;
				
				correct_algo->PerformLabeling();
				//n_labels_correct = sauf->n_labels_;
                YacclabTensorOutput* correct_algo_out = correct_algo->GetOutput();
                correct_algo_out->PrepareForCheck();

                std::unique_ptr<YacclabTensorOutput> labels_correct = correct_algo_out->Copy();

                correct_algo->FreeLabelingData();

                labels_correct->NormalizeLabels(correct_algo->IsLabelBackground());

				unsigned j = 0;
				for (const auto& algo_name : ccl_algorithms) {
					Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);

					// Perform labeling on current algorithm if it has no previously failed
					if (stats[j]) {
						(algorithm->*func)(std::forward<Args>(args)...);
                        YacclabTensorOutput* labels_to_check = algorithm->GetOutput();
                        labels_to_check->PrepareForCheck();
                        labels_to_check->NormalizeLabels(algorithm->IsLabelBackground());
                        const bool correct = labels_correct->Equals(labels_to_check);
                        // const bool diff = algorithm->Check(correct_algo);
						algorithm->FreeLabelingData();
						if (!correct) {
							stats[j] = false;
							first_fail[j] = (path(dataset_name) / path(filename)).string();

							// Stop check test if all the algorithms fail
							if (adjacent_find(stats.begin(), stats.end(), std::not_equal_to<int>()) == stats.end()) {
								stop = true;
								break;
							}
						}
					}
					++j;
				} // For all the Algorithms in the array
                //correct_algo->FreeLabelingData();

			}// END WHILE (LIST OF IMAGES)
			ob.StopUnitaryBox();
		}// END FOR (LIST OF DATASETS)

		// LabelingMapSingleton::GetLabeling(ccl_algorithms[0])->ReleaseInput();

		 // To display report of correctness test
		std::vector<std::string> messages(static_cast<unsigned int>(ccl_algorithms.size()));
		unsigned longest_name = static_cast<unsigned>(max_element(ccl_algorithms.begin(), ccl_algorithms.end(), CompareLengthCvString)->length());

		unsigned j = 0;
		for (const auto& algo_name : ccl_algorithms) {
			messages[j] = "'" + algo_name + "'" + std::string(longest_name - algo_name.size(), '-');
			if (stats[j]) {
				messages[j] += "-> correct!";
			}
			else {
				messages[j] += "-> NOT correct, it first fails on '" + first_fail[j] + "'";
			}
			++j;
		}
		ob.DisplayReport("Report", messages);
	}
};

//using TestsPerfPtr = std::unique_ptr<YacclabTests>;
//
//TestsPerfPtr YacclabTestsFactory(ModeConfig mode_cfg, GlobalConfig glob_cfg, std::error_code& ec) {
//	TestsPerfPtr ptr;
//	if (mode_cfg.mode == "2D_CPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//	else if (mode_cfg.mode == "3D_CPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//#if defined YACCLAB_WITH_CUDA
//	else if (mode_cfg.mode == "2D_GPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//	else if (mode_cfg.mode == "3D_GPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//#endif
//	else ptr = nullptr;
//	return ptr;
//}

#endif // !YACCLAB_TESTS_PERFORMER_H_
