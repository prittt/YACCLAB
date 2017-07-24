#ifndef YACCLAB_YACCLAB_TESTS_H_
#define YACCLAB_YACCLAB_TESTS_H_
#include "file_manager.h"

#include <map>

#include "config_data.h"
#include "file_manager.h"

class YacclabTests {
public:
	YacclabTests(const ConfigData& cfg) : cfg_(cfg) {}

	void CheckPerformLabeling();
	void CheckPerformLabelingWithSteps();
	void CheckPerformLabelingMem();

	void CheckAlgorithms(); // TODO move private

    void AverageTest();
    void AverageTestWithSteps();

	//TODO: Check correctness of memory tests also

	//Other test functions

private:
	ConfigData cfg_;
	
	bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const filesystem::path& files_path);
	
    void SaveBroadOutputResults(std::map<cv::String, cv::Mat1d>& results, const filesystem::path& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
};

#endif // !YACCLAB_YACCLAB_TESTS_H_

