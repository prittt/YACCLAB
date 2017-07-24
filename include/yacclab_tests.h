#ifndef YACCLAB_YACCLAB_TESTS_H_
#define YACCLAB_YACCLAB_TESTS_H_

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

	//TODO: Check correctness of memory tests also

	//Other test functions

private:
	ConfigData cfg_;
	
	bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const filesystem::path& files_path);
	

};

#endif // !YACCLAB_YACCLAB_TESTS_H_

