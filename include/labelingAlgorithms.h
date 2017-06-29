#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include "performanceEvaluator.h"

#define upperBound8Conn ((size_t)((aImg.rows + 1) / 2) * (size_t)((aImg.cols + 1) / 2) + 1);

#define REGISTER_LABELING(x) \
class register_##x { \
public: \
	register_##x() { \
		LabelingMapSingleton::GetInstance().data[#x] = new x; \
	} \
} reg_##x

class labeling {
public:
    static cv::Mat1b aImg;
    cv::Mat1i aImgLabels;
    PerformanceEvaluator perf;

    labeling() {}

    virtual ~labeling() {}

    virtual unsigned PerformLabeling() { throw std::runtime_error("Average Test not implemented"); }

    virtual unsigned PerformLabelingWithSteps() { throw std::runtime_error("Average Test with Steps not implemented"); }

    virtual unsigned PerformLabelingMem(std::vector<unsigned long>& accesses) { throw std::runtime_error("Memory Test not implemented"); }

    virtual void AllocateMemory() = 0;

    virtual void DeallocateMemory() = 0;

    static void SetImg(const cv::Mat1b& img) { aImg = img.clone(); }

private:

    virtual unsigned FirstScan() { throw std::runtime_error("First Scan not implemented"); }

    virtual unsigned SecondScan(const unsigned& lunique) { throw std::runtime_error("Second Scan not implemented"); }
};

class LabelingMapSingleton {
public:
    std::map<std::string, labeling*> data;

    static LabelingMapSingleton& GetInstance();

    LabelingMapSingleton(LabelingMapSingleton const&) = delete;

    void operator=(LabelingMapSingleton const&) = delete;

private:
    LabelingMapSingleton() {}
    ~LabelingMapSingleton()
    {
        for (std::map<std::string, labeling*>::iterator it = data.begin(); it != data.end(); ++it)
            delete it->second;
    }
};