#include "labeling_algorithms.h"

cv::Mat1b Labeling::img_;

LabelingMapSingleton& LabelingMapSingleton::GetInstance()
{
    static LabelingMapSingleton instance;	// Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return instance;
}

Labeling* LabelingMapSingleton::GetLabeling(const std::string& s)
{
    return LabelingMapSingleton::GetInstance().data_.at(s);
}
