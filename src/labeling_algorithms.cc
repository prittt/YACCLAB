// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling_algorithms.h"

#include "opencv2/imgcodecs.hpp"
#include "volume_util.h"

//#if defined YACCLAB_WITH_CUDA
//#include "cuda_runtime.h"
//#endif

//template <Connectivity Conn>
//cv::Mat1b Labeling2D<Conn>::img_;
//template <Connectivity Conn>
//cv::Mat Labeling3D<Conn>::img_;
//
//#if defined YACCLAB_WITH_CUDA
//template <Connectivity Conn>
//cv::cuda::GpuMat GpuLabeling2D<Conn>::d_img_;
//template <Connectivity Conn>
//cv::cuda::GpuMat3 GpuLabeling3D<Conn>::d_img_;
//#endif

//template <Connectivity Conn>
//std::string Labeling2D<Conn>::GetTitle(const SystemInfo& s_info) {
//	std::string s = "\"{/:Bold CPU}: " + s_info.cpu() + " {/:Bold BUILD}: " + s_info.build() + " {/:Bold OS}: " + s_info.os() +
//		" {/:Bold COMPILER}: " + s_info.compiler_name() + " " + s_info.compiler_version() + "\" font ', 9'";
//	return s;
//}
//template <Connectivity Conn>
//std::string Labeling3D<Conn>::GetTitle(const SystemInfo& s_info) {
//    std::string s = "\"{/:Bold CPU}: " + s_info.cpu() + " {/:Bold BUILD}: " + s_info.build() + " {/:Bold OS}: " + s_info.os() +
//        " {/:Bold COMPILER}: " + s_info.compiler_name() + " " + s_info.compiler_version() + "\" font ', 9'";
//    return s;
//}

//#if defined YACCLAB_WITH_CUDA
//template <Connectivity Conn>
//std::string GpuLabeling2D<Conn>::GetTitle(const SystemInfo& s_info) {
//    CudaInfo cuda_info;
//    std::string s = "\"{/:Bold GPU}: " + cuda_info.device_name_ + " {/:Bold CUDA Capability}: " + cuda_info.cuda_capability_ +
//        " {/:Bold Runtime}: " + cuda_info.runtime_version_ + " {/:Bold Driver}: " + cuda_info.driver_version_;
//    return s;
//}
//template <Connectivity Conn>
//std::string GpuLabeling3D<Conn>::GetTitle(const SystemInfo& s_info) {
//    CudaInfo cuda_info;
//    std::string s = "\"{/:Bold GPU}: " + cuda_info.device_name_ + " {/:Bold CUDA Capability}: " + cuda_info.cuda_capability_ +
//        " {/:Bold Runtime}: " + cuda_info.runtime_version_ + " {/:Bold Driver}: " + cuda_info.driver_version_;
//    return s;
//}
//#endif

//template <Connectivity Conn>
//bool Labeling2D<Conn>::Check(const Labeling *correct_alg) {
//    // correct_alg already ran
//    const Labeling2D<Conn> *correct_alg_2 = dynamic_cast<const Labeling2D<Conn>*>(correct_alg);
//    if (correct_alg_2 == nullptr)
//        return false;
//    return CompareMat(img_labels_, correct_alg_2->img_labels_);
//}

//template <Connectivity Conn>
//bool Labeling3D<Conn>::Check(const Labeling *correct_alg) {
//    // correct_alg already ran
//    const Labeling3D<Conn> *correct_alg_3 = dynamic_cast<const Labeling3D<Conn>*>(correct_alg);
//    if (correct_alg_3 == nullptr)
//        return false;
//    return CompareMat(img_labels_, correct_alg_3->img_labels_);
//}

//template <Connectivity Conn>
//void Labeling2D<Conn>::WriteColoredOutput(std::string filename) {
//    cv::Mat3b img_out;;
//    ColorLabels(img_labels_, img_out);
//    imwrite(filename, img_out);
//}

//template <Connectivity Conn>
//void Labeling3D<Conn>::WriteColoredOutput(std::string filename) {
//    cv::Mat img_out;
//    ColorLabels(img_labels_, img_out);
//    volwrite(filename, img_out);
//}


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

bool LabelingMapSingleton::Exists(const std::string& s)
{
    return LabelingMapSingleton::GetInstance().data_.end() != LabelingMapSingleton::GetInstance().data_.find(s);
}



KernelMapSingleton& KernelMapSingleton::GetInstance()
{
    static KernelMapSingleton instance;	    // Guaranteed to be destroyed.
                                            // Instantiated on first use.
    return instance;
}

const std::vector<std::string>& KernelMapSingleton::GetKernels(const std::string& s)
{
    return KernelMapSingleton::GetInstance().data_.at(s);
}

bool KernelMapSingleton::Exists(const std::string& s)
{
    return KernelMapSingleton::GetInstance().data_.end() != KernelMapSingleton::GetInstance().data_.find(s);
}

void KernelMapSingleton::InitializeAlgorithm(const std::string& algorithm, const std::string& kernels)
{
    std::vector<std::string> v;
    std::istringstream iss(kernels);
    std::string s;
    while (std::getline(iss, s, ',')) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
            }));
        v.push_back(s);
    }
    KernelMapSingleton::GetInstance().data_[algorithm] = move(v);
}



std::string Step(StepType n_step)
{
    switch (n_step) {
    case ALLOC_DEALLOC:
        return "Alloc Dealloc";
        break;
    case FIRST_SCAN:
        return "First Scan";
        break;
    case SECOND_SCAN:
        return "Second Scan";
        break;
    case ALL_SCANS:
        return "All Scans";
        break;
    case ST_SIZE: // To avoid warning on AppleClang
        break;
    }

    return "";
}