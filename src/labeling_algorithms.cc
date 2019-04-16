// Copyright(c) 2016 - 2019 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#include "labeling_algorithms.h"

#include "opencv2/imgcodecs.hpp"
#include "volume_util.h"

//#if defined USE_CUDA
//#include "cuda_runtime.h"
//#endif

//template <Connectivity Conn>
//cv::Mat1b Labeling2D<Conn>::img_;
//template <Connectivity Conn>
//cv::Mat Labeling3D<Conn>::img_;
//
//#if defined USE_CUDA
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

//#if defined USE_CUDA
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