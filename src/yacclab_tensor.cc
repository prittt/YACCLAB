#include "yacclab_tensor.h"

#include <map>
#include <vector>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "file_manager.h"

using namespace cv;

Mat1b YacclabTensorInput2D::mat_;
Mat YacclabTensorInput3D::mat_;

bool YacclabTensorInput2D::ReadBinary(const std::string &filename) {
    if (!filesystem::exists(filesystem::path(filename))) {
        return false;
    }

    Mat1b original_mat = imread(filename, IMREAD_GRAYSCALE);   // Read the file

    // Check if image exist
    if (original_mat.empty()) {
        return false;
    }

    // Adjust the threshold to make it binary
    threshold(original_mat, mat_, 100, 1, THRESH_BINARY);
    return true;
}

bool YacclabTensorInput3D::ReadBinary(const std::string &filename) {
    // Image load
    cv::Mat original_mat;
    bool is_dir;
    if (!filesystem::exists(filesystem::path(filename), is_dir))
        return false;
    if (!is_dir) {
        return false;
    }

    original_mat = volread(filename, IMREAD_GRAYSCALE);

    // Check if image exist
    if (original_mat.empty()) {
        return false;
    }

    // Adjust the threshold to make it binary
    // threshold(image, binary_mat, 100, 1, THRESH_BINARY);			
    mat_ = original_mat;
    return true;
}


bool YacclabTensorOutput::Equals(YacclabTensorOutput *other) {
    cv::Mat diff = GetMat() != other->GetMat();
    // Equal if no elements disagree
    return cv::countNonZero(diff) == 0;
}

void YacclabTensorOutput2D::NormalizeLabels() {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    for (int r = 0; r < mat_.rows; ++r) {
        unsigned * const mat_row = mat_.ptr<unsigned>(r);
        for (int c = 0; c < mat_.cols; ++c) {
            int iCurLabel = mat_row[c];
            if (iCurLabel > 0) {
                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                    map_new_labels[iCurLabel] = ++i_max_new_label;
                }
                mat_row[c] = map_new_labels.at(iCurLabel);
            }
        }
    }
}

void YacclabTensorOutput2D::WriteColored(const std::string &filename) const {
    cv::Mat3b mat_colored(mat_.size());
    for (int r = 0; r < mat_.rows; ++r) {
        unsigned const * const mat_row = mat_.ptr<unsigned>(r);
        Vec3b * const  mat_colored_row = mat_colored.ptr<Vec3b>(r);
        for (int c = 0; c < mat_.cols; ++c) {
            mat_colored_row[c] = Vec3b(mat_row[c] * 131 % 255, mat_row[c] * 241 % 255, mat_row[c] * 251 % 255);
        }
    }
    imwrite(filename, mat_colored);
}

void YacclabTensorOutput3D::NormalizeLabels() {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    if (mat_.dims == 3) {
        for (int z = 0; z < mat_.size[0]; z++) {
            for (int y = 0; y < mat_.size[1]; y++) {
                unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(mat_.data + z * mat_.step[0] + y * mat_.step[1]);
                for (int x = 0; x < mat_.size[2]; x++) {
                    int iCurLabel = img_labels_row[x];
                    if (iCurLabel > 0) {
                        if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                            map_new_labels[iCurLabel] = ++i_max_new_label;
                        }
                        img_labels_row[x] = map_new_labels.at(iCurLabel);
                    }
                }
            }
        }
    }
}

void YacclabTensorOutput3D::WriteColored(const std::string &filename) const {
    cv::Mat img_out(3, mat_.size.p, CV_8UC3);
    for (int z = 0; z < mat_.size[0]; z++) {
        for (int y = 0; y < mat_.size[1]; y++) {
			unsigned int const * const img_labels_row = mat_.ptr<unsigned int>(z, y);
            Vec3b * const img_out_row = img_out.ptr<Vec3b>(z, y);
            for (int x = 0; x < mat_.size[2]; x++) {
                img_out_row[x] = Vec3b(img_labels_row[x] * 131 % 255, img_labels_row[x] * 241 % 255, img_labels_row[x] * 251 % 255);
            }
        }
    }
    volwrite(filename, img_out);
}


#if defined USE_CUDA

cuda::GpuMat YacclabTensorInput2DCuda::d_mat_;

cuda::GpuMat3 YacclabTensorInput3DCuda::d_mat_;

bool YacclabTensorInput2DCuda::ReadBinary(const std::string &filename) {
    if (!YacclabTensorInput2D::ReadBinary(filename))
        return false;
    d_mat_.upload(YacclabTensorInput2D::mat_);
    return true;
}

bool YacclabTensorInput3DCuda::ReadBinary(const std::string &filename) {
    if (!YacclabTensorInput3D::ReadBinary(filename))
        return false;
    d_mat_.upload(YacclabTensorInput3D::mat_);
    return true;
}

#endif