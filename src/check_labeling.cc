#include "check_labeling.h"

LabelingCheckSingleton2D & LabelingCheckSingleton2D::GetInstance() {
    static LabelingCheckSingleton2D instance;
    return instance;
}

std::string LabelingCheckSingleton2D::GetCheckAlg(Connectivity2D conn) {
    return LabelingCheckSingleton2D::GetInstance().map_.at(conn);
}

LabelingCheckSingleton3D & LabelingCheckSingleton3D::GetInstance() {
    static LabelingCheckSingleton3D instance;
    return instance;
}

std::string LabelingCheckSingleton3D::GetCheckAlg(Connectivity3D conn) {
    return LabelingCheckSingleton3D::GetInstance().map_.at(conn);
}

namespace {
    class LabelingCheckAlgorithmsSet {
    public:
        LabelingCheckAlgorithmsSet() {
            // I do not have enough algorithms yet
            LabelingCheckSingleton2D::GetInstance().map_[CONN_4] = "SAUF_UF";
            LabelingCheckSingleton2D::GetInstance().map_[CONN_8] = "SAUF_UF";
            LabelingCheckSingleton3D::GetInstance().map_[CONN_6] = "CUDA_UF_3D";
            LabelingCheckSingleton3D::GetInstance().map_[CONN_18] = "CUDA_UF_3D";
            LabelingCheckSingleton3D::GetInstance().map_[CONN_26] = "CUDA_UF_3D";
        }
    } instance;
}