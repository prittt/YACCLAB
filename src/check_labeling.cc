#include "check_labeling.h"

using namespace std;

LabelingCheckSingleton2D& LabelingCheckSingleton2D::GetInstance() {
    static LabelingCheckSingleton2D instance;
    return instance;
}

std::string LabelingCheckSingleton2D::GetCheckAlg(Connectivity2D conn, bool label_background) {
    return LabelingCheckSingleton2D::GetInstance().map_.at(make_pair(conn, label_background));
}

LabelingCheckSingleton3D& LabelingCheckSingleton3D::GetInstance() {
    static LabelingCheckSingleton3D instance;
    return instance;
}

std::string LabelingCheckSingleton3D::GetCheckAlg(Connectivity3D conn, bool label_background) {
    return LabelingCheckSingleton3D::GetInstance().map_.at(make_pair(conn, label_background));
}

namespace {
    class LabelingCheckAlgorithmsSet {
    public:
        LabelingCheckAlgorithmsSet() {
            // LabelingCheckSingleton2D::GetInstance().map_[make_pair(Connectivity2D::CONN_4, false)] = "SAUF4C_UF";
            LabelingCheckSingleton2D::GetInstance().map_[make_pair(Connectivity2D::CONN_8, false)] = "SAUF_UF";
            // LabelingCheckSingleton2D::GetInstance().map_[make_pair(Connectivity2D::CONN_8, true)] = "SAUF_BG_UF";
            LabelingCheckSingleton3D::GetInstance().map_[make_pair(Connectivity3D::CONN_6, false)] = "naive_3D_UF";
            LabelingCheckSingleton3D::GetInstance().map_[make_pair(Connectivity3D::CONN_18, false)] = "naive_3D_UF";
            LabelingCheckSingleton3D::GetInstance().map_[make_pair(Connectivity3D::CONN_26, false)] = "naive_3D_UF";
        }
    } instance;
}