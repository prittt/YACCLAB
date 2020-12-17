// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_CHECK_LABELING_H_
#define YACCLAB_CHECK_LABELING_H_

#include <map>
#include <string>

enum class Connectivity2D {
    CONN_4 = 4,
    CONN_8 = 8
};

enum class Connectivity3D {
    CONN_6 = 6,
    CONN_18 = 18,
    CONN_26 = 26
};

class LabelingCheckSingleton2D {
public:
    std::map<std::pair<Connectivity2D, bool>, std::string> map_;

    static LabelingCheckSingleton2D& GetInstance();
    static std::string GetCheckAlg(Connectivity2D conn, bool label_background);
    LabelingCheckSingleton2D(LabelingCheckSingleton2D const&) = delete;
    void operator=(LabelingCheckSingleton2D const&) = delete;

private:
    LabelingCheckSingleton2D() {}
    ~LabelingCheckSingleton2D() = default;
};

class LabelingCheckSingleton3D {
public:
    std::map<std::pair<Connectivity3D, bool>, std::string> map_;

    static LabelingCheckSingleton3D& GetInstance();
    static std::string GetCheckAlg(Connectivity3D conn, bool label_background);
    LabelingCheckSingleton3D(LabelingCheckSingleton3D const&) = delete;
    void operator=(LabelingCheckSingleton3D const&) = delete;

private:
    LabelingCheckSingleton3D() {}
    ~LabelingCheckSingleton3D() = default;
};


#endif //YACCLAB_CHECK_LABELING_H_