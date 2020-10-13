// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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