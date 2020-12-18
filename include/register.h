// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_REGISTER_H_
#define YACCLAB_REGISTER_H_

#define REGISTER_LABELING(algorithm)                                           \
class register_##algorithm {                                                   \
  public:                                                                      \
    register_##algorithm() {                                                   \
        LabelingMapSingleton::GetInstance().data_[#algorithm] = new algorithm; \
    }                                                                          \
} reg_##algorithm;

//#define REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(algorithm) is defined in xxx.h

#define STRINGIFY(x) #x
#define CONCAT(x,y) STRINGIFY(x ## _ ## y)

#define REGISTER_SOLVER(algorithm, solver)                                                              \
class register_ ## algorithm ## _ ## solver{                                                            \
  public:                                                                                               \
      register_ ## algorithm ## _ ## solver() {                                                         \
          LabelingMapSingleton::GetInstance().data_[CONCAT(algorithm, solver)] = new algorithm<solver>; \
  }                                                                                                     \
}  register_ ## algorithm ## _ ## solver;

#endif // !YACCLAB_REGISTER_H_