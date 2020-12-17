// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling_lacassagne_2016.h"

#include "register.h"

// Light-Speed Labeling is an algorithm that can only use some types of equivalences solver, so they must be "manually" specified
REGISTER_SOLVER(LSL_STD, UF)
REGISTER_SOLVER(LSL_STD, TTA)

REGISTER_SOLVER(LSL_STDZ, UF)
REGISTER_SOLVER(LSL_STDZ, TTA)

REGISTER_SOLVER(LSL_RLE, UF)
REGISTER_SOLVER(LSL_RLE, TTA)

REGISTER_SOLVER(LSL_RLEZ, UF)
REGISTER_SOLVER(LSL_RLEZ, TTA)
