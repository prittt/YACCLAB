// Copyright (c) 2021, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_BIT_SCAN_FORWARD_H_
#define YACCLAB_BIT_SCAN_FORWARD_H_

#if defined _MSC_VER
#include <intrin.h>
#define YacclabBitScanForward64 _BitScanForward64
#else
#include <cstdint>
extern unsigned char YacclabBitScanForward64(unsigned long* Index, uint64_t Mask);
#endif

#endif // !YACCLAB_BIT_SCAN_FORWARD_H_