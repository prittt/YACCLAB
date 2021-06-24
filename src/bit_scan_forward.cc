// Copyright (c) 2021, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "bit_scan_forward.h"

#if __cplusplus > 201703L
#include <version>
#endif

#if defined _MSC_VER
#elif defined MY_PLATFORM_MACRO
// Define YacclabBitScanForward64 using the proper compiler intrinsic for your platform.
// If it's just a #define, put it in the .h instead.
// Don't forget to open a pull request! :)
#elif defined __cpp_lib_bitops
#include <bit>
unsigned char YacclabBitScanForward64(unsigned long* Index, unsigned __int64 Mask) {
    int count = std::countr_zero(Mask);
    if (count == 64)
        return 0;

    *Index = static_cast<unsigned long>(count);
    return 1;
}
#else
#include <stdexcept>
unsigned char YacclabBitScanForward64(unsigned long* Index, unsigned __int64 Mask) {
    throw std::runtime_error("YacclabBitScanForward64 undefined."
        " Define it for your platform in \"bit_scan_forward.cc\","
        " or enable C++20 for the generic version.");
}
#endif