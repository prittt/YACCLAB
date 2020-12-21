// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

if (CONDITION_X) {
    if (CONDITION_Q) {
		if (CONDITION_S) {
			// x <- q + s
			ACTION_5
		}else{
			// x <- q
			ACTION_3
		}
    }
    else {
        // q = 0
        if (CONDITION_S) {
			// x <- s
			ACTION_4
		}
		else{
			// new label
			ACTION_2
		}
    }
}
else {
    // Nothing to do, x is a background pixel
    ACTION_1
}