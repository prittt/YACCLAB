// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

if (CONDITION_X) {
    if (CONDITION_Q) {
        //x <- q
        ACTION_4
    }
    else {
        // q = 0
        if (CONDITION_R) {
            if (CONDITION_P) {
                // x <- p + r
                ACTION_7
            }
            else {
                // p = q = 0
                if (CONDITION_S) {
                    // x <- s + r
                    ACTION_8
                }
                else {
                    // p = q = s = 0
                    // x <- r
                    ACTION_5
                }
            }
        }
        else {
            // r = q = 0
            if (CONDITION_P) {
                // x <- p
                ACTION_3
            }
            else {
                // r = q = p = 0
                if (CONDITION_S) {
                    // x <- s
                    ACTION_6
                }
                else {
                    // New label
                    ACTION_2
                }
            }
        }
    }
}
else {
    if (CONDITION_Q_BG) {
		if (CONDITION_S_BG) {
			// x <- q + s
			ACTION_9
		}
        else {
			// x <- q
			ACTION_4
		}
    }
    else {
        // q = 1
        if (CONDITION_S_BG) {
			// x <- s
			ACTION_6
		}
		else {
			// new label
			ACTION_2
		}
    }
}