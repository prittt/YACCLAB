// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

	int c = 0; // First column
/*tree_0:*/ if (c == COLS - 1) goto break_0;
    if (CONDITION_X) {
        if (CONDITION_Q) {
			// x <- q
            ACTION_4 
            goto tree_A;
        }
        else {
            if (CONDITION_R) {
                // x <- r
                ACTION_5
				goto tree_B;
            }
            else {
                // new label
				ACTION_2
                goto tree_C;
            }
        }
    }
    else {
        // nothing
        goto tree_D;
    }

tree_A: if (++c >= COLS - 1) goto break_A;
    if (CONDITION_X) {
        if (CONDITION_Q) {
            // x <- q
            ACTION_4
            goto tree_A;
        }
        else {
            if (CONDITION_R) {
                // x <- s + r
				ACTION_8
                goto tree_B;
            }
            else {
                // x <- s
				ACTION_6
                goto tree_C;
            }
        }
    }
    else {
        // nothing
        goto tree_D;
    }
tree_B: if (++c >= COLS - 1) goto break_B;
    if (CONDITION_X) {
        // x <- q
        ACTION_4
        goto tree_A;
    }
    else {
        // nothing
        goto tree_D;
    }
tree_C: if (++c >= COLS - 1) goto break_C;
    if (CONDITION_X) {
        if (CONDITION_R) {
            // x <- s + r
			ACTION_8
            goto tree_B;
        }
        else {
            // x <- s
			ACTION_6
            goto tree_C;
        }
    }
    else {
        // nothing
        goto tree_D;
    }
tree_D: if (++c >= COLS - 1) goto break_D;
    if (CONDITION_X) {
        if (CONDITION_Q) {
            // x <- q
            ACTION_4
            goto tree_A;
        }
        else {
            if (CONDITION_R) {
                if (CONDITION_P) {
                    // x <- p + r
					ACTION_7
                    goto tree_B;
                }
                else {
                    // x <- r
					ACTION_5
                    goto tree_B;
                }
            }
            else {
                if (CONDITION_P) {
                    // x <- p
					ACTION_3
                    goto tree_C;
                }
                else {
                    // new label
					ACTION_2
                    goto tree_C;
                }
            }
        }
    }
    else {
        // nothing
        goto tree_D;
    }


    // Last column
break_A:
    if (CONDITION_X) {
        if (CONDITION_Q) {
            // x <- q
            ACTION_4
        }
        else {
            // x <- s
			ACTION_6
        }
    }
    continue;
break_B:
    if (CONDITION_X) {
        // x <- q
        ACTION_4
    }
    continue;
break_C:
    if (CONDITION_X) {
        // x <- s
		ACTION_6
    }
    continue;
break_D:
    if (CONDITION_X) {
        if (CONDITION_Q) {
            // x <- q
            ACTION_4
        }
        else {
            if (CONDITION_P) {
                // x <- p
				ACTION_3
            }
            else {
                // new label
				ACTION_2
            }
        }
    }
    continue;
break_0:
    // This tree is necessary to handle one column vector images
    if (CONDITION_X) {
        if (CONDITION_Q) {
            // x <- q
            ACTION_4
        }
        else {
            // new label
            ACTION_2
        }
    }