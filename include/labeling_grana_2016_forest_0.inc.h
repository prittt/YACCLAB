// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

	int c = -1;
tree_A0: if (++c >= COLS) goto break_A0;
    if (CONDITION_X) {
        // new label
        ACTION_2
        goto tree_B0;
    }
    else {
        // nothing
        ACTION_1
        goto tree_A0;
    }
tree_B0: if (++c >= COLS) goto break_B0;
    if (CONDITION_X) {
        // x <- s
        ACTION_6
        goto tree_B0;
    }
    else {
        // nothing
        ACTION_1
        goto tree_A0;
    }
break_A0:
break_B0:;