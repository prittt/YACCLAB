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