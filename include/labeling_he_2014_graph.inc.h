// Copyright(c) 2016 - 2019 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

switch (prev_state) {
case(CA):// Previous configuration was CA
    if (CONDITION_A) {
        prev_state = CB;
        if (CONDITION_N2) {
            // a <- n2
            ACTION_4
            // Probably follows state are CD, CG and CA. We choose CD as "delegate"
            prob_fol_state = CD; 
        }
        else {
            if (CONDITION_N3) {
                // a <- n3
                ACTION_5
                // Probably follows state are CE, CG and CA. We choose CE as "delegate"
                prob_fol_state = CE; 
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1 
                    ACTION_8
                }
            }
            else {
                // Probably follows state are CF, CG and CA. We choose CE as "delegate"
                prob_fol_state = CF;  
                if (CONDITION_N1) {
                    // a <- n1
                    ACTION_3
                }
                else {
                    // a new label
                    ACTION_2
                }
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // b new label
            ACTION_11
            prev_state = CC;
        }
        else {
            // nothing
            ACTION_1
        }
    }
    break;
case(CB): // Previous configuration was CB
    if (CONDITION_A) {
        // All possible configuration are CD, CE, CF
        if (prob_fol_state == CD) {
            // Current configuration is CD
            prev_state = CD;
            ACTION_6 // a <- n4
            if (CONDITION_N2) {
                // a <- n4 (already done)
                prob_fol_state = CD;
            }
            else {
                if (CONDITION_N3) {
                    // Solve equivalence between a and n3
                    ACTION_9
                    prob_fol_state = CE;
                }
                else {
                    // a <- n4 (already done)
                    prob_fol_state = CF;
                }
            }
        }
        else {
            if (prob_fol_state == CE) {
                // Current configuration is CE
                // a <- n4
                ACTION_6
                prev_state = CE;
            }
            else {
                if (prob_fol_state == CF) {
                    // Current configuration is CF
                    // a <- n4
                    ACTION_6
                    if (CONDITION_N3) {
                        // Solve equivalence between a and n3 
                        ACTION_9
                        prob_fol_state = CE;
                    }
                    else {
                        // a <- n4 (already done)
                        prob_fol_state = CF;
                    }
                }
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CG
            // b <- n4
            ACTION_12
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CC): // Previous configuration was CC
    if (CONDITION_A) {
        // Current configuration is CH
        prev_state = CH;
        if (CONDITION_N2) {
            // a <- n2
            ACTION_4
            // Solve equivalence between a and n5
            ACTION_10
            prob_fol_state = CD;
        }
        else {
            if (CONDITION_N3) {
                // a <- n3
                ACTION_5
                // Solve equivalence between a and n5
                ACTION_10
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1
                    ACTION_8
                }
                prob_fol_state = CE;
            }
            else {
                // a <- n5
                ACTION_7
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1
                    ACTION_8
                }
                prob_fol_state = CF;
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CI
            // b <- n5 
            ACTION_13
            prev_state = CI;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CD): // Previous configuration was CD
    if (CONDITION_A) {
        // All possible configuration are CD, CE, CF
        if (prob_fol_state == CD) {
            // Current configuration is CD
            prev_state = CD;
            // a <- n4 (in all cases)
            ACTION_6
            if (CONDITION_N2) {
                // a <- n4 (already done)
                prob_fol_state = CD;
            }
            else {
                if (CONDITION_N3) {
                    // a <- n4 (already done)
                    // Solve equivalence between a and n3
                    ACTION_9
                    prob_fol_state = CE;
                }
                else {
                    // a <- n4 (already done)
                    prob_fol_state = CF;
                }
            }
        }
        else {
            if (prob_fol_state == CE) {
                // Current configuration is CE
                // a <- n4
                ACTION_6
                prev_state = CE;
            }
            else {
                if (prob_fol_state == CF) {
                    // Current configuration is CF
                    // a <- n4
                    ACTION_6
                    if (CONDITION_N3) {
                        // Solve equivalence between a and n3
                        ACTION_9
                        prob_fol_state = CE;
                    }
                    else {
                        // a <- n4 (already done)
                        prob_fol_state = CF;
                    }
                }
            }
        }
        if (CONDITION_B) {
            // b <- a
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CG
            // b <- n4 
            ACTION_12
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CE): // Previous configuration was CE
    if (CONDITION_A) {
        // Current configuration is CD
        prev_state = CD;
        // a <- n4 (in all cases)
        ACTION_6
        if (CONDITION_N2) {
            // a <- n4 (already done)
            prob_fol_state = CD;
        }
        else {
            if (CONDITION_N3) {
                // a <- n4 (already done)
                // Solve equivalence between a and n3
                ACTION_9
                prob_fol_state = CE;
            }
            else {
                // a <- n4 (already done)
                prob_fol_state = CF;
            }
        }
        if (CONDITION_B) {
            // b <- a
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CG
            // b <- n4
            ACTION_12
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CF): // Previous configuration was CF
    if (CONDITION_A) {
        // Possible current configuration are CE and CF
        if (prob_fol_state == CE) {
            // Current configuration is CE
            // a <- n4
            ACTION_6
            prev_state = CE;
        }
        else {
            if (prob_fol_state == CF) {
                // Current configuration is CF
                // a <- n4
                ACTION_6
                if (CONDITION_N3) {
                    // Solve equivalence between a and n3
                    ACTION_9
                    prob_fol_state = CE;
                }
                else {
                    // a <- n4 (already done)
                    prob_fol_state = CF;
                }
            }
        }
        if (CONDITION_B) {
            // b <- a
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CG
            // b <- n4
            ACTION_12
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CG): // Previous configuration was CG
    if (CONDITION_A) {
        // Current state is CH
        prev_state = CH;
        if (CONDITION_N2) {
            // a <- n2
            ACTION_4
            // Solve equivalence between a and n5
            ACTION_10
            prob_fol_state = CD;
        }
        else {
            if (CONDITION_N3) {
                // a <- n3
                ACTION_5
                // Solve equivalence between a and n5
                ACTION_10
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1
                    ACTION_8
                }
                prob_fol_state = CE;
            }
            else {
                // a <- n5
                ACTION_7
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1
                    ACTION_8
                }
                prob_fol_state = CF;
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            //Current configuration in CI
            // b <- n5
            ACTION_13
            prev_state = CI;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CH): // Previous configuration was CH
    if (CONDITION_A) {
        // All possible configuration are CD, CE, CF
        if (prob_fol_state == CD) {
            // Current configuration is CD
            prev_state = CD;
            // a <- n4 (in all cases)
            ACTION_6 
            if (CONDITION_N2) {
                // a <- n4 (already done)
                prob_fol_state = CD;
            }
            else {
                if (CONDITION_N3) {
                    // a <- n4 (already done)
                    // Solve equivalence between a and n3
                    ACTION_9
                    prob_fol_state = CE;
                }
                else {
                    // a <- n4 (already done)
                    prob_fol_state = CF;
                }
            }
        }
        else {
            if (prob_fol_state == CE) {
                // Current configuration is CE
                // a <- n4
                ACTION_6
                prev_state = CE;
            }
            else {
                if (prob_fol_state == CF) {
                    // Current configuration is CF
                    // a <- n4 (in all cases)
                    ACTION_6
                    if (CONDITION_N3) {
                        // Solve equivalence between a and n3
                        ACTION_9
                        prob_fol_state = CE;
                    }
                    else {
                        // a <- n4 (already done)
                        prob_fol_state = CF;
                    }
                }
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CG
            // b <- n4
            ACTION_12
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(CI): // Previous configuration was CI
    if (CONDITION_A) {
        // Current configuration is CH
        prev_state = CH;
        if (CONDITION_N2) {
            // a <- n2
            ACTION_4
            // Solve equivalence between a and n5
            ACTION_10
            prob_fol_state = CD;
        }
        else {
            if (CONDITION_N3) {
                // a <- n3
                ACTION_5
                // Solve equivalence between a and n5
                ACTION_10
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1
                    ACTION_8
                }
                prob_fol_state = CE;
            }
            else {
                // a <- n5
                ACTION_7
                if (CONDITION_N1) {
                    // Solve equivalence between a and n1 
                    ACTION_8
                }
                prob_fol_state = CF;
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // Current configuration is CI
            // b <- n5
            ACTION_13
            prev_state = CI;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
case(BR): // No previous configuration defined (new line)
    if (CONDITION_A) {
        //a is foreground pixel
        prev_state = CB;
        if (CONDITION_N2) {
            // a <- n2
            ACTION_4
            prob_fol_state = CD;
        }
        else {
            if (CONDITION_N3) {
                // a <- n3
                ACTION_5
                prob_fol_state = CE;
            }
            else {
                // a new label
                ACTION_2
                prob_fol_state = CF;
            }
        }
        if (CONDITION_B) {
            // b <- a 
            ACTION_14
        }
    }
    else {
        if (CONDITION_B) {
            // b new label 
            ACTION_11
            prev_state = CG;
        }
        else {
            // Current configuration is CA
            // nothing 
            ACTION_1
            prev_state = CA;
        }
    }
    break;
}//End switch