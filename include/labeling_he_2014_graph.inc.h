// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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