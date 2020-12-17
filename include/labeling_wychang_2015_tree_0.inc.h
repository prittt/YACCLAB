// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright (C) 2015 - Wan-Yu Chang and Chung-Cheng Chiu
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
// 
// The "free" of library only licensed for the purposes of research and study. 
// For further information or business co-operation, please contact us at
// Wan-Yu Chang and Chung-Cheng Chiu - Chung Cheng Institute of Technology of National Defense University.
// No.75, Shiyuan Rd., Daxi Township, Taoyuan County 33551, Taiwan (R.O.C.)  - e-mail: david.cc.chiu@gmail.com 
//
// Specially thank for the help of Prof. Grana who provide his source code of the BBDT algorithm.

            if (CONDITION_B1) {
                NEW_LABEL;
                if ((CONDITION_B2) || (CONDITION_B4))
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (CONDITION_B2) {
                NEW_LABEL;
                nextprocedure2 = true;
            }
            else if (CONDITION_B3) {
                NEW_LABEL;
                if (CONDITION_B4)
                    nextprocedure2 = true;
                else
                    nextprocedure2 = false;
            }
            else if (CONDITION_B4) {
                NEW_LABEL;
                nextprocedure2 = true;
            }
            else {
                nextprocedure2 = false;
            }

            while (nextprocedure2 && x + 2 < w) {
                x = x + 2;

                if (CONDITION_B1) {
                    ASSIGN_LX;
                    if ((CONDITION_B2) || (CONDITION_B4))
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;
                }
                else if (CONDITION_B2) {

                    if (CONDITION_B3) {
                        ASSIGN_LX;
                    }
                    else {
                        NEW_LABEL;
                    }
                    nextprocedure2 = true;
                }
                else if (CONDITION_B3) {
                    ASSIGN_LX;
                    if (CONDITION_B4)
                        nextprocedure2 = true;
                    else
                        nextprocedure2 = false;
                }
                else if (CONDITION_B4) {
                    NEW_LABEL;
                    nextprocedure2 = true;
                }
                else {
                    nextprocedure2 = false;
                }

            }