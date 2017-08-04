// Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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
                if (CONDITION_B2 || CONDITION_B4)
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
                    if (CONDITION_B2 || CONDITION_B4)
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