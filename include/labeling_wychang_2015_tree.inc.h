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
                    if (CONDITION_B2) {
                        if (CONDITION_U2) {
                            lx = ASSIGN_Q;
                            if (CONDITION_U3) {

                            }
                            else {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_RV;
                                    RESOLVE_2(u, v);
                                }
                            }
                        }
                        else if (CONDITION_U3) {
                            lx = ASSIGN_Q;
                            if (CONDITION_U1) {
                                LOAD_LX;
                                LOAD_PV;
                                RESOLVE_2(u, v);
                            }

                        }
                        else if (CONDITION_U1) {
                            lx = ASSIGN_P;
                            if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                        }
                        else if (CONDITION_U4) {
                            lx = ASSIGN_R;
                        }
                        else {
                            NEW_LABEL;
                        }
                        nextprocedure2 = true;
                    }
                    else {
                        if (CONDITION_U2) {
                            lx = ASSIGN_Q;
                        }
                        else if (CONDITION_U1) {
                            lx = ASSIGN_P;
                            if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);

                            }
                        }
                        else if (CONDITION_U3) {
                            lx = ASSIGN_Q;
                        }
                        else {
                            NEW_LABEL;
                        }
                        if (CONDITION_B4)
                            nextprocedure2 = true;
                        else
                            nextprocedure2 = false;

                    }
                }
                else if (CONDITION_B2) {
                    if (CONDITION_U3) {
                        lx = ASSIGN_Q;
                    }
                    else if (CONDITION_U2) {
                        lx = ASSIGN_Q;
                        if (CONDITION_U4) {
                            LOAD_LX;
                            LOAD_RV;
                            RESOLVE_2(u, v);
                        }
                    }
                    else if (CONDITION_U4) {
                        lx = ASSIGN_R;
                    }
                    else {
                        NEW_LABEL;
                    }
                    nextprocedure2 = true;
                }
                else if (CONDITION_B3) {
                    NEW_LABEL;
                    if (CONDITION_B4)
                        nextprocedure2 = true;//
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
                        if (CONDITION_B2) {
                            if (CONDITION_U2) {
                                if (CONDITION_U3) {
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                                else {
                                    if (CONDITION_U4) {
                                        LOAD_LX;
                                        LOAD_QV;
                                        LOAD_RK;
                                        RESOLVE_3(u, v, k);
                                    }
                                    else {
                                        LOAD_LX;
                                        LOAD_QV;
                                        RESOLVE_2(u, v);
                                    }
                                }

                            }
                            else if (CONDITION_U3) {
                                if (CONDITION_U1) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_QK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    // Reslove S, Q
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U1) {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_RK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_PV;
                                    RESOLVE_2(u, v);
                                }

                            }
                            else if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                            nextprocedure2 = true;
                        }
                        else {
                            ASSIGN_LX;
                            if (CONDITION_U2) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }
                            else if (CONDITION_U1) {
                                if (CONDITION_U3) {
                                    LOAD_LX;
                                    LOAD_PV;
                                    LOAD_QK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_PV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }

                            if (CONDITION_B4)
                                nextprocedure2 = true;//
                            else
                                nextprocedure2 = false;
                        }

                    }
                    else if (CONDITION_B2) {
                        if (CONDITION_B3) {
                            ASSIGN_LX;
                            if (CONDITION_U3) {
                                LOAD_LX;
                                LOAD_QV;
                                RESOLVE_2(u, v);
                            }
                            else if (CONDITION_U2) {
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_QV;
                                    LOAD_RK;
                                    RESOLVE_3(u, v, k);
                                }
                                else {
                                    LOAD_LX;
                                    LOAD_QV;
                                    RESOLVE_2(u, v);
                                }
                            }if (CONDITION_U4) {
                                LOAD_LX;
                                LOAD_RV;
                                RESOLVE_2(u, v);
                            }
                        }
                        else {
                            if (CONDITION_U3) {
                                lx = ASSIGN_Q;
                            }
                            else if (CONDITION_U2) {
                                lx = ASSIGN_Q;
                                if (CONDITION_U4) {
                                    LOAD_LX;
                                    LOAD_RV;
                                    RESOLVE_2(u, v);
                                }
                            }
                            else if (CONDITION_U4) {
                                lx = ASSIGN_R;
                            }
                            else {
                                NEW_LABEL;
                            }
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
