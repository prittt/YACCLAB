// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#include "labelingGrana2010.h"

using namespace cv;
using namespace std;

inline static
void firstScanBBDT(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

	for (int r = 0; r<h; r += 2) {
		for (int c = 0; c < w; c += 2) {

			// We work with 2x2 blocks
			// +-+-+-+
			// |P|Q|R|
			// +-+-+-+
			// |S|X|
			// +-+-+

			// The pixels are named as follows
			// +---+---+---+
			// |a b|c d|e f|
			// |g h|i j|k l|
			// +---+---+---+
			// |m n|o p|
			// |q r|s t|
			// +---+---+

			// Pixels a, f, l, q are not needed, since we need to understand the 
			// the connectivity between these blocks and those pixels only metter
			// when considering the outer connectivities

			// A bunch of defines used to check if the pixels are foreground, 
			// without going outside the image limits.

#define condition_b c-1>=0 && r-2>=0 && img(r-2, c-1)>0
#define condition_c r-2>=0 && img(r-2, c)>0
#define condition_d c+1<w && r-2>=0 && img(r-2, c+1)>0
#define condition_e c+2<w && r-2>=0 && img(r-2, c+2)>0

#define condition_g c-2>=0 && r-1>=0 && img(r-1, c-2)>0
#define condition_h c-1>=0 && r-1>=0 && img(r-1, c-1)>0
#define condition_i r-1>=0 && img(r-1, c)>0
#define condition_j c+1<w && r-1>=0 && img(r-1, c+1)>0
#define condition_k c+2<w && r-1>=0 && img(r-1, c+2)>0

#define condition_m c-2>=0 && img(r, c-2)>0
#define condition_n c-1>=0 && img(r, c-1)>0
#define condition_o img(r,c)>0
#define condition_p c+1<w && img(r,c+1)>0

#define condition_r c-1>=0 && r+1<h && img(r+1, c-1)>0
#define condition_s r+1<h && img(r+1, c)>0
#define condition_t c+1<w && r+1<h && img(r+1, c+1)>0

			// This is a decision tree which allows to choose which action to 
			// perform, checking as few conditions as possible.
			// Actions are available after the tree.

			if (condition_o) {
				if (condition_n) {
					if (condition_j) {
						if (condition_i) {
							goto action_6;
						}
						else {
							if (condition_c) {
								if (condition_h) {
									goto action_6;
								}
								else {
									if (condition_g) {
										if (condition_b) {
											goto action_6;
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								goto action_11;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										goto action_6;
									}
									else {
										if (condition_c) {
											if (condition_h) {
												goto action_6;
											}
											else {
												if (condition_g) {
													if (condition_b) {
														goto action_6;
													}
													else {
														goto action_12;
													}
												}
												else {
													goto action_12;
												}
											}
										}
										else {
											goto action_12;
										}
									}
								}
								else {
									goto action_12;
								}
							}
							else {
								goto action_6;
							}
						}
						else {
							goto action_6;
						}
					}
				}
				else {
					if (condition_r) {
						if (condition_j) {
							if (condition_m) {
								if (condition_h) {
									if (condition_i) {
										goto action_6;
									}
									else {
										if (condition_c) {
											goto action_6;
										}
										else {
											goto action_11;
										}
									}
								}
								else {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												goto action_6;
											}
											else {
												if (condition_c) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								if (condition_i) {
									goto action_11;
								}
								else {
									if (condition_h) {
										if (condition_c) {
											goto action_11;
										}
										else {
											goto action_14;
										}
									}
									else {
										goto action_11;
									}
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_m) {
										if (condition_h) {
											if (condition_d) {
												if (condition_i) {
													goto action_6;
												}
												else {
													if (condition_c) {
														goto action_6;
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
										else {
											if (condition_d) {
												if (condition_g) {
													if (condition_b) {
														if (condition_i) {
															goto action_6;
														}
														else {
															if (condition_c) {
																goto action_6;
															}
															else {
																goto action_12;
															}
														}
													}
													else {
														goto action_12;
													}
												}
												else {
													goto action_12;
												}
											}
											else {
												if (condition_i) {
													if (condition_g) {
														if (condition_b) {
															goto action_12;
														}
														else {
															goto action_16;
														}
													}
													else {
														goto action_16;
													}
												}
												else {
													goto action_12;
												}
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_d) {
												goto action_12;
											}
											else {
												goto action_16;
											}
										}
										else {
											if (condition_h) {
												if (condition_d) {
													if (condition_c) {
														goto action_12;
													}
													else {
														goto action_15;
													}
												}
												else {
													goto action_15;
												}
											}
											else {
												goto action_12;
											}
										}
									}
								}
								else {
									if (condition_h) {
										if (condition_m) {
											goto action_6;
										}
										else {
											goto action_9;
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_g) {
													if (condition_b) {
														goto action_6;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_h) {
									if (condition_m) {
										goto action_6;
									}
									else {
										goto action_9;
									}
								}
								else {
									if (condition_i) {
										if (condition_m) {
											if (condition_g) {
												if (condition_b) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_6;
									}
								}
							}
						}
					}
					else {
						if (condition_j) {
							if (condition_i) {
								goto action_4;
							}
							else {
								if (condition_h) {
									if (condition_c) {
										goto action_4;
									}
									else {
										goto action_7;
									}
								}
								else {
									goto action_4;
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											goto action_5;
										}
										else {
											goto action_10;
										}
									}
									else {
										if (condition_h) {
											if (condition_d) {
												if (condition_c) {
													goto action_5;
												}
												else {
													goto action_8;
												}
											}
											else {
												goto action_8;
											}
										}
										else {
											goto action_5;
										}
									}
								}
								else {
									if (condition_i) {
										goto action_4;
									}
									else {
										if (condition_h) {
											goto action_3;
										}
										else {
											goto action_2;
										}
									}
								}
							}
							else {
								if (condition_i) {
									goto action_4;
								}
								else {
									if (condition_h) {
										goto action_3;
									}
									else {
										goto action_2;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (condition_s) {
					if (condition_p) {
						if (condition_n) {
							if (condition_j) {
								if (condition_i) {
									goto action_6;
								}
								else {
									if (condition_c) {
										if (condition_h) {
											goto action_6;
										}
										else {
											if (condition_g) {
												if (condition_b) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											goto action_6;
										}
										else {
											if (condition_c) {
												if (condition_h) {
													goto action_6;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															goto action_6;
														}
														else {
															goto action_12;
														}
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
									}
									else {
										goto action_12;
									}
								}
								else {
									goto action_6;
								}
							}
						}
						else {
							if (condition_r) {
								if (condition_j) {
									if (condition_m) {
										if (condition_h) {
											if (condition_i) {
												goto action_6;
											}
											else {
												if (condition_c) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
										}
										else {
											if (condition_g) {
												if (condition_b) {
													if (condition_i) {
														goto action_6;
													}
													else {
														if (condition_c) {
															goto action_6;
														}
														else {
															goto action_11;
														}
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
									}
									else {
										goto action_11;
									}
								}
								else {
									if (condition_k) {
										if (condition_d) {
											if (condition_m) {
												if (condition_h) {
													if (condition_i) {
														goto action_6;
													}
													else {
														if (condition_c) {
															goto action_6;
														}
														else {
															goto action_12;
														}
													}
												}
												else {
													if (condition_g) {
														if (condition_b) {
															if (condition_i) {
																goto action_6;
															}
															else {
																if (condition_c) {
																	goto action_6;
																}
																else {
																	goto action_12;
																}
															}
														}
														else {
															goto action_12;
														}
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
										else {
											if (condition_i) {
												if (condition_m) {
													if (condition_h) {
														goto action_12;
													}
													else {
														if (condition_g) {
															if (condition_b) {
																goto action_12;
															}
															else {
																goto action_16;
															}
														}
														else {
															goto action_16;
														}
													}
												}
												else {
													goto action_16;
												}
											}
											else {
												goto action_12;
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_h) {
													goto action_6;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															goto action_6;
														}
														else {
															goto action_11;
														}
													}
													else {
														goto action_11;
													}
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_j) {
									goto action_4;
								}
								else {
									if (condition_k) {
										if (condition_i) {
											if (condition_d) {
												goto action_5;
											}
											else {
												goto action_10;
											}
										}
										else {
											goto action_5;
										}
									}
									else {
										if (condition_i) {
											goto action_4;
										}
										else {
											goto action_2;
										}
									}
								}
							}
						}
					}
					else {
						if (condition_r) {
							goto action_6;
						}
						else {
							if (condition_n) {
								goto action_6;
							}
							else {
								goto action_2;
							}
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_j) {
							goto action_4;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										goto action_5;
									}
									else {
										goto action_10;
									}
								}
								else {
									goto action_5;
								}
							}
							else {
								if (condition_i) {
									goto action_4;
								}
								else {
									goto action_2;
								}
							}
						}
					}
					else {
						if (condition_t) {
							goto action_2;
						}
						else {
							goto action_1;
						}
					}
				}
			}

			// Actions: the blocks label are provisionally stored in the top left 
			// pixel of the block in the labels image

		action_1:	//Action_1: No action (the block has no foreground pixels)
			imgLabels(r, c) = 0;
			continue;
		action_2:	//Action_2: New label (the block has foreground pixels and is not connected to anything else)
			imgLabels(r, c) = lunique;
			P[lunique] = lunique;
			lunique = lunique + 1;
			continue;
		action_3:	//Action_3: Assign label of block P
			imgLabels(r, c) = imgLabels(r - 2, c - 2);
			continue;
		action_4:	//Action_4: Assign label of block Q 
			imgLabels(r, c) = imgLabels(r - 2, c);
			continue;
		action_5:	//Action_5: Assign label of block R
			imgLabels(r, c) = imgLabels(r - 2, c + 2);
			continue;
		action_6:	//Action_6: Assign label of block S
			imgLabels(r, c) = imgLabels(r, c - 2);
			continue;
		action_7:	//Action_7: Merge labels of block P and Q
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c));
			continue;
		action_8:	//Action_8: Merge labels of block P and R
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c + 2));
			continue;
		action_9:	// Action_9 Merge labels of block P and S
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r, c - 2));
			continue;
		action_10:	// Action_10 Merge labels of block Q and R
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r - 2, c + 2));
			continue;
		action_11:	//Action_11: Merge labels of block Q and S
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r, c - 2));
			continue;
		action_12:	//Action_12: Merge labels of block R and S
			imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c + 2), (uint)imgLabels(r, c - 2));
			continue;
		action_14:	//Action_14: Merge labels of block P, Q and S
			imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c)), (uint)imgLabels(r, c - 2));
			continue;
		action_15:	//Action_15: Merge labels of block P, R and S
			imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c + 2)), (uint)imgLabels(r, c - 2));
			continue;
		action_16:	//Action_16: labels of block Q, R and S
			imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r - 2, c + 2)), (uint)imgLabels(r, c - 2));
			continue;
		}
	}

#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e

#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k

#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p

#undef condition_r
#undef condition_s
#undef condition_t

}

int BBDT(const Mat1b &img, Mat1i &imgLabels) {

	imgLabels = cv::Mat1i(img.size());

	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;

	//Tree of labels
	vector<uint> P(Plength);

	//Background
	P[0] = 0;
	uint lunique = 1;

	firstScanBBDT(img, imgLabels, P.data(), lunique);

	uint nLabel = flattenL(P.data(), lunique);

	//Second scan
	for (int r = 0; r < imgLabels.rows; r += 2) {
		for (int c = 0; c < imgLabels.cols; c += 2) {
			int iLabel = imgLabels(r, c);
			if (iLabel>0) {
				iLabel = P[iLabel];
				if (img(r, c) > 0)
					imgLabels(r, c) = iLabel;
				else
					imgLabels(r, c) = 0;
				if (c + 1 < img.cols) {
					if (img(r, c + 1) > 0)
						imgLabels(r, c + 1) = iLabel;
					else
						imgLabels(r, c + 1) = 0;
					if (r + 1 < img.rows) {
						if (img(r + 1, c) > 0)
							imgLabels(r + 1, c) = iLabel;
						else
							imgLabels(r + 1, c) = 0;
						if (img(r + 1, c + 1) > 0)
							imgLabels(r + 1, c + 1) = iLabel;
						else
							imgLabels(r + 1, c + 1) = 0;
					}
				}
				else if (r + 1 < img.rows) {
					if (img(r + 1, c) > 0)
						imgLabels(r + 1, c) = iLabel;
					else
						imgLabels(r + 1, c) = 0;
				}
			}
			else {
				imgLabels(r, c) = 0;
				if (c + 1 < img.cols) {
					imgLabels(r, c + 1) = 0;
					if (r + 1 < img.rows) {
						imgLabels(r + 1, c) = 0;
						imgLabels(r + 1, c + 1) = 0;
					}
				}
				else if (r + 1 < img.rows) {
					imgLabels(r + 1, c) = 0;
				}
			}
		}
	}

	return nLabel;
}

inline static
void firstScanBBDT_OPT(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

	for (int r = 0; r<h; r += 2) {
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		const uchar* const img_row_prev_prev = (uchar *)(((char *)img_row_prev) - img.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
		for (int c = 0; c < w; c += 2) {

			// We work with 2x2 blocks
			// +-+-+-+
			// |P|Q|R|
			// +-+-+-+
			// |S|X|
			// +-+-+

			// The pixels are named as follows
			// +---+---+---+
			// |a b|c d|e f|
			// |g h|i j|k l|
			// +---+---+---+
			// |m n|o p|
			// |q r|s t|
			// +---+---+

			// Pixels a, f, l, q are not needed, since we need to understand the 
			// the connectivity between these blocks and those pixels only metter
			// when considering the outer connectivities

			// A bunch of defines used to check if the pixels are foreground, 
			// without going outside the image limits.

#define condition_b c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define condition_c r-2>=0 && img_row_prev_prev[c]>0
#define condition_d c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define condition_e c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define condition_g c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define condition_h c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define condition_i r-1>=0 && img_row_prev[c]>0
#define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define condition_m c-2>=0 && img_row[c-2]>0
#define condition_n c-1>=0 && img_row[c-1]>0
#define condition_o img_row[c]>0
#define condition_p c+1<w && img_row[c+1]>0

#define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define condition_s r+1<h && img_row_fol[c]>0
#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0

			// This is a decision tree which allows to choose which action to 
			// perform, checking as few conditions as possible.
			// Actions are available after the tree.

			if (condition_o) {
				if (condition_n) {
					if (condition_j) {
						if (condition_i) {
							//Action_6: Assign label of block S
							imgLabels_row[c] = imgLabels_row[c - 2];
							continue;
						}
						else {
							if (condition_c) {
								if (condition_h) {
									//Action_6: Assign label of block S
									imgLabels_row[c] = imgLabels_row[c - 2];
									continue;
								}
								else {
									if (condition_g) {
										if (condition_b) {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
										else {
											//Action_11: Merge labels of block Q and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
											continue;
										}
									}
									else {
										//Action_11: Merge labels of block Q and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
										continue;
									}
								}
							}
							else {
								//Action_11: Merge labels of block Q and S
								imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
								continue;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										//Action_6: Assign label of block S
										imgLabels_row[c] = imgLabels_row[c - 2];
										continue;
									}
									else {
										if (condition_c) {
											if (condition_h) {
												//Action_6: Assign label of block S
												imgLabels_row[c] = imgLabels_row[c - 2];
												continue;
											}
											else {
												if (condition_g) {
													if (condition_b) {
														//Action_6: Assign label of block S
														imgLabels_row[c] = imgLabels_row[c - 2];
														continue;
													}
													else {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
												}
												else {
													//Action_12: Merge labels of block R and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
													continue;
												}
											}
										}
										else {
											//Action_12: Merge labels of block R and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
											continue;
										}
									}
								}
								else {
									//Action_12: Merge labels of block R and S
									imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
									continue;
								}
							}
							else {
								//Action_6: Assign label of block S
								imgLabels_row[c] = imgLabels_row[c - 2];
								continue;
							}
						}
						else {
							//Action_6: Assign label of block S
							imgLabels_row[c] = imgLabels_row[c - 2];
							continue;
						}
					}
				}
				else {
					if (condition_r) {
						if (condition_j) {
							if (condition_m) {
								if (condition_h) {
									if (condition_i) {
										//Action_6: Assign label of block S
										imgLabels_row[c] = imgLabels_row[c - 2];
										continue;
									}
									else {
										if (condition_c) {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
										else {
											//Action_11: Merge labels of block Q and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
											continue;
										}
									}
								}
								else {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												//Action_6: Assign label of block S
												imgLabels_row[c] = imgLabels_row[c - 2];
												continue;
											}
											else {
												if (condition_c) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
										}
										else {
											//Action_11: Merge labels of block Q and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
											continue;
										}
									}
									else {
										//Action_11: Merge labels of block Q and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
										continue;
									}
								}
							}
							else {
								if (condition_i) {
									//Action_11: Merge labels of block Q and S
									imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
									continue;
								}
								else {
									if (condition_h) {
										if (condition_c) {
											//Action_11: Merge labels of block Q and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
											continue;
										}
										else {
											//Action_14: Merge labels of block P, Q and S
											imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
											continue;
										}
									}
									else {
										//Action_11: Merge labels of block Q and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
										continue;
									}
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_m) {
										if (condition_h) {
											if (condition_d) {
												if (condition_i) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													if (condition_c) {
														//Action_6: Assign label of block S
														imgLabels_row[c] = imgLabels_row[c - 2];
														continue;
													}
													else {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
												}
											}
											else {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											if (condition_d) {
												if (condition_g) {
													if (condition_b) {
														if (condition_i) {
															//Action_6: Assign label of block S
															imgLabels_row[c] = imgLabels_row[c - 2];
															continue;
														}
														else {
															if (condition_c) {
																//Action_6: Assign label of block S
																imgLabels_row[c] = imgLabels_row[c - 2];
																continue;
															}
															else {
																//Action_12: Merge labels of block R and S
																imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
																continue;
															}
														}
													}
													else {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
												}
												else {
													//Action_12: Merge labels of block R and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												if (condition_i) {
													if (condition_g) {
														if (condition_b) {
															//Action_12: Merge labels of block R and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
															continue;
														}
														else {
															//Action_16: labels of block Q, R and S
															imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
															continue;
														}
													}
													else {
														//Action_16: labels of block Q, R and S
														imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
														continue;
													}
												}
												else {
													//Action_12: Merge labels of block R and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
													continue;
												}
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_d) {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
											else {
												//Action_16: labels of block Q, R and S
												imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											if (condition_h) {
												if (condition_d) {
													if (condition_c) {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
													else {
														//Action_15: Merge labels of block P, R and S
														imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
														continue;
													}
												}
												else {
													//Action_15: Merge labels of block P, R and S
													imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
										}
									}
								}
								else {
									if (condition_h) {
										if (condition_m) {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
										else {
											// ACTION_9 Merge labels of block P and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
											continue;
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_g) {
													if (condition_b) {
														//Action_6: Assign label of block S
														imgLabels_row[c] = imgLabels_row[c - 2];
														continue;
													}
													else {
														//Action_11: Merge labels of block Q and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
														continue;
													}
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_11: Merge labels of block Q and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
									}
								}
							}
							else {
								if (condition_h) {
									if (condition_m) {
										//Action_6: Assign label of block S
										imgLabels_row[c] = imgLabels_row[c - 2];
										continue;
									}
									else {
										// ACTION_9 Merge labels of block P and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
										continue;
									}
								}
								else {
									if (condition_i) {
										if (condition_m) {
											if (condition_g) {
												if (condition_b) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_11: Merge labels of block Q and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											//Action_11: Merge labels of block Q and S
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
											continue;
										}
									}
									else {
										//Action_6: Assign label of block S
										imgLabels_row[c] = imgLabels_row[c - 2];
										continue;
									}
								}
							}
						}
					}
					else {
						if (condition_j) {
							if (condition_i) {
								//Action_4: Assign label of block Q 
								imgLabels_row[c] = imgLabels_row_prev_prev[c];
								continue;
							}
							else {
								if (condition_h) {
									if (condition_c) {
										//Action_4: Assign label of block Q 
										imgLabels_row[c] = imgLabels_row_prev_prev[c];
										continue;
									}
									else {
										//Action_7: Merge labels of block P and Q
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
										continue;
									}
								}
								else {
									//Action_4: Assign label of block Q 
									imgLabels_row[c] = imgLabels_row_prev_prev[c];
									continue;
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											//Action_5: Assign label of block R
											imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
											continue;
										}
										else {
											// ACTION_10 Merge labels of block Q and R
											imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
											continue;
										}
									}
									else {
										if (condition_h) {
											if (condition_d) {
												if (condition_c) {
													//Action_5: Assign label of block R
													imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
													continue;
												}
												else {
													//Action_8: Merge labels of block P and R
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
													continue;
												}
											}
											else {
												//Action_8: Merge labels of block P and R
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
												continue;
											}
										}
										else {
											//Action_5: Assign label of block R
											imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
											continue;
										}
									}
								}
								else {
									if (condition_i) {
										//Action_4: Assign label of block Q 
										imgLabels_row[c] = imgLabels_row_prev_prev[c];
										continue;
									}
									else {
										if (condition_h) {
											//Action_3: Assign label of block P
											imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
											continue;
										}
										else {
											//Action_2: New label (the block has foreground pixels and is not connected to anything else)
											imgLabels_row[c] = lunique;
											P[lunique] = lunique;
											lunique = lunique + 1;
											continue;
										}
									}
								}
							}
							else {
								if (condition_i) {
									//Action_4: Assign label of block Q 
									imgLabels_row[c] = imgLabels_row_prev_prev[c];
									continue;
								}
								else {
									if (condition_h) {
										//Action_3: Assign label of block P
										imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
										continue;
									}
									else {
										//Action_2: New label (the block has foreground pixels and is not connected to anything else)
										imgLabels_row[c] = lunique;
										P[lunique] = lunique;
										lunique = lunique + 1;
										continue;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (condition_s) {
					if (condition_p) {
						if (condition_n) {
							if (condition_j) {
								if (condition_i) {
									//Action_6: Assign label of block S
									imgLabels_row[c] = imgLabels_row[c - 2];
									continue;
								}
								else {
									if (condition_c) {
										if (condition_h) {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
										else {
											if (condition_g) {
												if (condition_b) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_11: Merge labels of block Q and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
												continue;
											}
										}
									}
									else {
										//Action_11: Merge labels of block Q and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
										continue;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
										else {
											if (condition_c) {
												if (condition_h) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															//Action_6: Assign label of block S
															imgLabels_row[c] = imgLabels_row[c - 2];
															continue;
														}
														else {
															//Action_12: Merge labels of block R and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
															continue;
														}
													}
													else {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
												}
											}
											else {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
										}
									}
									else {
										//Action_12: Merge labels of block R and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
										continue;
									}
								}
								else {
									//Action_6: Assign label of block S
									imgLabels_row[c] = imgLabels_row[c - 2];
									continue;
								}
							}
						}
						else {
							if (condition_r) {
								if (condition_j) {
									if (condition_m) {
										if (condition_h) {
											if (condition_i) {
												//Action_6: Assign label of block S
												imgLabels_row[c] = imgLabels_row[c - 2];
												continue;
											}
											else {
												if (condition_c) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
										}
										else {
											if (condition_g) {
												if (condition_b) {
													if (condition_i) {
														//Action_6: Assign label of block S
														imgLabels_row[c] = imgLabels_row[c - 2];
														continue;
													}
													else {
														if (condition_c) {
															//Action_6: Assign label of block S
															imgLabels_row[c] = imgLabels_row[c - 2];
															continue;
														}
														else {
															//Action_11: Merge labels of block Q and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
															continue;
														}
													}
												}
												else {
													//Action_11: Merge labels of block Q and S
													imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_11: Merge labels of block Q and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
												continue;
											}
										}
									}
									else {
										//Action_11: Merge labels of block Q and S
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
										continue;
									}
								}
								else {
									if (condition_k) {
										if (condition_d) {
											if (condition_m) {
												if (condition_h) {
													if (condition_i) {
														//Action_6: Assign label of block S
														imgLabels_row[c] = imgLabels_row[c - 2];
														continue;
													}
													else {
														if (condition_c) {
															//Action_6: Assign label of block S
															imgLabels_row[c] = imgLabels_row[c - 2];
															continue;
														}
														else {
															//Action_12: Merge labels of block R and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
															continue;
														}
													}
												}
												else {
													if (condition_g) {
														if (condition_b) {
															if (condition_i) {
																//Action_6: Assign label of block S
																imgLabels_row[c] = imgLabels_row[c - 2];
																continue;
															}
															else {
																if (condition_c) {
																	//Action_6: Assign label of block S
																	imgLabels_row[c] = imgLabels_row[c - 2];
																	continue;
																}
																else {
																	//Action_12: Merge labels of block R and S
																	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
																	continue;
																}
															}
														}
														else {
															//Action_12: Merge labels of block R and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
															continue;
														}
													}
													else {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
												}
											}
											else {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											if (condition_i) {
												if (condition_m) {
													if (condition_h) {
														//Action_12: Merge labels of block R and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
														continue;
													}
													else {
														if (condition_g) {
															if (condition_b) {
																//Action_12: Merge labels of block R and S
																imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
																continue;
															}
															else {
																//Action_16: labels of block Q, R and S
																imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
																continue;
															}
														}
														else {
															//Action_16: labels of block Q, R and S
															imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
															continue;
														}
													}
												}
												else {
													//Action_16: labels of block Q, R and S
													imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
													continue;
												}
											}
											else {
												//Action_12: Merge labels of block R and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
												continue;
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_h) {
													//Action_6: Assign label of block S
													imgLabels_row[c] = imgLabels_row[c - 2];
													continue;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															//Action_6: Assign label of block S
															imgLabels_row[c] = imgLabels_row[c - 2];
															continue;
														}
														else {
															//Action_11: Merge labels of block Q and S
															imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
															continue;
														}
													}
													else {
														//Action_11: Merge labels of block Q and S
														imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
														continue;
													}
												}
											}
											else {
												//Action_11: Merge labels of block Q and S
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
												continue;
											}
										}
										else {
											//Action_6: Assign label of block S
											imgLabels_row[c] = imgLabels_row[c - 2];
											continue;
										}
									}
								}
							}
							else {
								if (condition_j) {
									//Action_4: Assign label of block Q 
									imgLabels_row[c] = imgLabels_row_prev_prev[c];
									continue;
								}
								else {
									if (condition_k) {
										if (condition_i) {
											if (condition_d) {
												//Action_5: Assign label of block R
												imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
												continue;
											}
											else {
												// ACTION_10 Merge labels of block Q and R
												imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
												continue;
											}
										}
										else {
											//Action_5: Assign label of block R
											imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
											continue;
										}
									}
									else {
										if (condition_i) {
											//Action_4: Assign label of block Q 
											imgLabels_row[c] = imgLabels_row_prev_prev[c];
											continue;
										}
										else {
											//Action_2: New label (the block has foreground pixels and is not connected to anything else)
											imgLabels_row[c] = lunique;
											P[lunique] = lunique;
											lunique = lunique + 1;
											continue;
										}
									}
								}
							}
						}
					}
					else {
						if (condition_r) {
							//Action_6: Assign label of block S
							imgLabels_row[c] = imgLabels_row[c - 2];
							continue;
						}
						else {
							if (condition_n) {
								//Action_6: Assign label of block S
								imgLabels_row[c] = imgLabels_row[c - 2];
								continue;
							}
							else {
								//Action_2: New label (the block has foreground pixels and is not connected to anything else)
								imgLabels_row[c] = lunique;
								P[lunique] = lunique;
								lunique = lunique + 1;
								continue;
							}
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_j) {
							//Action_4: Assign label of block Q 
							imgLabels_row[c] = imgLabels_row_prev_prev[c];
							continue;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										//Action_5: Assign label of block R
										imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
										continue;
									}
									else {
										// ACTION_10 Merge labels of block Q and R
										imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
										continue;
									}
								}
								else {
									//Action_5: Assign label of block R
									imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
									continue;
								}
							}
							else {
								if (condition_i) {
									//Action_4: Assign label of block Q 
									imgLabels_row[c] = imgLabels_row_prev_prev[c];
									continue;
								}
								else {
									//Action_2: New label (the block has foreground pixels and is not connected to anything else)
									imgLabels_row[c] = lunique;
									P[lunique] = lunique;
									lunique = lunique + 1;
									continue;
								}
							}
						}
					}
					else {
						if (condition_t) {
							//Action_2: New label (the block has foreground pixels and is not connected to anything else)
							imgLabels_row[c] = lunique;
							P[lunique] = lunique;
							lunique = lunique + 1;
							continue;
						}
						else {
							// Action_1: No action (the block has no foreground pixels)
							imgLabels_row[c] = 0;
							continue;
						}
					}
				}
			}
		}
	}

#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e
 
#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k
 
#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p
 
#undef condition_r
#undef condition_s
#undef condition_t

}

int BBDT_OPT(const Mat1b &img, Mat1i &imgLabels) {
	
    imgLabels = cv::Mat1i(img.size());
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

    firstScanBBDT_OPT(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	// Second scan
	if (imgLabels.rows & 1){
		if (imgLabels.cols & 1){
			//Case 1: both rows and cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								if (img_row_fol[c]>0)
									imgLabels_row_fol[c] = iLabel;
								else
									imgLabels_row_fol[c] = 0;
								if (img_row_fol[c + 1]>0)
									imgLabels_row_fol[c + 1] = iLabel;
								else
									imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								imgLabels_row_fol[c] = 0;
								imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
						}
					}
				}
			}
		}//END Case 1
		else{
			//Case 2: only rows odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1]>0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END Case 2
	} 
	else{
		if (imgLabels.cols & 1){
			//Case 3: only cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row_fol[c]>0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else{
						imgLabels_row[c] = 0;
						imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END case 3
		else{
			//Case 4: nothing odd
			for (int r = 0; r < imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1] > 0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (img_row_fol[c] > 0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (img_row_fol[c + 1] > 0)
							imgLabels_row_fol[c + 1] = iLabel;
						else
							imgLabels_row_fol[c + 1] = 0;
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						imgLabels_row_fol[c] = 0;
						imgLabels_row_fol[c + 1] = 0;
					}
				}
			}
		}//END case 4
	}

	fastFree(P);
	return nLabel;
}

inline static
void firstScanBBDT_MEM(memMat<uchar> &img, memMat<int>& imgLabels, memVector<uint> &P, uint &lunique) {
	int w(img.cols), h(img.rows);

	for (int r = 0; r<h; r += 2) {
		for (int c = 0; c < w; c += 2) {

			// We work with 2x2 blocks
			// +-+-+-+
			// |P|Q|R|
			// +-+-+-+
			// |S|X|
			// +-+-+

			// The pixels are named as follows
			// +---+---+---+
			// |a b|c d|e f|
			// |g h|i j|k l|
			// +---+---+---+
			// |m n|o p|
			// |q r|s t|
			// +---+---+

			// Pixels a, f, l, q are not needed, since we need to understand the 
			// the connectivity between these blocks and those pixels only metter
			// when considering the outer connectivities

			// A bunch of defines used to check if the pixels are foreground, 
			// without going outside the image limits.

#define condition_b c-1>=0 && r-2>=0 && img(r-2, c-1)>0
#define condition_c r-2>=0 && img(r-2, c)>0
#define condition_d c+1<w && r-2>=0 && img(r-2, c+1)>0
#define condition_e c+2<w && r-2>=0 && img(r-2, c+2)>0

#define condition_g c-2>=0 && r-1>=0 && img(r-1, c-2)>0
#define condition_h c-1>=0 && r-1>=0 && img(r-1, c-1)>0
#define condition_i r-1>=0 && img(r-1, c)>0
#define condition_j c+1<w && r-1>=0 && img(r-1, c+1)>0
#define condition_k c+2<w && r-1>=0 && img(r-1, c+2)>0

#define condition_m c-2>=0 && img(r, c-2)>0
#define condition_n c-1>=0 && img(r, c-1)>0
#define condition_o img(r,c)>0
#define condition_p c+1<w && img(r,c+1)>0

#define condition_r c-1>=0 && r+1<h && img(r+1, c-1)>0
#define condition_s r+1<h && img(r+1, c)>0
#define condition_t c+1<w && r+1<h && img(r+1, c+1)>0

			// This is a decision tree which allows to choose which action to 
			// perform, checking as few conditions as possible.
			// Actions are available after the tree.

			if (condition_o) {
				if (condition_n) {
					if (condition_j) {
						if (condition_i) {
							goto action_6;
						}
						else {
							if (condition_c) {
								if (condition_h) {
									goto action_6;
								}
								else {
									if (condition_g) {
										if (condition_b) {
											goto action_6;
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								goto action_11;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										goto action_6;
									}
									else {
										if (condition_c) {
											if (condition_h) {
												goto action_6;
											}
											else {
												if (condition_g) {
													if (condition_b) {
														goto action_6;
													}
													else {
														goto action_12;
													}
												}
												else {
													goto action_12;
												}
											}
										}
										else {
											goto action_12;
										}
									}
								}
								else {
									goto action_12;
								}
							}
							else {
								goto action_6;
							}
						}
						else {
							goto action_6;
						}
					}
				}
				else {
					if (condition_r) {
						if (condition_j) {
							if (condition_m) {
								if (condition_h) {
									if (condition_i) {
										goto action_6;
									}
									else {
										if (condition_c) {
											goto action_6;
										}
										else {
											goto action_11;
										}
									}
								}
								else {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												goto action_6;
											}
											else {
												if (condition_c) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								if (condition_i) {
									goto action_11;
								}
								else {
									if (condition_h) {
										if (condition_c) {
											goto action_11;
										}
										else {
											goto action_14;
										}
									}
									else {
										goto action_11;
									}
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_m) {
										if (condition_h) {
											if (condition_d) {
												if (condition_i) {
													goto action_6;
												}
												else {
													if (condition_c) {
														goto action_6;
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
										else {
											if (condition_d) {
												if (condition_g) {
													if (condition_b) {
														if (condition_i) {
															goto action_6;
														}
														else {
															if (condition_c) {
																goto action_6;
															}
															else {
																goto action_12;
															}
														}
													}
													else {
														goto action_12;
													}
												}
												else {
													goto action_12;
												}
											}
											else {
												if (condition_i) {
													if (condition_g) {
														if (condition_b) {
															goto action_12;
														}
														else {
															goto action_16;
														}
													}
													else {
														goto action_16;
													}
												}
												else {
													goto action_12;
												}
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_d) {
												goto action_12;
											}
											else {
												goto action_16;
											}
										}
										else {
											if (condition_h) {
												if (condition_d) {
													if (condition_c) {
														goto action_12;
													}
													else {
														goto action_15;
													}
												}
												else {
													goto action_15;
												}
											}
											else {
												goto action_12;
											}
										}
									}
								}
								else {
									if (condition_h) {
										if (condition_m) {
											goto action_6;
										}
										else {
											goto action_9;
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_g) {
													if (condition_b) {
														goto action_6;
													}
													else {
														goto action_11;
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_h) {
									if (condition_m) {
										goto action_6;
									}
									else {
										goto action_9;
									}
								}
								else {
									if (condition_i) {
										if (condition_m) {
											if (condition_g) {
												if (condition_b) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_11;
										}
									}
									else {
										goto action_6;
									}
								}
							}
						}
					}
					else {
						if (condition_j) {
							if (condition_i) {
								goto action_4;
							}
							else {
								if (condition_h) {
									if (condition_c) {
										goto action_4;
									}
									else {
										goto action_7;
									}
								}
								else {
									goto action_4;
								}
							}
						}
						else {
							if (condition_p) {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											goto action_5;
										}
										else {
											goto action_10;
										}
									}
									else {
										if (condition_h) {
											if (condition_d) {
												if (condition_c) {
													goto action_5;
												}
												else {
													goto action_8;
												}
											}
											else {
												goto action_8;
											}
										}
										else {
											goto action_5;
										}
									}
								}
								else {
									if (condition_i) {
										goto action_4;
									}
									else {
										if (condition_h) {
											goto action_3;
										}
										else {
											goto action_2;
										}
									}
								}
							}
							else {
								if (condition_i) {
									goto action_4;
								}
								else {
									if (condition_h) {
										goto action_3;
									}
									else {
										goto action_2;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (condition_s) {
					if (condition_p) {
						if (condition_n) {
							if (condition_j) {
								if (condition_i) {
									goto action_6;
								}
								else {
									if (condition_c) {
										if (condition_h) {
											goto action_6;
										}
										else {
											if (condition_g) {
												if (condition_b) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
									}
									else {
										goto action_11;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											goto action_6;
										}
										else {
											if (condition_c) {
												if (condition_h) {
													goto action_6;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															goto action_6;
														}
														else {
															goto action_12;
														}
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
									}
									else {
										goto action_12;
									}
								}
								else {
									goto action_6;
								}
							}
						}
						else {
							if (condition_r) {
								if (condition_j) {
									if (condition_m) {
										if (condition_h) {
											if (condition_i) {
												goto action_6;
											}
											else {
												if (condition_c) {
													goto action_6;
												}
												else {
													goto action_11;
												}
											}
										}
										else {
											if (condition_g) {
												if (condition_b) {
													if (condition_i) {
														goto action_6;
													}
													else {
														if (condition_c) {
															goto action_6;
														}
														else {
															goto action_11;
														}
													}
												}
												else {
													goto action_11;
												}
											}
											else {
												goto action_11;
											}
										}
									}
									else {
										goto action_11;
									}
								}
								else {
									if (condition_k) {
										if (condition_d) {
											if (condition_m) {
												if (condition_h) {
													if (condition_i) {
														goto action_6;
													}
													else {
														if (condition_c) {
															goto action_6;
														}
														else {
															goto action_12;
														}
													}
												}
												else {
													if (condition_g) {
														if (condition_b) {
															if (condition_i) {
																goto action_6;
															}
															else {
																if (condition_c) {
																	goto action_6;
																}
																else {
																	goto action_12;
																}
															}
														}
														else {
															goto action_12;
														}
													}
													else {
														goto action_12;
													}
												}
											}
											else {
												goto action_12;
											}
										}
										else {
											if (condition_i) {
												if (condition_m) {
													if (condition_h) {
														goto action_12;
													}
													else {
														if (condition_g) {
															if (condition_b) {
																goto action_12;
															}
															else {
																goto action_16;
															}
														}
														else {
															goto action_16;
														}
													}
												}
												else {
													goto action_16;
												}
											}
											else {
												goto action_12;
											}
										}
									}
									else {
										if (condition_i) {
											if (condition_m) {
												if (condition_h) {
													goto action_6;
												}
												else {
													if (condition_g) {
														if (condition_b) {
															goto action_6;
														}
														else {
															goto action_11;
														}
													}
													else {
														goto action_11;
													}
												}
											}
											else {
												goto action_11;
											}
										}
										else {
											goto action_6;
										}
									}
								}
							}
							else {
								if (condition_j) {
									goto action_4;
								}
								else {
									if (condition_k) {
										if (condition_i) {
											if (condition_d) {
												goto action_5;
											}
											else {
												goto action_10;
											}
										}
										else {
											goto action_5;
										}
									}
									else {
										if (condition_i) {
											goto action_4;
										}
										else {
											goto action_2;
										}
									}
								}
							}
						}
					}
					else {
						if (condition_r) {
							goto action_6;
						}
						else {
							if (condition_n) {
								goto action_6;
							}
							else {
								goto action_2;
							}
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_j) {
							goto action_4;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										goto action_5;
									}
									else {
										goto action_10;
									}
								}
								else {
									goto action_5;
								}
							}
							else {
								if (condition_i) {
									goto action_4;
								}
								else {
									goto action_2;
								}
							}
						}
					}
					else {
						if (condition_t) {
							goto action_2;
						}
						else {
							goto action_1; 
						}
					}
				}
			}

			// Actions: the blocks label are provisionally stored in the top left 
			// pixel of the block in the labels image

			action_1:	//Action_1: No action (the block has no foreground pixels)
				imgLabels(r,c) = 0;
				continue;
			action_2:	//Action_2: New label (the block has foreground pixels and is not connected to anything else)
				imgLabels(r, c) = lunique;
				P[lunique] = lunique;
				lunique = lunique + 1;
				continue;
			action_3:	//Action_3: Assign label of block P
				imgLabels(r,c) = imgLabels(r - 2, c - 2);
				continue;
			action_4:	//Action_4: Assign label of block Q 
				imgLabels(r, c) = imgLabels(r - 2, c);
				continue;
			action_5:	//Action_5: Assign label of block R
				imgLabels(r,c) = imgLabels(r - 2, c + 2);
				continue;			
			action_6:	//Action_6: Assign label of block S
				imgLabels(r,c) = imgLabels(r, c - 2);
				continue;			
			action_7:	//Action_7: Merge labels of block P and Q
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c));
				continue;			
			action_8:	//Action_8: Merge labels of block P and R
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c + 2));
				continue;
			action_9:	// Action_9 Merge labels of block P and S
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r, c - 2));
				continue;
			action_10:	// Action_10 Merge labels of block Q and R
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r - 2, c + 2));
				continue;
			action_11:	//Action_11: Merge labels of block Q and S
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r, c - 2));
				continue;
			action_12:	//Action_12: Merge labels of block R and S
				imgLabels(r, c) = set_union(P, (uint)imgLabels(r - 2, c + 2), (uint)imgLabels(r, c - 2));
				continue;
			action_14:	//Action_14: Merge labels of block P, Q and S
				imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c)), (uint)imgLabels(r, c - 2));
				continue;
			action_15:	//Action_15: Merge labels of block P, R and S
				imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c - 2), (uint)imgLabels(r - 2, c + 2)), (uint)imgLabels(r, c - 2));
				continue;
			action_16:	//Action_16: labels of block Q, R and S
				imgLabels(r, c) = set_union(P, set_union(P, (uint)imgLabels(r - 2, c), (uint)imgLabels(r - 2, c + 2)), (uint)imgLabels(r, c - 2));
				continue;
		}
	}

#undef condition_b
#undef condition_c
#undef condition_d
#undef condition_e

#undef condition_g
#undef condition_h
#undef condition_i
#undef condition_j
#undef condition_k

#undef condition_m
#undef condition_n
#undef condition_o
#undef condition_p

#undef condition_r
#undef condition_s
#undef condition_t

}

int BBDT_MEM(const Mat1b &img_origin, vector<unsigned long int> &accesses) {

	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img_origin.rows*img_origin.cols / 4;
	
	//Tree of labels
	memMat<uchar> img(img_origin); 
	memMat<int> imgLabels(img_origin.size());
	memVector<uint> P(Plength);
	
	//Background
	P[0] = 0;
	uint lunique = 1;

	firstScanBBDT_MEM(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	//Second scan
	for (int r = 0; r < img_origin.rows; r += 2) {
		for (int c = 0; c < img_origin.cols; c += 2) {
			int iLabel = imgLabels(r,c);
			if (iLabel>0) {
				iLabel = P[iLabel];
				if (img(r,c) > 0)
					imgLabels(r,c) = iLabel;
				else
					imgLabels(r, c) = 0;
				if (c + 1 < img_origin.cols) {
					if (img(r, c + 1) > 0)
						imgLabels(r, c + 1) = iLabel;
					else
						imgLabels(r, c + 1) = 0;
					if (r + 1 < img_origin.rows) {
						if (img(r + 1, c) > 0)
							imgLabels(r + 1, c) = iLabel;
						else
							imgLabels(r + 1, c) = 0;
						if (img(r + 1, c + 1) > 0)
							imgLabels(r + 1, c + 1) = iLabel;
						else
							imgLabels(r + 1, c + 1) = 0;
					}
				}
				else if (r + 1 < img_origin.rows) {
					if (img(r + 1, c) > 0)
						imgLabels(r + 1, c) = iLabel;
					else
						imgLabels(r + 1, c) = 0;
				}
			}
			else {
				imgLabels(r , c) = 0;
				if (c + 1 < img_origin.cols) {
					imgLabels(r, c + 1) = 0;
					if (r + 1 < img_origin.rows) {
						imgLabels(r + 1, c) = 0;
						imgLabels(r + 1, c + 1) = 0;
					}
				}
				else if (r + 1 < img_origin.rows) {
					imgLabels(r + 1, c) = 0;
				}
			}
		}
	}

	// Store total accesses in the output vector 'accesses'
	accesses = vector<unsigned long int>((int)MD_SIZE, 0);

	accesses[MD_BINARY_MAT] = (unsigned long int)img.getTotalAcesses();
	accesses[MD_LABELED_MAT] = (unsigned long int)imgLabels.getTotalAcesses();
	accesses[MD_EQUIVALENCE_VEC] = (unsigned long int)P.getTotalAcesses();

	//a = imgLabels.getImage(); 

	return nLabel;
}
