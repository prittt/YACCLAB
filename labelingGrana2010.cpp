#include "labelingGrana2010.h"

using namespace cv;
using namespace std;

//Find the root of the tree of node i
//template<typename LabelT>
inline static
uint findRoot(const uint *P, uint i){
	uint root = i;
	while (P[root] < root){
		root = P[root];
	}
	return root;
}

//Make all nodes in the path of node i point to root
//template<typename LabelT>
inline static
void setRoot(uint *P, uint i, uint root){
	while (P[i] < i){
		uint j = P[i];
		P[i] = root;
		i = j;
	}
	P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
//template<typename LabelT>
inline static
uint find(uint *P, uint i){
	uint root = findRoot(P, i);
	setRoot(P, i, root);
	return root;
}

//unite the two trees containing nodes i and j and return the new root
//template<typename LabelT>
inline static
uint set_union(uint *P, uint i, uint j){
	uint root = findRoot(P, i);
	if (i != j){
		uint rootj = findRoot(P, j);
		if (root > rootj){
			root = rootj;
		}
		setRoot(P, j, root);
	}
	setRoot(P, i, root);
	return root;
}

//Flatten the Union Find tree and relabel the components
//template<typename LabelT>
inline static
uint flattenL(uint *P, uint length){
	uint k = 1;
	for (uint i = 1; i < length; ++i){
		if (P[i] < i){
			P[i] = P[P[i]];
		}
		else{
			P[i] = k; k = k + 1;
		}
	}
	return k;
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

			// Actions: the blocks label are provisionally stored in the top left 
			// pixel of the block in the labels image

//		action_1:	//Action_1: No action (the block has no foreground pixels)
//			imgLabels_row[c] = 0;
//			continue;
//
//		action_2:	//Action_2: New label (the block has foreground pixels and is not connected to anything else)
//			imgLabels_row[c] = lunique;
//			P[lunique] = lunique;
//			lunique = lunique + 1;
//			continue;
//
//		action_3:	//Action_3: Assign label of block P
//			imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
//			continue;
//
//		action_4:	//Action_4: Assign label of block Q 
//			imgLabels_row[c] = imgLabels_row_prev_prev[c];
//			continue;
//
//		action_5:	//Action_5: Assign label of block R
//			imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
//			continue;
//
//		action_6:	//Action_6: Assign label of block S
//			imgLabels_row[c] = imgLabels_row[c - 2];
//			continue;
//
//		action_7:	//Action_7: Merge labels of block P and Q
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
//			continue;
//
//		action_8:	//Action_8: Merge labels of block P and R
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
//			continue;
//
//		action_9:	// ACTION_9 Merge labels of block P and S
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
//			continue;
//
//		action_10:	// ACTION_10 Merge labels of block Q and R
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
//			continue;
//
//		action_11:	//Action_11: Merge labels of block Q and S
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
//			continue;
//
//		action_12:	//Action_12: Merge labels of block R and S
//			imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
//			continue;
//
//			//Action_13:	// Merge labels of block P, Q and R
//			//			imgLabels(r,c) = es.resolve(imgLabels(r-2,c-2),imgLabels(r-2,c),imgLabels(r-2,c+2));
//			//			continue;
//
//		action_14:	//Action_14: Merge labels of block P, Q and S
//			imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
//			continue;
//
//		action_15:	//Action_15: Merge labels of block P, R and S
//			imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
//			continue;
//
//		action_16:	//Action_16: labels of block Q, R and S
//			imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
//			continue;
		}
	}

}

int BBDT_OPT(const cv::Mat1b &img, cv::Mat1i &imgLabels) {
	
    imgLabels = cv::Mat1i(img.size()); // Controlla!!! TODO
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
