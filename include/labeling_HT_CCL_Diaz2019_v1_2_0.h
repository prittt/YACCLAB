// Copyright(c) 2019 
/*\author{ Fernando Diaz - del - Rio, Pablo Sanchez - Cuevas }
\address{ Department of Computer Architecture and Technology.
University  of Seville.Spain. }
\author{ Helena Molina - Abril, Pedro Real }
\address{ Department of Applied Mathematics I.University  of Seville.Spain. }
*/
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

#ifndef YACCLAB_LABELING_CCLHSF_H_
#define YACCLAB_LABELING_CCLHSF_H_

////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;

#include <omp.h>

////////////////////
// for testing and debugging purposes. 
////////////////////

//#define  DEBUG_INNER_STATISTICS
//#define  DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS   //for debugging purposes . 

////////////////////

#define MAX_NUM_THREADS 8
#define _OMP_SCHEDULING  dynamic 
#define CHUNK_SIZE 16

#pragma once
#include "opencv2/opencv.hpp"

////////////////////////////////  typedef_constants.h
typedef unsigned char Bool;

// binary images are allowed
typedef unsigned char ImageType;
typedef int JumpType;

// the index type has sign
typedef int RowColType;

//////////////////
// colors for :
#define  FG  1
#define  BG  0

/////////////
// Macros
#define tag_ind(row, col) ( (col)+(N_COLS*row) )

////////////////////

int N_ROWS, N_COLS;

////////////////////// --------------------
// protoypes of four main functions 

////////////////////// --------------------
// protoypes of four main functions 
//1st stage
void J_init(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* lista, int id, int trozo);
//2nd stage
void J_computation(cv::Mat1i &J, struct listaPadres* lista, int id, int trozo);

// fusing 1st and 2nd stages:
void init_2LookUpTables_and_infection_process(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* lista, int id, int trozo);

// 3rd and 4th stages:
void Transports_only(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* lista);
unsigned Labelling_only(const cv::Mat1b* Itemp, cv::Mat1i* Jtemp, cv::Mat1i* Ltemp, int id, int numHilos);
void init_borders(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres *lista, int numHilos, int trozo);

struct listaPadres {
	int* padresI;
	int* padresJ;
	int numPadres;
	int* saltos;
};
struct listaPadres listaGeneral[MAX_NUM_THREADS];


///%%%%%%%%%%%%%%%%%%%%%%///////////////////%%%%%%%%%%%%%%%%%%%%%%%%
#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

class labeling_CCLHSF : public Labeling2D<Connectivity2D::CONN_8, true> {
public:

	void PerformLabeling()
	{
		N_ROWS = img_.rows, N_COLS = img_.cols;


		img_labels_ = cv::Mat1i(img_.size()); // Memory allocation for the output image
											  // jumps accross the image (positive for FG, negative for BG)
		cv::Mat1i imgJ ;
		imgJ = cv::Mat1i(img_.size());

		if (N_ROWS == 1 && N_COLS == 1) {
			n_labels_ = 1;
			return;
		}
		int numHilos;

		cv::Mat1b& img_ref = img_;
		
#define NOF_ASKED_THREADS 8
		omp_set_num_threads(NOF_ASKED_THREADS);
		
#ifdef CALCULO_TIEMPO
		t = omp_get_wtime();
#endif
#pragma omp parallel default(none) shared(imgJ, numHilos, img_ref, listaGeneral, N_ROWS)
		{
			int id = omp_get_thread_num();
			numHilos = omp_get_num_threads();
			int trozo = (N_ROWS / numHilos) + (N_ROWS % numHilos);

			if (id == 0) {

				for (int i = 0; i < numHilos; i++) {
					listaGeneral[i].padresI = (int*)malloc(trozo * img_.cols * sizeof(int) / 4);
					listaGeneral[i].padresJ = (int*)malloc(trozo * img_.cols * sizeof(int) / 4);
					listaGeneral[i].numPadres = 0;
					listaGeneral[i].saltos = (int*)malloc(trozo * img_.cols * sizeof(int) / 4);
				}

				init_borders(img_, imgJ, listaGeneral, numHilos, trozo);

			}
#pragma omp barrier
			// initializing Jump matrices
			trozo = (N_ROWS / numHilos);

			J_init(img_, imgJ, &listaGeneral[id], id, trozo);
#pragma omp barrier
			J_computation(imgJ, &listaGeneral[id], id, trozo);

		}
		// transports
		Transports_only(img_, imgJ, listaGeneral); 

												   // label assignement 
		int nLabel = 0;
#pragma omp parallel default(none) shared(imgJ, numHilos, img_ref, listaGeneral,  N_ROWS) reduction(+:nLabel)
		{
			int id = omp_get_thread_num();
			int trozo = N_ROWS / numHilos;
			nLabel = Labelling_only(&img_, &imgJ, &img_labels_, id, numHilos);
		}
		for (int i = 0; i < numHilos; i++) {
			free(listaGeneral[i].padresI);
			free(listaGeneral[i].padresJ);
			free(listaGeneral[i].saltos);

		}

		n_labels_ = nLabel+1;

	}

	void PerformLabelingWithSteps()
	{
		double alloc_timing = Alloc();

		perf_.start();
		AllScans();
		perf_.stop();
		perf_.store(Step(StepType::ALL_SCANS), perf_.last());

		perf_.start();
		Dealloc();
		perf_.stop();
		perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
	}

	void PerformLabelingMem(std::vector<uint64_t>& accesses)
	{
		MemMat<uchar> img(img_);
		MemMat<int> img_labels(img_.size());

		for (int r = 0; r < img_labels.rows; ++r) {
			for (int c = 0; c < img_labels.cols; ++c) {
				img_labels(r, c) = img(r, c);
			}
		}

		// Store total accesses in the output vector 'accesses'
		accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

		accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
		accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
	}

private:
	double Alloc()
	{
		// Memory allocation for the output image
		perf_.start();
		img_labels_ = cv::Mat1i(img_.size());
		memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
		perf_.stop();
		double t = perf_.last();
		perf_.start();
		memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
		perf_.stop();
		double ma_t = t - perf_.last();
		// Return total time
		return ma_t;
	}
	void Dealloc()
	{
		// No free for img_labels_ because it is required at the end of the algorithm 
	}
	void AllScans()
	{
		for (int r = 0; r < img_labels_.rows; ++r) {
			// Get rows pointer
			const uchar* const img_row = img_.ptr<uchar>(r);
			unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);

			for (int c = 0; c < img_labels_.cols; ++c) {
				img_labels_row[c] = img_row[c];
			}
		}
	}
};


//=========================================

//////////////////////////////////////////////////////////////
// FOUR MAIN STEPS:
////////////////////////////

// auxiliar function prototypes
RowColType FG_lookup_table_3neighb(RowColType Right, RowColType RiDo, RowColType Down);
RowColType BG_lookup_table_2neighb(RowColType Left, RowColType Up);

void init_2LookUpTables_and_infection_process(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres *lista, int id, int trozo) {

	int numHilos = omp_get_num_threads();

	int filaInicial = id * trozo;
	int filaFinal = filaInicial + trozo;

	filaFinal = (id == (omp_get_num_threads() - 1)) ? N_ROWS : filaFinal;
	filaInicial = (id == 0) ? filaInicial + 1 : filaInicial;

	for (RowColType r = filaInicial; r < filaFinal; r++) {

		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

		for (RowColType c = 1; c < N_COLS; c++) {
			if ((img_row[c] == BG)) {
				RowColType Le = img_row[c - 1];
				RowColType U = img_row_pre[c];

				JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc++ ?
				JumpType jump_rc = BG_lookup_table_2neighb(Le, U); //  priority: : +1 for FG;  -1 for BG  // 1º load  
				if (jump_rc != 0) {
					JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  
																 // a little faster using conditional code here
#ifdef DEBUG_INNER_STATISTICS
					int cond1 = (other_jump != 0);
					flag_any_change += cond1;
#endif
					JumpType total_jump = jump_rc + (other_jump);  // NO LOAD   
												
					*pjump_rc = total_jump;  // 1º store with if  /*	instead of if() version  ... see below 
				}
				else {
					*pjump_rc = 0;
				}

			}

		} //end o for (RowColType c = 1; c < N_COLS - 1; c++)
	}

	filaFinal = (id == (numHilos - 1)) ? filaFinal - 1 : filaFinal;
	filaInicial = (id == 0) ? filaInicial - 1 : filaInicial;

	for (RowColType r = filaFinal - 1; r >= filaInicial; r--) {

		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 
		for (int c = N_COLS - 2; c >= 0; c--) {
			if ((img_row[c] == FG)) {
				RowColType R = img_row[c + 1];
				RowColType RD = img_row_fol[c + 1];
				RowColType D = img_row_fol[c];

				JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc-- ?
				JumpType jump_rc = FG_lookup_table_3neighb(R, RD, D); //  priority: : +1 for FG;  -1 for BG ;  // 1º load  

				if (jump_rc != 0) {
					JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  
					JumpType total_jump = jump_rc + (other_jump);  // NO LOAD   
					*pjump_rc = total_jump;  // 1º store with if  /*	instead of if() version 
				}
				else {
					*pjump_rc = 0;
					lista->padresI[lista->numPadres] = r;
					lista->padresJ[lista->numPadres] = c;
					lista->saltos[lista->numPadres] = 0;
					lista->numPadres++;
				}
				// a little faster using conditional code here
#ifdef DEBUG_INNER_STATISTICS
				int cond1 = (other_jump != 0);
				flag_any_change += cond1;
#endif
			}
		}
	}
	return;
}
/////////////////////////

void J_init(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* lista, int id, int trozo) {

	int numHilos = omp_get_num_threads();

	int filaInicial = id * trozo, filaFinal = (id + 1) * trozo;
	filaInicial = (id == 0) ? 1 : filaInicial;
	filaFinal = (id == (numHilos - 1)) ? (N_ROWS - 1) : filaFinal; 

																  
	for (RowColType r = filaInicial; r < filaFinal; r++) {

		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

		for (RowColType c = 1; c < N_COLS - 1; c++) {
			if ((img_row[c] == FG)) {
				RowColType R = img_row[c + 1];
				RowColType RD = img_row_fol[c + 1];
				RowColType D = img_row_fol[c];

				imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D); //  priority: : +1 for FG;  -1 for BG

				if (imgJump_row[c] == 0) {
					lista->padresI[lista->numPadres] = r;
					lista->padresJ[lista->numPadres] = c;
					lista->saltos[lista->numPadres] = 0;
					lista->numPadres++;
				}
			}
			else {
				RowColType Le = img_row[c - 1];
				RowColType U = img_row_pre[c];

				imgJump_row[c] = BG_lookup_table_2neighb(Le, U); //  priority: : +1 for FG;  -1 for BG
			}

		} //end o for (RowColType c = 1; c < N_COLS - 1; c++)
	}
	return;
}

///////////////////////////////////////////
void J_computation(cv::Mat1i &J, struct listaPadres* lista, int id, int trozo) {

	int filaInicial = id * trozo, filaFinal = (id + 1) * trozo;
	filaFinal = (id == (omp_get_num_threads() - 1)) ? N_ROWS : filaFinal; 
	
#ifdef DEBUG_INNER_STATISTICS
	int flag_any_change = 0;  //counter of J changes 
#endif
	// In this version only two passes are done. The more parallel version has more passes (until no more infection is possible)
// this is a more guided version to help the compiler to avoid load/stores. Only 2 loads and one store for each change 

#ifdef DEBUG_INNER_STATISTICS
							  //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE) reduction (+:flag_any_change) 
#endif
#ifndef DEBUG_INNER_STATISTICS
							  //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE) 
#endif

	for (int r = filaInicial; r < filaFinal; r++) {
		// Get rows pointers
		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

   // forward travelling  or run promotes BG pixel jumps in a sequential fashion !
		for (int c = 0; c < N_COLS; c++) {
			JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc++ ?
			JumpType jump_rc = *pjump_rc;  // 1º load  
			JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  
			  // a little faster using conditional code here
			int cond1 = (other_jump != 0);

#ifdef DEBUG_INNER_STATISTICS
			flag_any_change += cond1;
#endif

			JumpType total_jump = jump_rc + (other_jump);  // NO LOAD   
			*pjump_rc = total_jump;  // 1º store with if 
		}
	} // endof   for (int r = 0; r < N_ROWS; r++)  

	for (int r = filaFinal - 1; r >= filaInicial; r--) {
		// Get rows pointers
		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

													   // reverse travelling  or run promotes FG pixel jumps in a sequential fashion !
		for (int c = N_COLS - 1; c >= 0; c--) {
			JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc-- ?
			JumpType jump_rc = *pjump_rc;  // 1º load  
			JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  

    // a little faster using conditional code here
			int cond1 = (other_jump != 0);
#ifdef DEBUG_INNER_STATISTICS
			flag_any_change += cond1;
#endif
			JumpType total_jump = jump_rc + (other_jump * (cond1));  // NO LOAD  
			*pjump_rc = total_jump;  // 1º store with if 
		}
	} // endof   for (int r = 0; r < N_ROWS; r++)  

#ifdef DEBUG_INNER_STATISTICS
	cout << "   - LJ_infection: Total nof changes, flag_any_change: " << flag_any_change << endl;
#endif
}  //endof  J_computation()

void Jonly_infection_process_original(cv::Mat1i &J, struct listaPadres* lista, int id, int trozo) {

	int filaInicial = id * trozo, filaFinal = (id + 1) * trozo;
	filaFinal = (id == (omp_get_num_threads() - 1)) ? N_ROWS : filaFinal; 
	int numCambios = -1;

#ifdef DEBUG_INNER_STATISTICS
	int flag_any_change = 0;  //counter of J changes 
#endif
							  // In this version only two passes are done. The more parallel version has more passes (until no more infection is possible)
							  // this is a more guided version to help the compiler to avoid load/stores. Only 2 loads and one store for each change 
#ifdef DEBUG_INNER_STATISTICS
							  //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE) reduction (+:flag_any_change) 
#endif
#ifndef DEBUG_INNER_STATISTICS
							  //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE) 
#endif
	while (numCambios != 0) { 
		numCambios = 0;
		for (int r = filaInicial; r < filaFinal; r++) {
			// Get rows pointers
			int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

														   // forward travelling  or run promotes BG pixel jumps in a sequential fashion !
			for (int c = 0; c < N_COLS; c++) {
				JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc++ ?
				JumpType jump_rc = *pjump_rc;  // 1º load  
				JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  

															 // a little faster using conditional code here
				int cond1 = (other_jump != 0);

#ifdef DEBUG_INNER_STATISTICS
				flag_any_change += cond1;
#endif

				JumpType total_jump = jump_rc + (other_jump);  // NO LOAD   
				numCambios = (total_jump != jump_rc) ? (numCambios + 1) : (numCambios);
				//JumpType total_jump = jump_rc + (other_jump * (cond1));  // note that it is not necessary the product
				*pjump_rc = total_jump;  // 1º store with if 

			}
			// reverse travelling  or run promotes FG pixel jumps in a sequential fashion !
			for (int c = N_COLS - 1; c >= 0; c--) {
				JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc-- ?
				JumpType jump_rc = *pjump_rc;  // 1º load  
				JumpType other_jump = *(pjump_rc + jump_rc); // 2º load  

															 // a little faster using conditional code here
				int cond1 = (other_jump != 0);
#ifdef DEBUG_INNER_STATISTICS
				flag_any_change += cond1;
#endif
				JumpType total_jump = jump_rc + (other_jump * (cond1));  // NO LOAD  
				numCambios = (total_jump != jump_rc) ? (numCambios + 1) : (numCambios);
				*pjump_rc = total_jump;  // 1º store with if 
			}
		} // endof   for (int r = 0; r < N_ROWS; r++)  

#ifdef DEBUG_INNER_STATISTICS
		cout << "   - LJ_infection: Total nof changes, flag_any_change: " << flag_any_change << endl;
#endif
	}
}  //endof  J_computation()


   //////////////////////////////////////////////////////////////
   // auxiliar function prototypes
int Jonly_transports_lots_of_searches(int R_SHIFT_SEARCH, int C_SHIFT_SEARCH, int R_SHIFT_NEW_ROOT, int C_SHIFT_NEW_ROOT,
	const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* lista);

JumpType  * travel_inside_J(JumpType *pprevious_other_color_crit_cell, JumpType jump_other_color_crit_cell, JumpType *);

//////////////////////////////////////////////////////////////
void Transports_only(const cv::Mat1b &Itemp, cv::Mat1i &Jtemp, struct listaPadres* lista) {
#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
#define NOF_CHRONOS_TRANSPORTS 18 //NO MORE THATN 18 WJILE ITER. ARE EXPECTED @
	double t0[NOF_CHRONOS_TRANSPORTS], t_inc[NOF_CHRONOS_TRANSPORTS];
	for (int j = 0; j < NOF_CHRONOS_TRANSPORTS; j++) {
		t_inc[j] = 1.0e+35;
	}
#endif

	int nof_transport_iterations = 0;
	int total_nof_transports = 0;  //initial  value 
	int flag_transports_1;
	int flag_transports_2;

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
	int ctimer = 0;
	t0[ctimer] = omp_get_wtime();
#endif
	//// First set of transports:

	// to the North-SOuth; arguments are :
	flag_transports_1 = Jonly_transports_lots_of_searches(0, 1, -1, 0, Itemp, Jtemp, lista);
	nof_transport_iterations++;

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
	t_inc[ctimer++] = omp_get_wtime() - t0[ctimer];
#endif

#ifdef DEBUG_INNER_STATISTICS
	cout << "   - 1. nof_transports:" << flag_transports_1 << endl;
#endif

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
	t0[ctimer] = omp_get_wtime();
#endif
	//// Second set of transports:
	// Now to the West-East; so the arguments are different:
	flag_transports_2 = Jonly_transports_lots_of_searches(1, 0, 0, -1, Itemp, Jtemp, lista);
	nof_transport_iterations++;

	total_nof_transports += (flag_transports_1 + flag_transports_2);

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
	t_inc[ctimer++] = omp_get_wtime() - t0[ctimer];
#endif

#ifdef DEBUG_INNER_STATISTICS
	cout << "   - 1 and 2. nof_transports:" << total_nof_transports << endl;
#endif

	while ((flag_transports_1 + flag_transports_2) != 0) {
		// to the North-SOuth; arguments are :
#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
		t0[ctimer] = omp_get_wtime();
#endif
		flag_transports_1 = Jonly_transports_lots_of_searches(0, 1, -1, 0, Itemp, Jtemp, lista);
		nof_transport_iterations++;

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
		t_inc[ctimer++] = omp_get_wtime() - t0[ctimer];
#endif

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
		t0[ctimer] = omp_get_wtime();
#endif
		// Now to the West-East; so the arguments are different:
		flag_transports_2 = Jonly_transports_lots_of_searches(1, 0, 0, -1, Itemp, Jtemp, lista);
		nof_transport_iterations++;

		total_nof_transports += (flag_transports_1 + flag_transports_2);

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
		t_inc[ctimer++] = omp_get_wtime() - t0[ctimer];
#endif
	}

#ifdef DEBUG_INNER_STATISTICS
	cout << "   - J: nof_transport_iterations: " << nof_transport_iterations << endl;
	cout << "   - J: total_nof_transports:" << total_nof_transports << endl;
#endif

#ifdef DEBUG_DEBUG_INNER_TIME_MEASUREMENT_CCLHSF_TRANSPORTS
	for (int j = 0; j < nof_transport_iterations; j++) {
		printf("    * Inner transport %2d : %lf \n", j, t_inc[j]);
	}
#endif

	return;
}

//////////////////////
unsigned Labelling_only(const cv::Mat1b* I, cv::Mat1i* J, cv::Mat1i* L, int id, int numHilos) {

	int* const imgJump_row_origin = (int*)J->ptr<uint>(0); //sign is required here. 
	unsigned total_nof_FG_labels = 0;

	int trozo = N_ROWS / numHilos;
	int filaInicial = id * trozo;
	int filaFinal = (id == (numHilos - 1)) ? N_ROWS : filaInicial + trozo;

	for (int r = filaInicial; r < filaFinal; r++) {    // FG critical cells at the SOUTH (BOTTOM) border are considered as real critical cells

		const uchar* const img_row = I->ptr<uchar>(r);
		int* const imgJump_row = (int*)J->ptr<uint>(r); //sign is required here. 
		int* const imgLabels_row = (int*)L->ptr<uint>(r); //sign is required here. 


		JumpType label_rc_00 = r * N_COLS;  //@ this must be recomputed according to the nof threads 

		for (int c = 0; c < N_COLS; c++) {  // FG critical cells at the East (RIGHT) border are considered as real critical cells

			const uchar *pI_rc = &(img_row[c]); //@ it is faster than pjump_rc++ ?
			JumpType *pjump_rc = &(imgJump_row[c]); //@ it is faster than pjump_rc++ ?
			int *pLabels_rc = &(imgLabels_row[c]); //@ it is faster than pjump_rc++ ?


			JumpType jump_rc = *pjump_rc;  // 1º load  

			total_nof_FG_labels += (jump_rc == 0 && *pI_rc == FG);

			{  // 2º load  // Labels L for possible critical cells were written at the init()
#ifdef DEBUG_INNER_STATISTICS
				nof_non_critical_FG_pixels++;
#endif
				JumpType *pnew_jump = pjump_rc + jump_rc; //&
				JumpType *pprevious_jump = pjump_rc + jump_rc; //&

				JumpType previous_jump = jump_rc;//&
				JumpType new_jump = *(pnew_jump); //  3º load  //&

				JumpType sum_jumps_rc = new_jump + previous_jump; //& total sum to do the final label assignment after the while loop

				pprevious_jump = pnew_jump;
				previous_jump = new_jump;

				pnew_jump = pnew_jump + new_jump;
				new_jump = *(pnew_jump); //  next loads  //&

				sum_jumps_rc += new_jump;

				while (new_jump != 0) { //&
					*(pprevious_jump) = (new_jump + previous_jump); //& new store to optimize future accessses

					pprevious_jump = pnew_jump;
					previous_jump = new_jump;

					pnew_jump = pnew_jump + new_jump;
					new_jump = *(pnew_jump); //  next loads  //&

					sum_jumps_rc += new_jump;
#ifdef DEBUG_INNER_STATISTICS
					inner_FG_jumps++;
#endif
				}

				*pLabels_rc = label_rc_00 + sum_jumps_rc; //& //1º store @ this must be recomputed according to the nof threads 
														  // also sth like: L[r][c] = (pjump_rc - &J[0][0] ) + jump_rc; //1º store @ this must be recomputed according to the nof threads 

			}  //end of if (jump_rc != 0 && I[r][c] == FG) {  
			label_rc_00++;  // @ this must be recomputed according to the nof threads
		}
	} // endof   for (int r = 0; r < N_ROWS; r++)  
#ifdef DEBUG_INNER_STATISTICS
	  //cout << "   # Final Labeling: Mean number of BG jumps per FG crit. cell: " << (inner_BG_jumps*1.0) / nof_non_critical_BG_pixels << endl;
	cout << "   # Final Labeling: Mean number of FG jumps per FG pixel: " << (inner_FG_jumps*1.0) / nof_non_critical_FG_pixels << endl;
#endif

	return total_nof_FG_labels;
}
//////////////////////

//////////////////////////////////////////////////////////////
//internal aux. functions 
//////////////////////////////////////////////////////////////

// @Note that init_borders() can be optimized . But it is negligible for big matrices 
void init_borders(const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres *lista, int numHilos, int trozo) {

	// four corners
	{
		RowColType r, c;

		r = 0;  c = 0;
		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

		if ((img_row[c] == FG))
		{
			ImageType R = img_row[c + 1];
			ImageType RD = img_row_fol[c + 1];
			ImageType D = img_row_fol[c];

			imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
			if (imgJump_row[c] == 0) {
				lista[0].padresI[lista[0].numPadres] = r;
				lista[0].padresJ[lista[0].numPadres] = c;
				lista[0].saltos[lista[0].numPadres] = 0;
				lista[0].numPadres++;
			}
		}
		else
		{
			imgJump_row[c] = 0; // undefined
		}

	}

	{
		RowColType r, c;
		r = 0;  c = N_COLS - 1;
		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 
		if ((img_row[c] == FG))
		{
			ImageType R = !img_row[c]; // img_row[c + 1];
			ImageType RD = !img_row[c]; //  img_row_fol[c + 1];
			ImageType D = img_row_fol[c];

			// L[r][c] = 0 + !(D)* tag_ind(r, c);
			imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
			if (imgJump_row[c] == 0) {
				lista[0].padresI[lista[0].numPadres] = r;
				lista[0].padresJ[lista[0].numPadres] = c;
				lista[0].saltos[lista[0].numPadres] = 0;
				lista[0].numPadres++;
			}
		}
		else
		{
			ImageType Le = img_row[c - 1];
			ImageType U = !img_row[c]; //  img_row_pre[c];

			imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
		}
	}

	{
		RowColType r, c;
		r = N_ROWS - 1;  c = 0;
		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

		if ((img_row[c] == FG))
		{
			ImageType R = img_row[c + 1];
			ImageType RD = !img_row[c]; //  img_row_fol[c + 1];
			ImageType D = !img_row[c]; //  img_row_fol[c];

			imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
			if (imgJump_row[c] == 0) {
				lista[numHilos - 1].padresI[lista[numHilos - 1].numPadres] = r;
				lista[numHilos - 1].padresJ[lista[numHilos - 1].numPadres] = c;
				lista[numHilos - 1].saltos[lista[numHilos - 1].numPadres] = 0;
				lista[numHilos - 1].numPadres++;
			}
		}
		else
		{
			ImageType Le = !img_row[c]; //  img_row[c - 1];
			ImageType U = img_row_pre[c];

			// L[r][c] = 0 + (U)* tag_ind(r, c);
			imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
		}

	}

	{
		RowColType r, c;
		r = N_ROWS - 1;  c = N_COLS - 1;
		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 
		if ((img_row[c] == FG))
		{
			imgJump_row[c] = 0; //undefined  
			lista[numHilos - 1].padresI[lista[numHilos - 1].numPadres] = r;
			lista[numHilos - 1].padresJ[lista[numHilos - 1].numPadres] = c;
			lista[numHilos - 1].saltos[lista[numHilos - 1].numPadres] = 0;
			lista[numHilos - 1].numPadres++;
		}
		else
		{
			ImageType Le = img_row[c - 1];
			ImageType U = img_row_pre[c];

			imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
		}

	} //end of four corners
	  //////////

	  // left and right cols
	for (RowColType r = 1; r < N_ROWS - 1; r++) {
		int id = r / trozo;

		if (id >(omp_get_num_threads() - 1)) id = id - 1;
		// Get rows pointers
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

		RowColType c;

		c = 0;
		if ((img_row[c] == FG))
		{
			ImageType R = img_row[c + 1];
			ImageType RD = img_row_fol[c + 1];
			ImageType D = img_row_fol[c];

			imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
			if (imgJump_row[c] == 0) {
				lista[id].padresI[lista[id].numPadres] = r;
				lista[id].padresJ[lista[id].numPadres] = c;
				lista[id].saltos[lista[id].numPadres] = 0;
				lista[id].numPadres++;
			}
		}
		else
		{
			ImageType Le = !img_row[c]; //  img_row[c - 1];
			ImageType U = img_row_pre[c];

			imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
		}

		c = N_COLS - 1;
		if ((img_row[c] == FG))
		{
			ImageType R = !img_row[c]; //  img_row[c + 1];
			ImageType RD = !img_row[c]; //  img_row_fol[c + 1];
			ImageType D = img_row_fol[c];

			imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
			if (imgJump_row[c] == 0) {
				lista[id].padresI[lista[id].numPadres] = r;
				lista[id].padresJ[lista[id].numPadres] = c;
				lista[id].saltos[lista[id].numPadres] = 0;
				lista[id].numPadres++;
			}
		}
		else
		{
			ImageType Le = img_row[c - 1];
			ImageType U = img_row_pre[c];

			// L[r][c] = 0 + (Le & U) * tag_ind(r, c);
			imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
		}
	}

	// up and bottom rows
	{
		// up row
		RowColType r;
		r = 0;
		// Get rows pointers @this has been extracted from the loop (be careful with omp variable scopes)
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

    //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE)
		for (RowColType c = 1; c < N_COLS - 1; c++) {
			if ((img_row[c] == FG))
			{
				ImageType R = img_row[c + 1];
				ImageType RD = img_row_fol[c + 1];
				ImageType D = img_row_fol[c];

				imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
				if (imgJump_row[c] == 0) {
					lista[0].padresI[lista[0].numPadres] = r;
					lista[0].padresJ[lista[0].numPadres] = c;
					lista[0].saltos[lista[0].numPadres] = 0;
					lista[0].numPadres++;
				}
			}
			else
			{
				ImageType Le = img_row[c - 1];
				ImageType U = !img_row[c]; //  img_row_pre[c];

				imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
			}

		} //endof for (RowColType c = 1; c < N_COLS - 1; c++)   up row
	}

	{
		// bottom row
		RowColType r;
		r = N_ROWS - 1;
		// Get rows pointers @this has been extracted from the loop (be careful with omp variable scopes)
		const uchar* const img_row = I.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + I.step.p[0]); //pointer to the following row
		const uchar* const img_row_pre = (uchar *)(((char *)img_row) - I.step.p[0]); //pointer to the previous row

		int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

													   //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE)
		for (RowColType c = 1; c < N_COLS - 1; c++) {
			if ((img_row[c] == FG))
			{
				ImageType R = img_row[c + 1];
				ImageType RD = !img_row[c]; //  img_row_fol[c + 1];
				ImageType D = !img_row[c]; //  img_row_fol[c];

				imgJump_row[c] = FG_lookup_table_3neighb(R, RD, D);
				if (imgJump_row[c] == 0) {
					lista[numHilos - 1].padresI[lista[numHilos - 1].numPadres] = r;
					lista[numHilos - 1].padresJ[lista[numHilos - 1].numPadres] = c;
					lista[numHilos - 1].saltos[lista[numHilos - 1].numPadres] = 0;
					lista[numHilos - 1].numPadres++;
				}
			}
			else
			{
				ImageType Le = img_row[c - 1];
				ImageType U = img_row_pre[c];

				// L[r][c] = 0 + (Le & U) * tag_ind(r, c);
				imgJump_row[c] = BG_lookup_table_2neighb(Le, U);
			}
		}  //endof for (RowColType c = 1; c < N_COLS - 1; c++)   bottom row
	}

}  //endof function

   /////////////////////////////////////////


   // lookup tables with 3 FG / 2 BG neighbors: 
RowColType FG_lookup_table_3neighb(RowColType Right, RowColType RiDo, RowColType Down) { //  priority for FG : +1, +N_COLS+1, + N_COLS
	RowColType table[8] = { 0, N_COLS, N_COLS + 1, N_COLS + 1,
		1, 1, 1, 1 }; // FG must be 1; BG 0
	int ind = (Right << 2) + (RiDo << 1) + Down;
	return table[ind];
}

RowColType BG_lookup_table_2neighb(RowColType Left, RowColType Up) { //  priority for BG : -1, -N_COLS
	RowColType table[4] = { -1, -1, -N_COLS, 0
	}; // FG must be 1; BG 0
	int ind = (Left << 1) + Up;
	return table[ind];
}

///////////////////////////////////////
int Jonly_transports_lots_of_searches(int R_SHIFT_SEARCH, int C_SHIFT_SEARCH, int R_SHIFT_NEW_ROOT, int C_SHIFT_NEW_ROOT,
	const cv::Mat1b &I, cv::Mat1i &J, struct listaPadres* listaGlobal) {

	int nof_FG_transports = 0;  //initial  value 
	int nof_BG_transports = 0;  //initial  value 
#ifdef DEBUG_INNER_STATISTICS
	int inner_FG_jumps = 0, inner_BG_jumps = 0;
	int inner_FG_crit_cells = 0;
	int inner_FG_crit_cells_backwards = 0;
#endif
	// Get row pointer to the most left upper pixel 
	int* const imgJump_row_origin = (int*)J.ptr<uint>(0); //sign is required here. 

														  // Because FG has more connectivity, it has less critical cells. It is preferable transporting false critical FG cells than BG ones
#ifdef DEBUG_INNER_STATISTICS
														  //#pragma omp parallel for schedule(_OMP_SCHEDULING, CHUNK_SIZE) reduction (+:flag_any_transport) reduction (+:inner_FG_jumps ) reduction (+:inner_BG_jumps ) reduction (+: inner_FG_crit_cells ) reduction (+:inner_FG_crit_cells_backwards) shared ( I)
#else
														  //#pragma omp parallel  reduction (+:nof_FG_transports)reduction (+:nof_BG_transports) shared ( I)
#endif
#pragma omp parallel reduction (+:nof_FG_transports)reduction (+:nof_BG_transports) shared ( I)
	{
		int id = omp_get_thread_num();
		struct listaPadres* lista = &listaGlobal[id];

		for (int k = 0; k < lista->numPadres; k++) {    // FG critical cells at the SOUTH (BOTTOM) border are considered as real critical cells

		// Get rows pointers
			int r = lista->padresI[k];
			int c = lista->padresJ[k];
			int* const imgJump_row = (int*)J.ptr<uint>(r); //sign is required here. 

			JumpType *pjump_rc = &(imgJump_row[c]);
			JumpType jump_rc = lista->saltos[k];

			if (jump_rc == 0) {  // 2º load  if it were a FG critical cell

				if (((c + C_SHIFT_SEARCH) < N_COLS) && ((r + R_SHIFT_SEARCH) < N_ROWS)) {   //(( @ NO BORDERS //%CHECK 1
					JumpType other_color_jump = *(pjump_rc + C_SHIFT_SEARCH + N_COLS * R_SHIFT_SEARCH); // 3º load  
					JumpType *pother_color_crit_cell = (pjump_rc + C_SHIFT_SEARCH + N_COLS * R_SHIFT_SEARCH) + other_color_jump;
					//now the *pother_color_crit_cell may not be zero , that is the BG area may fall into a previously deleted critical cells. Thus, inner jumps until a zero were found are necessary:
					JumpType jump_other_color_crit_cell = *pother_color_crit_cell;
#ifdef DEBUG_INNER_STATISTICS
					inner_FG_crit_cells++;
#endif
					JumpType *pnew_other_color_crit_cell = pother_color_crit_cell;

					if (jump_other_color_crit_cell != 0) {

						// %%%%  TRAVELLING ALONG BG CELLS:   
						JumpType sum_jump_other_color_crit_cell;

						// travel_inside_J  returns the destin of the travel (a pointer to the destin )
						// receives: a pointer to origin,  initial jump (that is *origin) , the total distance by reference (the sum from destin to origin)
						pnew_other_color_crit_cell = travel_inside_J(pother_color_crit_cell, jump_other_color_crit_cell, &sum_jump_other_color_crit_cell);

						*(pjump_rc + C_SHIFT_SEARCH + N_COLS * R_SHIFT_SEARCH) =
							sum_jump_other_color_crit_cell + other_color_jump; // optimizing for the next time it checks this FG sink //&
					}

					// %%%%  BG CRITICAL  CELL WAS FOUND :   I LOOK INTO ITS FG ADJ. CELLS: 

					if (pnew_other_color_crit_cell != (JumpType *)imgJump_row_origin) //(((@ NO_BORDERS_ADDITIONAL .  //%CHECK 2
																					  //@ it could be not necessary (because all the checking is done 3 lines below ) 
					{
						JumpType other_color_jump_backwards;
						JumpType *pother_color_jump_backwards;

						// checking borders for the other BG crit. cell we have just travelled to 
						///////

						JumpType *padj_to_new_other_color_crit_cell = pnew_other_color_crit_cell - C_SHIFT_SEARCH - N_COLS * R_SHIFT_SEARCH;
						int dist_between_BG_cell_and_origin = static_cast<int>(padj_to_new_other_color_crit_cell - ((JumpType *)imgJump_row_origin));

						if (
							((-R_SHIFT_SEARCH == -1) && (padj_to_new_other_color_crit_cell > (JumpType *)imgJump_row_origin))
							||
							((-C_SHIFT_SEARCH == -1) && (dist_between_BG_cell_and_origin % N_COLS) != (N_COLS - 1))
							)    //%CHECK 3
						{ //(((@ NO_BORDERS_ADDITIONAL 

						  // normal case: not in   borders:
							other_color_jump_backwards = *(pnew_other_color_crit_cell - C_SHIFT_SEARCH - N_COLS * R_SHIFT_SEARCH); // 4º load  
							pother_color_jump_backwards = (pnew_other_color_crit_cell - C_SHIFT_SEARCH - N_COLS * R_SHIFT_SEARCH) + other_color_jump_backwards;

							//now the *pother_color_jump_backwards may not be zero , that is the FG area may fall into a previously deleted critical cells. 
							//Thus, inner jumps until a zero were found are necessary:
							JumpType jump_other_color_jump_backwards = *pother_color_jump_backwards;
#ifdef DEBUG_INNER_STATISTICS
							inner_FG_crit_cells_backwards++;
#endif	
							JumpType *pnew_other_color_jump_backwards = pother_color_jump_backwards;
							JumpType *pprevious_other_color_jump_backwards = pother_color_jump_backwards; //&

							if (jump_other_color_jump_backwards != 0) {
								// %%%%  TRAVELLING ALONG FG CELLS:   
								JumpType sum_jump_other_color_jump_backwards;
								pnew_other_color_jump_backwards = travel_inside_J(pother_color_jump_backwards, jump_other_color_jump_backwards, &sum_jump_other_color_jump_backwards);

								*(pnew_other_color_crit_cell - C_SHIFT_SEARCH - N_COLS * R_SHIFT_SEARCH) =
									(sum_jump_other_color_jump_backwards + other_color_jump_backwards); // optimizing for the next time it checks this FG sink //&
							}

							// %%%%  MAKING THE TRANSPORT
							if (pnew_other_color_jump_backwards == pjump_rc) {  // do the transport:
																				// or  semitransport !!!:   

																				// checking the BG crit. cell that is "below" the FG crit. cell  :   //(((@ NO_BORDERS_ADDITIONAL
								if (((c - C_SHIFT_NEW_ROOT) < N_COLS) && ((r - R_SHIFT_NEW_ROOT) < N_ROWS)) { //(((@ NO BORDERS  //%CHECK 4a
																											  // the deleted BG crit. cell is written with:
									JumpType * pother_color_new_root_origin = (pjump_rc - C_SHIFT_NEW_ROOT - N_COLS * R_SHIFT_NEW_ROOT); // 5º load  
									JumpType sum_jump_other_color_new_root;
									JumpType * pnew_other_color_crit_cell_destin = travel_inside_J(pother_color_new_root_origin, *pother_color_new_root_origin, &sum_jump_other_color_new_root);

									JumpType new_value = static_cast<JumpType>(sum_jump_other_color_new_root + (pother_color_new_root_origin - pnew_other_color_crit_cell));	// 2º store with if 

									*pnew_other_color_crit_cell = new_value; // 2º store with if 
																			 // there can be considered transport only when the new value to be written in this crit. cell is not zero 
									nof_BG_transports += (new_value != 0);
								}

								// checking the FG crit. cell that is "above" the BG crit. cell :   //(((@ NO_BORDERS_ADDITIONAL
								JumpType *psearch_to_new_other_color_crit_cell = pnew_other_color_crit_cell + C_SHIFT_NEW_ROOT + N_COLS * R_SHIFT_NEW_ROOT;
								int dist_between_psearch_BG_cell_and_origin = static_cast<int>(psearch_to_new_other_color_crit_cell - ((JumpType *)imgJump_row_origin));

								if (
									((R_SHIFT_NEW_ROOT == -1) && (psearch_to_new_other_color_crit_cell > (JumpType *)imgJump_row_origin))
									||
									((C_SHIFT_NEW_ROOT == -1) && (dist_between_psearch_BG_cell_and_origin % N_COLS) != (N_COLS - 1))
									)
								{ //(((@ NO BORDERS //%CHECK 4b
								  //the deleted FG crit. cell is written with
									JumpType * pnew_root_origin = (pnew_other_color_crit_cell + C_SHIFT_NEW_ROOT + N_COLS * R_SHIFT_NEW_ROOT); // 5º load  
									JumpType sum_jump_new_root;
									JumpType * pnew_crit_cell_destin = travel_inside_J(pnew_root_origin, *pnew_root_origin, &sum_jump_new_root);
									JumpType new_value = static_cast<JumpType>(sum_jump_new_root - (pjump_rc - pnew_root_origin));

									*pjump_rc = new_value; //// 1º store with if ç
									lista->saltos[k] = new_value;
									// there can be considered transport only when the new value to be written in this crit. cell is not zero 
									nof_FG_transports += (new_value != 0);

								} //(((@ NO BORDERS

							}   // endof if (pnew_other_color_jump_backwards == pjump_rc) {  // do the transport:

						}  //end of if (  ((-R_SHIFT_SEARCH == -1) && (padj_to_new_other_color_crit_cell > (JumpType *)pjump_origin)) ...

					}   //endof if (((c + C_SHIFT_SEARCH) < N_COLS) && ((r + R_SHIFT_SEARCH) < N_ROWS)) 

				} // endof if (jump_rc == 0 && I[r][c] == FG) {  // 2º load  if it were a FG critical cell
			}
		} // endof   for (int r = 0; r < N_ROWS; r++)  
	}

#ifdef DEBUG_INNER_STATISTICS
	cout << "   + Transports: Mean number of BG jumps per FG crit. cell: " << (inner_BG_jumps*1.0) / inner_FG_crit_cells << endl;
	cout << "   + Transports: Mean number of FG jumps per FG crit. cell: " << (inner_FG_jumps*1.0) / inner_FG_crit_cells_backwards << endl;
#endif

	return (nof_FG_transports + nof_BG_transports);
}
//////////////////////
//////////////////////
// it returns the destin of the travel (a pointer to the destin )
// receives: a pointer to origin,  initial jump (that is *origin) , the total distance by reference (the sum from destin to origin)

JumpType * travel_inside_J(JumpType *porigin, JumpType jump, JumpType *psum_jump) {
	JumpType *pdestin;
	JumpType *pprevious_other_color_crit_cell = porigin; //&
	*psum_jump = jump;

	JumpType previous_jump = jump;//&
	do {
		//  update the new jumps:
		pdestin = (jump + pprevious_other_color_crit_cell);//&
		jump = *pdestin;
		*psum_jump += jump;
		// TODO@ optimize this, so that the jumps were stored in the deleted crit. cells.

		*pprevious_other_color_crit_cell = (previous_jump + jump);//&

		previous_jump = jump;//&
		pprevious_other_color_crit_cell = pdestin; //&
#ifdef DEBUG_INNER_STATISTICS
		inner_BG_jumps++;
#endif
	} while (jump != 0);
	*porigin = *psum_jump;   // optimizing 

	return pdestin;
}
///////////////////////////////////////



//========================================
#endif // !YACCLAB_LABELING_CCLHSF_H_