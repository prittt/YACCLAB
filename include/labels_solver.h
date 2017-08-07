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

#ifndef YACCLAB_LABELS_SOLVER_H_
#define YACCLAB_LABELS_SOLVER_H_

#include "register.h"
#include "memory_tester.h"

// Union-find (UF)
class UF {
// Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
	static void Alloc(unsigned max_length) {
		P_ = new unsigned[max_length];
	}
	static void Dealloc() {
		delete[] P_;
	}
	static void Setup() {
		P_[0] = 0;// First label is for background pixels
		length_ = 1;
	}
	static unsigned NewLabel() {
		P_[length_] = length_;
		return length_++;
	}
	static unsigned GetLabel(unsigned index) {
		return P_[index];
	}

	// Basic functions of the UF solver required only by the Light-Speed Labeling Algorithms:
	// - "UpdateTable" updates equivalences array without performing "Find" operations
	// - "FindRoot" finds the root of the tree of node i (for the other algorithms it is 
	//		already included in the "Merge" and "Flatten" functions).
	static void UpdateTable(unsigned e, unsigned r) {
		P_[e] = r;
	}
	static unsigned FindRoot(unsigned root) {
		while (P_[root] < root) {
			root = P_[root];
		}
		return root;
	}

	static unsigned Merge(unsigned i, unsigned j)
	{
		// FindRoot(i)
		while (P_[i] < i) {
			i = P_[i];
		}

		// FindRoot(j)
		while (P_[j] < j) {
			j = P_[j];
		}

		if (i < j)
			return P_[j] = i;
		return P_[i] = j;
	}
	static unsigned Flatten()
	{
		unsigned k = 1;
		for (unsigned i = 1; i < length_; ++i) {
			if (P_[i] < i) {
				P_[i] = P_[P_[i]];
			}
			else {
				P_[i] = k;
				k = k + 1;
			}
		}
		return k;
	}

	/***************************************************************/

	static void MemAlloc(unsigned max_length)
	{
		mem_P_ = MemVector<unsigned>(max_length);
	}
	static void MemDealloc() {}
	static void MemSetup() {
		mem_P_[0] = 0;	 // First label is for background pixels
		length_ = 1;
	}
	static unsigned MemNewLabel() {
		mem_P_[length_] = length_;
		return length_++;
	}
	static unsigned MemGetLabel(unsigned index) {
		return mem_P_[index];
	}

	static double MemTotalAccesses() {
		return mem_P_.GetTotalAccesses();
	}

	// Basic functions of the UF solver required only by the Light-Speed Labeling Algorithms:
	// - "MemUpdateTable" updates equivalences array without performing "MemFind" operations
	// - "MemFindRoot" finds the root of the tree of node i (for the other algorithms it is 
	//		already included in the "MemMerge" and "MemFlatten" functions).
	static void MemUpdateTable(unsigned e, unsigned r) {
		mem_P_[e] = r;
	}
	static unsigned MemFindRoot(unsigned root) {
		while (mem_P_[root] < root) {
			root = mem_P_[root];
		}
		return root;
	}

	static unsigned MemMerge(unsigned i, unsigned j)
	{
		// FindRoot(i)
		while (mem_P_[i] < i) {
			i = mem_P_[i];
		}

		// FindRoot(j)
		while (mem_P_[j] < j) {
			j = mem_P_[j];
		}

		if (i < j)
			return mem_P_[j] = i;
		return mem_P_[i] = j;
	}
    static unsigned MemFlatten()
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length_; ++i) {
            if (mem_P_[i] < i) {
                mem_P_[i] = mem_P_[mem_P_[i]];
            }
            else {
                mem_P_[i] = k;
                k = k + 1;
            }
        }
        return k;
    }
private:
	static unsigned *P_;
	static unsigned length_;
	static MemVector<unsigned> mem_P_;
};

// Union-Find (UF) with path compression (PC) as in:
// Two Strategies to Speed up Connected Component Labeling Algorithms
// Kesheng Wu, Ekow Otoo, Kenji Suzuki
class UFPC {
// Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
	static void Alloc(unsigned max_length) {
		P_ = new unsigned[max_length];
	}
	static void Dealloc() {
		delete[] P_;
	}
	static void Setup() {
		P_[0] = 0;	 // First label is for background pixels
		length_ = 1;
	}
	static unsigned NewLabel() {
		P_[length_] = length_;
		return length_++;
	}
	static unsigned GetLabel(unsigned index) {
		return P_[index];
	}

	/*static void SetRoot(unsigned i, unsigned root)
	{
		while (P_[i] < i) {
			unsigned j = P_[i];
			P_[i] = root;
			i = j;
		}
		P_[i] = root;
	}*/
	/*static unsigned FindRoot(unsigned i)
	{
		// TODO: check if the following while is really necessary. 
		// Remove it also from "Merge" in case!!
		while (P_[i] < i) {
			i = P_[i];
		}
		return i;
	}*/

	static unsigned Merge(unsigned i, unsigned j)
	{
		// FindRoot(i)
		unsigned root(i);
		while (P_[root] < root) {
			root = P_[root];
		}
		if (i != j) {
			// FindRoot(j)
			unsigned root_j(j);
			while (P_[root_j] < root_j) {
				root_j = P_[root_j];
			}
			if (root > root_j) {
				root = root_j;
			}
			// SetRoot(j, root);
			while (P_[j] < j) {
				unsigned t = P_[j];
				P_[j] = root;
				j = t;
			}
			P_[j] = root;
		}
		// SetRoot(i, root);
		while (P_[i] < i) {
			unsigned t = P_[i];
			P_[i] = root;
			i = t;
		}
		P_[i] = root;
		return root;
	}
	static unsigned Flatten()
	{
		unsigned k = 1;
		for (unsigned i = 1; i < length_; ++i) {
			if (P_[i] < i) {
				P_[i] = P_[P_[i]];
			}
			else {
				P_[i] = k;
				k = k + 1;
			}
		}
		return k;
	}

	/***************************************************************/

	static void MemAlloc(unsigned max_length) {
		mem_P_ = MemVector<unsigned>(max_length);
	}
	static void MemDealloc() {}
	static void MemSetup() {
		mem_P_[0] = 0;	 // First label is for background pixels
		length_ = 1;
	}
	static unsigned MemNewLabel() {
		mem_P_[length_] = length_;
		return length_++;
	}
	static unsigned MemGetLabel(unsigned index) {
		return mem_P_[index];
	}

	static double MemTotalAccesses() {
		return mem_P_.GetTotalAccesses();
	}

	/*static void MemSetRoot(unsigned i, unsigned root)
	{
		while (mem_P_[i] < i) {
			unsigned j = mem_P_[i];
			mem_P_[i] = root;
			i = j;
		}
		mem_P_[i] = root;
	}*/
	/*static unsigned MemFindRoot(unsigned i)
	{
		while (mem_P_[i] < i) {
			i = mem_P_[i];
		}
		return i;
	}*/

	static unsigned MemMerge(unsigned i, unsigned j)
	{
		// FindRoot(i)
		unsigned root(i);
		while (mem_P_[root] < root) {
			root = mem_P_[root];
		}
		if (i != j) {
			// FindRoot(j)
			unsigned root_j(j);
			while (mem_P_[root_j] < root_j) {
				root_j = mem_P_[root_j];
			}
			if (root > root_j) {
				root = root_j;
			}
			// SetRoot(j, root);
			while (mem_P_[j] < j) {
				unsigned t = mem_P_[j];
				mem_P_[j] = root;
				j = t;
			}
			mem_P_[j] = root;
		}
		// SetRoot(i, root);
		while (mem_P_[i] < i) {
			unsigned t = mem_P_[i];
			mem_P_[i] = root;
			i = t;
		}
		mem_P_[i] = root;
		return root;
	}
	static unsigned MemFlatten()
	{
		unsigned k = 1;
		for (unsigned i = 1; i < length_; ++i) {
			if (mem_P_[i] < i) {
				mem_P_[i] = mem_P_[mem_P_[i]];
			}
			else {
				mem_P_[i] = k;
				k = k + 1;
			}
		}
		return k;
	}

private:
	static unsigned *P_;
	static unsigned length_;
	static MemVector<unsigned> mem_P_;
};

// Interleaved Rem algorithm with SPlicing (SP) as in: 
// A New Parallel Algorithm for Two - Pass Connected Component Labeling
// S Gupta, D Palsetia, MMA Patwary
class RemSP {
// Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8)
public:
	static void Alloc(unsigned max_length) {
		P_ = new unsigned[max_length];
	}
	static void Dealloc() {
		delete[] P_;
	}
	static void Setup() {
		P_[0] = 0;	 // First label is for background pixels
		length_ = 1;
	}
	static unsigned NewLabel() {
		P_[length_] = length_;
		return length_++;
	}
	static unsigned GetLabel(unsigned index) {
		return P_[index];
	}

	static unsigned Merge(unsigned i, unsigned j)
	{
		unsigned root_i(i), root_j(j);

		while (P_[root_i] != P_[root_j]) {
			if (P_[root_i] > P_[root_j]) {
				if (root_i == P_[root_i]) {
					P_[root_i] = P_[root_j];
					return P_[root_i];
				}
				unsigned z = P_[root_i];
				P_[root_i] = P_[root_j];
				root_i = z;
			}
			else {
				if (root_j == P_[root_j]) {
					P_[root_j] = P_[root_i];
					return P_[root_i];
				}
				unsigned z = P_[root_j];
				P_[root_j] = P_[root_i];
				root_j = z;
			}
		}
		return P_[root_i];
	}
	static unsigned Flatten()
	{
		unsigned k = 1;
		for (unsigned i = 1; i < length_; ++i) {
			if (P_[i] < i) {
				P_[i] = P_[P_[i]];
			}
			else {
				P_[i] = k;
				k = k + 1;
			}
		}
		return k;
	}

	/***************************************************************/

	static void MemAlloc(unsigned max_length) {
		mem_P_ = MemVector<unsigned>(max_length);
	}
	static void MemDealloc() {}
	static void MemSetup() {
		mem_P_[0] = 0;	 // First label is for background pixels
		length_ = 1;
	}
	static unsigned MemNewLabel() {
		mem_P_[length_] = length_;
		return length_++;
	}
	static unsigned MemGetLabel(unsigned index) {
		return mem_P_[index];
	}

	static double MemTotalAccesses() {
		return mem_P_.GetTotalAccesses();
	}

	static unsigned MemMerge(unsigned i, unsigned j)
	{
		unsigned root_i(i), root_j(j);
		while (mem_P_[root_i] != mem_P_[root_j]) {
			if (mem_P_[root_i] > mem_P_[root_j]) {
				if (root_i == mem_P_[root_i]) {
					mem_P_[root_i] = mem_P_[root_j];
					return mem_P_[root_i];
				}
				unsigned z = mem_P_[root_i];
				mem_P_[root_i] = mem_P_[root_j];
				root_i = z;
			}
			else {
				if (root_j == mem_P_[root_j]) {
					mem_P_[root_j] = mem_P_[root_i];
					return mem_P_[root_i];
				}
				unsigned z = mem_P_[root_j];
				mem_P_[root_j] = mem_P_[root_i];
				root_j = z;
			}
		}
		return mem_P_[root_i];
	}
	static unsigned MemFlatten()
	{
		unsigned k = 1;
		for (unsigned i = 1; i < length_; ++i) {
			if (mem_P_[i] < i) {
				mem_P_[i] = mem_P_[mem_P_[i]];
			}
			else {
				mem_P_[i] = k;
				k = k + 1;
			}
		}
		return k;
	}

private:
	static unsigned *P_;
	static unsigned length_;
	static MemVector<unsigned> mem_P_;
};

// Three Table Array as in: 
// A Run-Based Two-Scan Labeling Algorithm
// Lifeng He, Yuyan Chao, Kenji Suzuki
class TTA {
	// Maximum number of labels (included background) = 2^(sizeof(unsigned) x 8) - 1:
	// the special value "-1" for next_ table array has been replace with UINT_MAX
public:
	static void Alloc(unsigned max_length) {
		rtable_ = new unsigned[max_length];
		next_ = new unsigned[max_length];
		tail_ = new unsigned[max_length];
	}
	static void Dealloc() {
		delete[] rtable_;
		delete[] next_;
		delete[] tail_;
	}
	static void Setup() {
		rtable_[0] = 0; 
		length_ = 1;
	}
	static unsigned NewLabel() {
		rtable_[length_] = length_;
		next_[length_] = UINT_MAX;
		tail_[length_] = length_;
		return length_++;
	}
	static unsigned GetLabel(unsigned index) {
		return rtable_[index];
	}

	// Basic functions of the TTA solver required only by the Light-Speed Labeling Algorithms:
	// - "UpdateTable" updates equivalences tables without performing "Find" operations
	// - "FindRoot" finds the root of the tree of node i (for the other algorithms it is 
	//		already included in the "Merge" and "Flatten" functions).
	static void UpdateTable(unsigned u, unsigned v) 
	{
		if (u < v) {
			unsigned i = v;
			while (i != UINT_MAX) {
				rtable_[i] = u;
				i = next_[i];
			}
			next_[tail_[u]] = v;
			tail_[u] = tail_[v];
		}
		else if (u > v) {
			unsigned i = u;
			while (i != UINT_MAX) {
				rtable_[i] = v;
				i = next_[i];
			}
			next_[tail_[v]] = u;
			tail_[v] = tail_[u];
		}
	}
	static unsigned FindRoot(unsigned i)
	{
		return rtable_[i];
	}

	static unsigned Merge(unsigned u, unsigned v)
	{
		// FindRoot(u);
		u = rtable_[u];
		// FindRoot(v);
		v = rtable_[v];

		if (u < v) {
			unsigned i = v;
			while (i != UINT_MAX) {
				rtable_[i] = u;
				i = next_[i];
			}
			next_[tail_[u]] = v;
			tail_[u] = tail_[v];
			return u;
		}
		else if (u > v) {
			unsigned i = u;
			while (i != UINT_MAX) {
				rtable_[i] = v;
				i = next_[i];
			}
			next_[tail_[v]] = u;
			tail_[v] = tail_[u];
			return v;
		}

		return u;  // equal to v
	}
	static unsigned Flatten()
	{
		unsigned cur_label = 1;
		for (unsigned k = 1; k < length_; k++) {
			if (rtable_[k] == k) {
				cur_label++;
				rtable_[k] = cur_label;
			}
			else
				rtable_[k] = rtable_[rtable_[k]];
		}

		return cur_label;
	}

	/***************************************************************/

	static void MemAlloc(unsigned max_length) {
		mem_rtable_ = MemVector<unsigned>(max_length);
		mem_next_ = MemVector<unsigned>(max_length);
		mem_tail_ = MemVector<unsigned>(max_length);
	}
	static void MemDealloc() {}
	static void MemSetup() {
		mem_rtable_[0] = 0;
		length_ = 1;
	}
	static unsigned MemNewLabel() {
		mem_rtable_[length_] = length_;
		mem_next_[length_] = UINT_MAX;
		mem_tail_[length_] = length_;
		return length_++;
	}
	static unsigned MemGetLabel(unsigned index) {
		return mem_rtable_[index];
	}

	static double MemTotalAccesses() {
		return mem_rtable_.GetTotalAccesses() +
			mem_next_.GetTotalAccesses() +
			mem_tail_.GetTotalAccesses();
	}

	// Basic functions of the TTA solver required only by the Light-Speed Labeling Algorithms:
	// - "MemUpdateTable" updates equivalences tables without performing "MemFind" operations
	// - "MemFindRoot" finds the root of the tree of node i (for the other algorithms it is 
	//		already included in the "MemMerge" and "MemFlatten" functions).
	static void MemUpdateTable(unsigned u, unsigned v)
	{
		if (u < v) {
			unsigned i = v;
			while (i != UINT_MAX) {
				mem_rtable_[i] = u;
				i = mem_next_[i];
			}
			mem_next_[mem_tail_[u]] = v;
			mem_tail_[u] = mem_tail_[v];
		}
		else if (u > v) {
			unsigned i = u;
			while (i != UINT_MAX) {
				mem_rtable_[i] = v;
				i = mem_next_[i];
			}
			mem_next_[mem_tail_[v]] = u;
			mem_tail_[v] = mem_tail_[u];
		}
	}
	static unsigned MemFindRoot(unsigned i)
	{
		return mem_rtable_[i];
	}

	static unsigned MemMerge(unsigned u, unsigned v)
	{
		// FindRoot(u);
		u = mem_rtable_[u];
		// FindRoot(v);
		v = mem_rtable_[v];

		if (u < v) {
			unsigned i = v;
			while (i != UINT_MAX) {
				mem_rtable_[i] = u;
				i = mem_next_[i];
			}
			mem_next_[mem_tail_[u]] = v;
			mem_tail_[u] = mem_tail_[v];
			return u;
		}
		else if (u > v) {
			unsigned i = u;
			while (i != UINT_MAX) {
				mem_rtable_[i] = v;
				i = mem_next_[i];
			}
			mem_next_[mem_tail_[v]] = u;
			mem_tail_[v] = mem_tail_[u];
			return v;
		}

		return u;  // equal to v
	}
	static unsigned MemFlatten()
	{
		// In order to renumber and count the labels: is it really necessary? 
		unsigned cur_label = 1;
		for (unsigned k = 1; k < length_; k++) {
			if (mem_rtable_[k] == k) {
				cur_label++;
				mem_rtable_[k] = cur_label;
			}
			else
				mem_rtable_[k] = mem_rtable_[mem_rtable_[k]];
		}

		return cur_label;
	}

private:
	static unsigned *rtable_;
	static unsigned *next_;
	static unsigned *tail_;
	static unsigned length_;
	static MemVector<unsigned> mem_rtable_;
	static MemVector<unsigned> mem_next_;
	static MemVector<unsigned> mem_tail_;

};

#define REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(algorithm) \
    REGISTER_SOLVER(algorithm, UF) \
	REGISTER_SOLVER(algorithm, UFPC) \
    REGISTER_SOLVER(algorithm, RemSP) \
	REGISTER_SOLVER(algorithm, TTA) \

#endif // !YACCLAB_LABELS_SOLVER_H_