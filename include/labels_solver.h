// Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

// Union-find (UF) with path compression (PC)
class UfPC {
public:
    unsigned Merge(unsigned *P, unsigned i, unsigned j)
    {
        // findRoot(P, i)
        unsigned root(i);
        while (P[root] < root) {
            root = P[root];
        }
        if (i != j) {
            // findRoot(P, j)
            unsigned root_j(j);
            while (P[root_j] < root_j) {
                root_j = P[root_j];
            }
            if (root > root_j) {
                root = root_j;
            }
            // setRoot(P, j, root);
            while (P[j] < j) {
                unsigned t = P[j];
                P[j] = root;
                j = t;
            }
            P[j] = root;
        }
        // setRoot(P, i, root);
        while (P[i] < i) {
            unsigned t = P[i];
            P[i] = root;
            i = t;
        }
        P[i] = root;
        return root;
    }
    unsigned Flatten(unsigned *P, const unsigned length)
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length; ++i) {
            if (P[i] < i) {
                P[i] = P[P[i]];
            }
            else {
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    unsigned Merge(MemVector<unsigned>& P, unsigned i, unsigned j)
    {
        // findRoot(P, i)
        unsigned root(i);
        while (P[root] < root) {
            root = P[root];
        }
        if (i != j) {
            // findRoot(P, j)
            unsigned root_j(j);
            while (P[root_j] < root_j) {
                root_j = P[root_j];
            }
            if (root > root_j) {
                root = root_j;
            }
            // setRoot(P, j, root);
            while (P[j] < j) {
                unsigned t = P[j];
                P[j] = root;
                j = t;
            }
            P[j] = root;
        }
        // setRoot(P, i, root);
        while (P[i] < i) {
            unsigned t = P[i];
            P[i] = root;
            i = t;
        }
        P[i] = root;
        return root;
    }
    unsigned Flatten(MemVector<unsigned>& P, const unsigned length)
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length; ++i) {
            if (P[i] < i) {
                P[i] = P[P[i]];
            }
            else {
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }
};

// Interleaved Rem algorithm with splicing
class RemSp {
public:
    unsigned Merge(unsigned *P, unsigned i, unsigned j)
    {
        unsigned rootI(i), rootJ(j);

        while (P[rootI] != P[rootJ]) {
            if (P[rootI] > P[rootJ]) {
                if (rootI == P[rootI]) {
                    P[rootI] = P[rootJ];
                    return P[rootI];
                }
                unsigned z = P[rootI];
                P[rootI] = P[rootJ];
                rootI = z;
            }
            else {
                if (rootJ == P[rootJ]) {
                    P[rootJ] = P[rootI];
                    return P[rootI];
                }
                unsigned z = P[rootJ];
                P[rootJ] = P[rootI];
                rootJ = z;
            }
        }
        return P[rootI];
    }
    unsigned Flatten(unsigned *P, const unsigned length)
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length; ++i) {
            if (P[i] < i) {
                P[i] = P[P[i]];
            }
            else {
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    unsigned Merge(MemVector<unsigned>& P, unsigned i, unsigned j)
    {
        unsigned rootI(i), rootJ(j);

        while (P[rootI] != P[rootJ]) {
            if (P[rootI] > P[rootJ]) {
                if (rootI == P[rootI]) {
                    P[rootI] = P[rootJ];
                    return P[rootI];
                }
                unsigned z = P[rootI];
                P[rootI] = P[rootJ];
                rootI = z;
            }
            else {
                if (rootJ == P[rootJ]) {
                    P[rootJ] = P[rootI];
                    return P[rootI];
                }
                unsigned z = P[rootJ];
                P[rootJ] = P[rootI];
                rootJ = z;
            }
        }
        return P[rootI];
    }
    unsigned Flatten(MemVector<unsigned>& P, const unsigned length)
    {
        unsigned k = 1;
        for (unsigned i = 1; i < length; ++i) {
            if (P[i] < i) {
                P[i] = P[P[i]];
            }
            else {
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }
};

#define REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(algorithm) \
    REGISTER_SOLVER(algorithm, UfPC) \
    REGISTER_SOLVER(algorithm, RemSp) \

#endif // !YACCLAB_LABELS_SOLVER_H_