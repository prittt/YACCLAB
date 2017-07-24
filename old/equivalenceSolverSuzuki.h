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

#pragma once
#include <vector>
#include "memory_tester.h"

//class UnionFind {
//};

// "STANDARD" VERSION
//Find the root of the tree of node i
template<typename LabelT>
inline static
LabelT findRoot(const LabelT *P, LabelT i)
{
    LabelT root = i;
    while (P[root] < root) {
        root = P[root];
    }
    return root;
}

//Make all nodes in the path of node i point to root
template<typename LabelT>
inline static
void setRoot(LabelT *P, LabelT i, LabelT root)
{
    while (P[i] < i) {
        LabelT j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
template<typename LabelT>
inline static
LabelT find(LabelT *P, LabelT i)
{
    LabelT root = findRoot(P, i);
    setRoot(P, i, root);
    return root;
}

//unite the two trees containing nodes i and j and return the new root
template<typename LabelT>
inline static
LabelT set_union(LabelT *P, LabelT i, LabelT j)
{
    LabelT root = findRoot(P, i);
    if (i != j) {
        LabelT rootj = findRoot(P, j);
        if (root > rootj) {
            root = rootj;
        }
        setRoot(P, j, root);
    }
    setRoot(P, i, root);
    return root;
}

template<typename LabelT>
inline static
LabelT SetUnion(LabelT *P, LabelT i, LabelT j)
{
    LabelT root(i);
    while (P[rooti] < root) {
        root = P[root];
    }
    if (i != j) {
        LabelT root_j(j);
        while (P[root_j] < root_j) {
            root_j = P[root_j];
        }
        if (root > root_j) {
            root = root_j;
        }
        //setRoot(P, j, root);
        while (P[j] < j) {
            LabelT t = P[j];
            P[j] = root_j;
            j = t;
        }
        P[j] = root_j;
    }
    //setRoot(P, i, root);
    while (P[i] < i) {
        LabelT t = P[i];
        P[i] = root;
        i = t;
    }
    P[i] = root;
    return root;
}

template<typename LabelT>
inline static
LabelT MergeRemSp(LabelT *P, LabelT i, LabelT j)
{
    LabelT rootI(i), rootJ(j);

    while (P[rootI] != P[rootJ]) {
        if (P[rootI] > P[rootJ]) {
            if (rootI == P[rootI]) {
                P[rootI] = P[rootJ];
                return P[rootI];
            }
            LabelT z = P[rootI];
            P[rootI] = P[rootJ];
            rootI = z;
        }
        else {
            if (rootJ == P[rootJ]) {
                P[rootJ] = P[rootI];
                return P[rootI];
            }
            LabelT z = P[rootJ];
            P[rootJ] = P[rootI];
            rootJ = z;
        }
    }
    return P[rootI];
}

//Flatten the Union Find tree and relabel the components
template<typename LabelT>
inline static
LabelT flattenL(LabelT *P, LabelT length)
{
    LabelT k = 1;
    for (LabelT i = 1; i < length; ++i) {
        if (P[i] < i) {
            P[i] = P[P[i]];
        }
        else {
            P[i] = k; k = k + 1;
        }
    }
    return k;
}
// "STANDARD" VERSION
//
//
// "MEMORY TEST" VERSION
//Find the root of the tree of node i
template<typename LabelT>
inline static
LabelT findRoot(MemVector<LabelT> &P, LabelT i)
{
    LabelT root = i;
    while (P[root] < root) {
        root = P[root];
    }
    return root;
}

//Make all nodes in the path of node i point to root
template<typename LabelT>
inline static
void setRoot(MemVector<LabelT> &P, LabelT i, LabelT root)
{
    while (P[i] < i) {
        LabelT j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
template<typename LabelT>
inline static
LabelT find(MemVector<LabelT> &P, LabelT i)
{
    LabelT root = findRoot(P, i);
    setRoot(P, i, root);
    return root;
}

//unite the two trees containing nodes i and j and return the new root
template<typename LabelT>
inline static
LabelT set_union(MemVector<LabelT> &P, LabelT i, LabelT j)
{
    LabelT root = findRoot(P, i);
    if (i != j) {
        LabelT rootj = findRoot(P, j);
        if (root > rootj) {
            root = rootj;
        }
        setRoot(P, j, root);
    }
    setRoot(P, i, root);
    return root;
}

//Flatten the Union Find tree and relabel the components
template<typename LabelT>
inline static
LabelT flattenL(MemVector<LabelT> &P, LabelT length)
{
    LabelT k = 1;
    for (LabelT i = 1; i < length; ++i) {
        if (P[i] < i) {
            P[i] = P[P[i]];
        }
        else {
            P[i] = k; k = k + 1;
        }
    }
    return k;
}
// "MEMORY TEST" VERSION