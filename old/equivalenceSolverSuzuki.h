// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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