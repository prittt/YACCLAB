// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labels_solver.h"

// Union-Find (UF):
unsigned *UF::P_;
unsigned UF::length_;
MemVector<unsigned> UF::mem_P_;

// Union-Find (UF) with path compression (PC):
unsigned *UFPC::P_;
unsigned UFPC::length_;
MemVector<unsigned> UFPC::mem_P_;

// Interleaved Rem algorithm with SPlicing (SP):
unsigned *RemSP::P_;
unsigned RemSP::length_;
MemVector<unsigned> RemSP::mem_P_;

// Three Table Array (TTA):
unsigned *TTA::rtable_;
unsigned *TTA::next_;
unsigned *TTA::tail_;
unsigned TTA::length_;
MemVector<unsigned> TTA::mem_rtable_;
MemVector<unsigned> TTA::mem_next_;
MemVector<unsigned> TTA::mem_tail_;

