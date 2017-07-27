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

#include "labels_solver.h"

// Union-Find (UF):
unsigned *UF::P_;
unsigned *UF::A_;
unsigned UF::length_;
MemVector<unsigned> UF::mem_P_;
MemVector<unsigned> UF::mem_A_;

// Union-Find (UF) with path compression (PC):
unsigned *UF_PC::P_;
unsigned UF_PC::length_;
MemVector<unsigned> UF_PC::mem_P_;

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

