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

