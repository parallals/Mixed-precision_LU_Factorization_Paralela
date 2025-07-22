#pragma once

// --- MPF: Mixed‑precision Pre‑pivoting Factorization ---
// A [in/out] pointer to the matrix A
// N [in] size of the matrix A (N x N)
// r [in] panel size for mixed-precision factorization
// IPIV [out] array to store pivot indices (1-based global indexing)
void MPF(double *h_A, int N, int r, int *IPIV);