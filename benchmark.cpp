#include <iostream>
#include <iomanip>
#include <fstream>
#include <lapacke.h>
#include <vector>
#include "MPF.h"
#include <chrono>
#include <cstring>
#include <cblas.h>
#include <random>

using namespace std;

void print_sqrMatrix(const char *msg, double *mat, int n, bool verbose = true) {
    if (verbose && n < 10) {
        cout << msg << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << mat[j * n + i] << " "; 
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_LU(const double *lu, int n, bool verbose = true) {
    if (verbose && n < 10) {
        // Print L
        cout << "L matrix:" << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j)
                    cout << lu[j * n + i] << " ";
                else if (i == j)
                    cout << "1 ";
                else
                    cout << "0 ";
            }
            cout << endl;
        }
        cout << endl;

        // Print U
        cout << "U matrix:" << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i <= j)
                    cout << lu[j * n + i] << " "; 
                else
                    cout << "0 ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void get_LU(const double *A, double *L, double *U, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                L[j * n + i] = A[j * n + i]; 
                U[j * n + i] = 0.0;
            } else if (i == j) {
                L[j * n + i] = 1.0;
                U[j * n + i] = A[j * n + i];
            } else {
                L[j * n + i] = 0.0;
                U[j * n + i] = A[j * n + i];
            }
        }
    }
}

void multiply_sqrMatrices(const double *A, const double *B, double *C, int n) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, n, n,
        1.0, A, n, B, n, 0.0, C, n);
}

void row_permute(double *A, const int *ipiv, int n) {
    for (int i = n - 1; i >= 0; --i) {
        int piv = ipiv[i] - 1; // Convert to 0-based
        if (piv != i) {
            // Swap rows i and piv
            for (int j = 0; j < n; ++j) {
                swap(A[j * n + i], A[j * n + piv]); // Column-major: A[col * lda + row]
            }
        }
    }
}

bool check_sqrMatrix_equality(double *A, double *B, int n, double tol = 1e-10) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            return false;
        }
    }
    return true;
}


void generate_random_matrix(double *matrix, int n, double sparsity = 0.0, int seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dis(0.1, 10.0);
    uniform_real_distribution<double> sparse_dis(0.0, 1.0);
    
    // Generate random matrix elements
    for (int i = 0; i < n * n; i++) {
        if (sparsity > 0.0 && sparse_dis(gen) < sparsity) {
            matrix[i] = 0.0;  // Make element sparse, adds zeros if sparsity is set higher
        } else {
            matrix[i] = dis(gen);
        }
    }
    
    // Make matrix slightly diagonally dominant for better numerical stability
    for (int i = 0; i < n; i++) {
        matrix[i * n + i] += n * 0.1;  // Add to diagonal (ensure non-zero)
    }
}

bool check_correctitude(double *A, double *Data, int ipiv[], int n, bool verbose = false) {
    // if --no-check isn't set, we check correctness with this vvv
    // it takes a while since it copies and multiplies matrices, so we advice to use it with small matrices. 

    // get l u
    double *L = new double[n * n];
    double *U = new double[n * n];
    get_LU(Data, L, U, n);

    print_LU(Data, n, verbose);
    
    // get L*U
    double *LU = new double[n * n];
    multiply_sqrMatrices(L, U, LU, n);

    print_sqrMatrix("LU matrix:", LU, n, verbose);

    double *PLU = new double[n * n];

    // Initialize PLU with LU
    memcpy(PLU, LU, n * n * sizeof(double));
    // Apply row permutations based on pivot indices
    row_permute(PLU, ipiv, n);

    print_sqrMatrix("PLU matrix:", PLU, n, verbose);
    // Check if PLU is equal to A (original matrix)
    bool correctitude = check_sqrMatrix_equality(A, PLU, n);
    if (verbose) {
        cout << "Correctitude: " << (correctitude ? "True" : "False") << endl;
    }
    delete[] L;
    delete[] U;
    delete[] LU;
    delete[] PLU;

    return correctitude;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " start_size max_size [step=2] [function=exp] [sparsity=0.0] [-v] [--no-check]" << endl;
        cout << "  start_size: starting matrix size" << endl;
        cout << "  max_size: maximum matrix size" << endl;
        cout << "  step: step size for matrix generation (default: 2)" << endl;
        cout << "  function: 'exp' for exponential growth, 'lin' for linear growth (default: exp)" << endl;
        cout << "  sparsity: fraction of zeros in the matrix (0.0 = dense, 0.9 = 90% zeros, default: 0.0)" << endl;
        cout << "  -v: verbose output" << endl;
        cout << "  --no-check: skip correctness checking" << endl;
        return -1;
    }

    bool verbose = false;
    bool check_correct = true;
    int start_size = atoi(argv[1]);
    int max_size = atoi(argv[2]);
    int step = 2;
    bool exp_step_function = true;
    double sparsity = 0.0;
    
    // Parsing
    if (argc > 3 && isdigit(argv[3][0])) {
        step = atoi(argv[3]);
        if (step <= 0) {
            cout << "Invalid step: " << step << endl;
            return -1;
        }
    }
    
    if (argc > 4 && (string(argv[4]) == "exp" || string(argv[4]) == "lin")) {
        if (string(argv[4]) == "lin") {
            exp_step_function = false;
        }
    }
    
    if (argc > 5 && (argv[5][0] == '0' || argv[5][0] == '1' || argv[5][0] == '.')) {
        sparsity = atof(argv[5]);
        if (sparsity < 0.0 || sparsity >= 1.0) {
            cout << "Invalid sparsity: " << sparsity << ". Must be in [0.0, 1.0)." << endl;
            return -1;
        }
    }
    
    // Parse flags
    for (int i = 3; i < argc; ++i) {
        if (string(argv[i]) == "-v") {
            verbose = true;
        } else if (string(argv[i]) == "--no-check") {
            check_correct = false;
        }
    }

    if (start_size <= 0 || max_size < start_size) {
        cout << "Invalid matrix size(s)" << endl;
        return -1;
    }

    cout << "Configuration:" << endl;
    cout << "  Size range: " << start_size << " to " << max_size << endl;
    cout << "  Step: " << step << " (" << (exp_step_function ? "exponential" : "linear") << ")" << endl;
    cout << "  Sparsity: " << (sparsity * 100) << "%" << endl;
    cout << "  Verbose: " << (verbose ? "yes" : "no") << endl;
    cout << "  Check correctness: " << (check_correct ? "yes" : "no") << endl;
    cout << endl;

    ofstream csv("benchmark_times.csv");
    csv << "matrix_size,mpf_time,lapack_time\n" << fixed << setprecision(10);

    // Generate test sizes based on function type (matching matrix_generator logic)
    vector<int> test_sizes;
    int size = start_size;
    while (size <= max_size) {
        test_sizes.push_back(size);
        
        if (exp_step_function) {
            size *= step;
        } else {
            size += step;
        }
    }
    
    cout << "Will test " << test_sizes.size() << " matrix sizes:" << endl;
    for (int i = 0; i < test_sizes.size(); i++) {
        cout << "  " << test_sizes[i] << "x" << test_sizes[i];
        if (i < test_sizes.size() - 1) cout << ", ";
    }
    cout << endl << endl;

    for (int test_idx = 0; test_idx < test_sizes.size(); test_idx++) {
        int n = test_sizes[test_idx];
        
        cout << "Processing matrix " << (test_idx + 1) << " of " << test_sizes.size() 
             << ", size: " << n << "x" << n << endl;

        // Generate random matrix
        double *data_original = new double[n * n];
        generate_random_matrix(data_original, n, sparsity, 42 + test_idx);  // Use sparsity parameter and different seed per matrix

        if (verbose) {
            cout << "Generated matrix with " << (sparsity * 100) << "% sparsity" << endl;
        }

        // Make copies of A for fair benchmarking
        double *data_dgetrf = new double[n * n];
        double *data_mpf = new double[n * n];
        memcpy(data_dgetrf, data_original, n * n * sizeof(double));
        memcpy(data_mpf, data_original, n * n * sizeof(double));

        // *Benchmark MPF* 
        int* ipiv_mpf = new int[n];
        double mpf_time = 0.0;
        
        // Initialize IPIV array to prevent garbage values
        for (int i = 0; i < n; i++) {
            ipiv_mpf[i] = i + 1;  // Initialize to identity permutation, like LAPACK does
        }
        
        cout << "Starting MPF factorization..." << endl;
        auto start = chrono::high_resolution_clock::now();
        try {
            MPF(data_mpf, n, 128, ipiv_mpf);
            auto end = chrono::high_resolution_clock::now();
            mpf_time = chrono::duration<double>(end - start).count();
            
            cout << "MPF completed successfully in " << mpf_time << " seconds" << endl;

            if (verbose) {
                cout << "MPF() time: " << mpf_time << " seconds\n" << endl;
            }

            if (check_correct) {
                cout<< "Checking correctness of MPF results..." << endl;
                if (!check_correctitude(data_original, data_mpf, ipiv_mpf, n, verbose)) {
                    cout << "MPF produced incorrect results." << endl;
                } else {
                    cout << "MPF correctness check passed!" << endl;
                }
            }
        } catch (const exception& e) {
            cout << "MPF failed with exception: " << e.what() << endl;
            auto end = chrono::high_resolution_clock::now();
            mpf_time = chrono::duration<double>(end - start).count();
            cout << "Time before failure: " << mpf_time << " seconds" << endl;
        }

        // *Benchmark LAPACKE_dgetrf*
        int *ipiv = new int[n];
        start = chrono::high_resolution_clock::now();
        int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, data_dgetrf, n, ipiv);
        auto end = chrono::high_resolution_clock::now();
        double lapack_time = chrono::duration<double>(end - start).count();

        cout << endl;
        if (info != 0) {
            cout << "LAPACKE_dgetrf failed with error code " << info << endl;
        }

        if (verbose) {
            cout << "LAPACKE_dgetrf time: " << lapack_time << " seconds\n" << endl;
        }

        if (check_correct) {
            if (!check_correctitude(data_original, data_dgetrf, ipiv, n, verbose)) {
                cout << "LAPACKE_dgetrf produced incorrect results." << endl;
            }
        }
        // Cleanup
        delete[] ipiv_mpf;
        delete[] data_original;
        delete[] data_mpf;
        delete[] data_dgetrf;
        delete[] ipiv;
        // Print results to CSV "benchmark_times.csv". order is: matrix_size, mpf_time, lapack_time
        csv << n << "," << mpf_time << "," << lapack_time << endl;
    }
    csv.close();

    return 0;
}