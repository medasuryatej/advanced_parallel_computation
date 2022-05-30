#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <time.h>
#include <omp.h>

#define THRESHOLD 128 // min matrix size needed to traditionally perform matmul

using namespace std;

void add_or_sub_matrices(vector<vector<double>> &A, vector<vector<double>> &B, 
                  vector<vector<double>> &C, int mat_size, int operation) {

    /**
     * @brief Add or Subtract two matrices and store result in another matrix
     * 
     */
    #pragma omp parallel for
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            if (operation == 1) {
                C[i][j] = A[i][j] + B[i][j];
            } else {
                C[i][j] = A[i][j] - B[i][j];
            }
            
        }
    }

}

void print_matrices(vector<vector<double>> &A, int mat_size) {

    /**
     * @brief Print Given Matrix
     * 
     */
    std::cout <<"Printing Matrix "<<endl;
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            std::cout << A[i][j] <<"\t";
        }
        std::cout <<endl;
    }

}

void normal_matrix_multiplication(vector<vector<double>> &A,
                                vector<vector<double>> &B,
                                vector<vector<double>> &C)
{
    /**
     * @brief Once the matrix is divided into enough small parts
     * using the current method to perform traditional matrix multiplication
     * base 2
     */
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0];
        }
        #pragma omp section
        {
            C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1];
        }
        #pragma omp section
        {
            C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0];
        }
        #pragma omp section
        {
            C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1];
        }
    }
    
}

void baseline_mat_mul(vector<vector<double>> &A,
                    vector<vector<double>> &B,
                    vector<vector<double>> &C)
{
    /**
     * @brief Matrix Multiplication when mat_size is under threshold
     * 
     */

    double psum = 0.0;
    #pragma omp parallel for reduction(+:psum)
    for (int i=0; i<THRESHOLD; i++) {
        for (int j=0; j<THRESHOLD; j++) {
            psum = 0.0;
            for (int k=0; k<THRESHOLD; k++) {
                psum += A[i][k] * B[k][j];
            }
            C[i][j] = psum;
        }
    }
}

int GET_MIN(int var1, int var2) {
    if (var1 < var2) {
        return var1;
    } 
    else {
        return var2;
    }
}

void baseline_mat_mul_block(vector<vector<double>> &A, 
                                    vector<vector<double>> &B,
                                    vector<vector<double>> &C,
                                    int mat_size)
{
    int block_size = 32;
    double psum = 0.0;
    #pragma omp parallel for reduction(+:psum)
    for (int b_i=0; b_i<mat_size; b_i+=block_size) {
        for (int b_j=0; b_j<mat_size; b_j+=block_size) {
            for (int b_k=0; b_k<mat_size; b_k+=block_size) {
                #pragma omp parallel for
                for (int ii=b_i; ii<GET_MIN(b_i+block_size, mat_size); ii++) {
                    for (int jj=b_j; jj<GET_MIN(b_j+block_size, mat_size); jj++) {
                        psum = 0.0;
                        for (int kk =b_k; kk < GET_MIN(b_k+block_size, mat_size); kk++) {
                            psum += A[ii][kk] * B[kk][jj]; 
                        }
                        C[ii][jj] += psum;
                    }
                }
            }
        }
    }

}


void strassen_recursive_multiply(vector<vector<double>> &A,
                                vector<vector<double>> &B,
                                vector<vector<double>> &C,
                                int mat_size) {

    // base case 
    if (mat_size == THRESHOLD) {
        // normal_matrix_multiplication(A, B, C);
        baseline_mat_mul(A, B, C);
        // baseline_mat_mul_block(A, B, C, THRESHOLD);
        return;
    }
    else {
        int half_mat_size = mat_size / 2;
        vector<double> rowVector(half_mat_size, 0.0);
        // recursive sub matrices
        /**
         * @brief 
         * P1 = A11 * (B12 - B22)
         * P2 = (A11 + A12) * B22
         * P3 = (A21 + A22) * B11
         * P4 = A22 * (B21 - B11) 
         * P5 = (A11 + A22) * (B11 + B22)
         * P6 = (A12 - A22) * (B21 + B22)
         * P7 = (A11 - A21) * (B11 + B12)
         * 
         * C11 = P5 + P4 - P2 + P6
         * C12 = P1 + P2
         * C21 = P3 + P4
         * C22 = P5 + P1 - P3 - P7
         */

        // A Sub matrices
        vector<vector<double>> A11(half_mat_size, rowVector),
                               A12(half_mat_size, rowVector),
                               A21(half_mat_size, rowVector),
                               A22(half_mat_size, rowVector);
        // B Sub matrices
        vector<vector<double>> B11(half_mat_size, rowVector),
                               B12(half_mat_size, rowVector),
                               B21(half_mat_size, rowVector),
                               B22(half_mat_size, rowVector);
        // C sub matrices
        vector<vector<double>> C11(half_mat_size, rowVector),
                               C12(half_mat_size, rowVector),
                               C21(half_mat_size, rowVector),
                               C22(half_mat_size, rowVector);

        /**
         * @brief Simplying P1 to P7 equations
         * 
         * Op1 = B12 - B22
         * Op2 = A11 + A12
         * Op3 = A21 + A22
         * Op4 = B21 - B11
         * Op5 = A11 + A22
         * Op6 = B11 + B22
         * Op7 = A12 - A22
         * Op8 = B21 + B22
         * Op9 = A11 - A21
         * Op10 = B11 + B12
         * 
         */

        vector<vector<double>> Op1(half_mat_size, rowVector), Op2(half_mat_size, rowVector),
                               Op3(half_mat_size, rowVector), Op4(half_mat_size, rowVector),
                               Op5(half_mat_size, rowVector), Op6(half_mat_size, rowVector),
                               Op7(half_mat_size, rowVector), Op8(half_mat_size, rowVector),
                               Op9(half_mat_size, rowVector), Op10(half_mat_size, rowVector);

        // Equations P1 to P7
        vector<vector<double>> P1(half_mat_size, rowVector), P2(half_mat_size, rowVector),
                               P3(half_mat_size, rowVector), P4(half_mat_size, rowVector),
                               P5(half_mat_size, rowVector), P6(half_mat_size, rowVector),
                               P7(half_mat_size, rowVector);   

        // intermediate results
        vector<vector<double>> intermediateA(half_mat_size, rowVector), intermediateB(half_mat_size, rowVector),
                                intermediateC(half_mat_size, rowVector), intermediateD(half_mat_size, rowVector);

        // Split Matirx A into 4 parts (A11, A12, A21, A22)
        // SPlit Matrix B into 4 parts (B11, B12, B21, B22)
        #pragma omp parallel for
        for (int i = 0; i < half_mat_size; i++) {
            for (int j = 0; j < half_mat_size; j++) {
                // split A
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + half_mat_size];
                A21[i][j] = A[i + half_mat_size][j];
                A22[i][j] = A[i + half_mat_size][j + half_mat_size];

                // split B
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j + half_mat_size];
                B21[i][j] = B[i + half_mat_size][j];
                B22[i][j] = B[i + half_mat_size][j + half_mat_size];

            }
        }

        // start the parallel region
        #pragma omp parallel
        {
            // single thread should create the tasks
            #pragma omp single
            {
                #pragma omp task
                {
                    // Op1 = B12 - B22
                    add_or_sub_matrices(B12, B22, Op1, half_mat_size, 0);
                    // P1 = A11 * (B12 - B22) = A11 * Op1
                    strassen_recursive_multiply(A11, Op1, P1, half_mat_size);   
                }
                #pragma omp task
                {
                    // Op2 = A11 + A12
                    add_or_sub_matrices(A11, A12, Op2, half_mat_size, 1);
                    // P2 = (A11 + A12) * B22 = Op2 * B22
                    strassen_recursive_multiply(Op2, B22, P2, half_mat_size);
                }
                #pragma omp task
                {
                    // Op3 = A21 + A22
                    add_or_sub_matrices(A21, A22, Op3, half_mat_size, 1);
                    // P3 = (A21 + A22) * B11 = Op3 * B11
                    strassen_recursive_multiply(Op3, B11, P3, half_mat_size);
                }
                #pragma omp task
                {
                    // Op4 = B21 - B11
                    add_or_sub_matrices(B21, B11, Op4, half_mat_size, 0);
                    // P4 = A22 * (B21 - B11) = A22 * Op4
                    strassen_recursive_multiply(A22, Op4, P4, half_mat_size);
                }
                #pragma omp task
                {
                    // Op5 = A11 + A22
                    // Op6 = B11 + B22
                    add_or_sub_matrices(A11, A22, Op5, half_mat_size, 1);
                    add_or_sub_matrices(B11, B22, Op6, half_mat_size, 1);
                    // P5 = (A11 + A22) * (B11 + B22) = Op5 * Op6
                    strassen_recursive_multiply(Op5, Op6, P5, half_mat_size);

                }
                #pragma omp task
                {
                    // Op7 = A12 - A22
                    // Op8 = B21 + B22
                    add_or_sub_matrices(A12, A22, Op7, half_mat_size, 0);
                    add_or_sub_matrices(B21, B22, Op8, half_mat_size, 1);
                    // P6 = (A12 - A22) * (B21 + B22) = Op7 * Op8
                    strassen_recursive_multiply(Op7, Op8, P6, half_mat_size);
                }
                #pragma omp task
                {
                    // Op9 = A11 - A21
                    // Op10 = B11 + B12
                    add_or_sub_matrices(A11, A21, Op9, half_mat_size, 0);
                    add_or_sub_matrices(B11, B12, Op10, half_mat_size, 1);
                    // P7 = (A11 - A21) * (B11 + B12) = Op9 * Op10   
                    strassen_recursive_multiply(Op9, Op10, P7, half_mat_size);
                }

                // individual operations can be tasked but have to wait before merging the results
                #pragma omp taskwait

                #pragma omp task
                {
                    // C11 = P5 + P4 - P2 + P6
                    add_or_sub_matrices(P5, P4, intermediateA, half_mat_size, 1); // addition
                    add_or_sub_matrices(intermediateA, P2, intermediateB, half_mat_size,0); // subtraction
                    add_or_sub_matrices(intermediateB, P6, C11, half_mat_size, 1);
                }

                #pragma omp task
                {
                    // C12 = P1 + P2
                    add_or_sub_matrices(P1, P2, C12, half_mat_size, 1); // addition   
                }

                #pragma omp task
                {
                    // C21 = P3 + P4
                    add_or_sub_matrices(P3, P4, C21, half_mat_size, 1);   
                }

                #pragma omp task
                {
                    // C22 = P5 + P1 - P3 - P7
                    add_or_sub_matrices(P5, P1, intermediateC, half_mat_size, 1); // addition
                    add_or_sub_matrices(intermediateC, P3, intermediateD, half_mat_size, 0); // subtraction
                    add_or_sub_matrices(intermediateD, P7, C22, half_mat_size, 0); // subtraction
                }

                #pragma omp taskwait
            } // single end
            #pragma omp for
                    for (int i=0; i < half_mat_size; i++) {
                        for (int j=0; j < half_mat_size; j++) {
                            C[i][j] = C11[i][j];
                            C[i][j + half_mat_size] = C12[i][j];
                            C[i + half_mat_size][j] = C21[i][j];
                            C[i + half_mat_size][j + half_mat_size] = C22[i][j];
                        }
                    }
        } // parallel region end
    }
}

void strassen_multiplication(vector<vector<double>> &A, 
                            vector<vector<double>> &B,
                            vector<vector<double>> &C,
                            int mat_size) {

    strassen_recursive_multiply(A, B, C, mat_size);
}

void generate_matrix(vector<vector<double>> &matrix, int mat_size, int prime) {
    srand(time(NULL));
    // std::cout <<"Entered here"<<endl;
    for (int i = 0; i < mat_size; i++) {
        vector<double> tempVector;
        for (int j = 0; j < mat_size; j++) {
            tempVector.push_back(rand() % prime);
            // matrix[i][j] = rand() % 10;
            // std::cout << matrix[i][j] << "\t";
        }
        matrix.push_back(tempVector);
        // std::cout <<endl;
    }
}

void compare_results(vector<vector<double>> &MAT1, vector<vector<double>> &MAT2, int mat_size) {
    for (int i=0; i<mat_size; i++) {
        for (int j=0; j<mat_size; j++) {
            if (MAT1[i][j] != MAT2[i][j]) {
                cout << "Incompatible results" <<endl;
                return;
            }
        }
    }
    cout << "Matrix results valid" <<endl;
}

void matrix_multiplication_sequential(vector<vector<double>> &A, 
                                    vector<vector<double>> &B,
                                    vector<vector<double>> &C,
                                    int mat_size) {
    double start = omp_get_wtime();
    double a = 0.0, b=0.0, c=0.0, d=0.0;

    for (int i=0; i<mat_size; i+=2) {
        for (int j=0; j<mat_size; j+=2) {
            a = 0.0; b=0.0; c=0.0; d=0.0;
            for (int k=0; k<mat_size; k++) {
                // psum += A[i][k] * B[k][j];
                a += A[i][k] * B[k][j];
                b += A[i][k] * B[k][j+1];
                c += A[i+1][k] * B[k][j];
                d += A[i+1][k] * B[k][j+1];
            }
            // C[i][j] = psum;
            C[i][j] = a;
            C[i][j+1] = b;
            C[i+1][j] = c;
            C[i+1][j+1] = d;
        }
    }
    double end = omp_get_wtime();
    cout << "Serial: Dim: " << mat_size << " Mat Mul, time taken: " << (end - start) <<endl;
}

int main(int argc, char *argv[]) {
    int mat_size = 2048; // powers of 2
    double start, end;
    vector<int> thread_count;
    for (int i=1; i<=6; i++) {
        thread_count.push_back(pow(2,i));
    }
    // omp_set_num_threads(64);
    vector<vector<double>> A;
    // std::cout <<"Matrix A" <<endl;
    generate_matrix(A, mat_size, 13);
    // print_matrices(A, mat_size);

    vector<vector<double>> B;
    // std::cout <<"Matrix B" <<endl;
    generate_matrix(B, mat_size, 17);
    // print_matrices(B, mat_size);

    for (int i=0; i<thread_count.size(); i++) {
        omp_set_num_threads(thread_count[i]);
        // initialize C to zero
        vector<vector<double>> C(mat_size, vector<double>(mat_size, 0.0));  
        start = omp_get_wtime();
        strassen_multiplication(A, B, C, mat_size);
        end = omp_get_wtime();
        cout << "Strassen; Dim; " << mat_size << " Threads; "<< thread_count[i] << "; Time taken: " << (end - start) <<endl;
    }

    return 0;
}