#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>

using namespace std;

#define R 2000
#define C 2000

ofstream q1_part1;

// void generate_matrix(double (*mat)[C], int n) {
// Method to generate a matrix with random values
double** generate_matrix( int n) {
    double**A = new double*[n]; 
    for(int i = 0; i < n; i++) {
        A[i] = new double[n];
    	for(int j = 0; j < n; j++) {
			A[i][j] = rand() % 100;
    	}
    }
    return A; 
}

// void initialize_matrix(double (*mat)[C], int n) {
// Method to initialize results with zero
double** initialize_matrix(int n) {
    double**result = new double*[n]; 
    for (int i=0; i<n; i++) {
        result[i] = new double[n];
        for (int j=0; j<n; j++) {
            result[i][j] = 0.0;
        }
    }
    return result;
}

// void block_multiply_normal(double (*A)[C], double (*B)[C], int block) {
void block_multiply_normal(double** A, double** B, int block, int dimensions) {
    double** result = new double*[dimensions];
    result = initialize_matrix(dimensions);
    int i, j, k, ii, jj, kk;

    auto start = chrono::high_resolution_clock::now();

    for (ii = 0; ii < dimensions; ii+=block) {
        for (jj = 0; jj < dimensions; jj+=block) {
            for (kk = 0; kk < dimensions; kk+=block) {
                for (i = ii; i < ii+block; i++) {
                    for (j = jj; j < jj+block; j++) {
                        for (k = kk; k < kk+block; k++) {
                            result[i][j] += A[i][k]*B[k][j];
                        }
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "Dimensions: "<< dimensions << " Block: " << block << " Time for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    q1_part1 << "Dimensions: "<< dimensions << " Block: " << block << " Time for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";

}

// void block_multiply_optimized(double (*A)[C], double (*B)[C], int block, const char* optim) {
void block_multiply_optimized(double** A, double** B, int block, const char* optim, int dimensions)  {
    double** result = new double*[dimensions];
    result = initialize_matrix(dimensions);
    int i, j, k, ii, jj, kk;
    double sum = 0.0;

    auto start = chrono::high_resolution_clock::now();

    for (ii = 0; ii < dimensions; ii+=block) {
        for (jj = 0; jj < dimensions; jj+=block) {
            for (kk = 0; kk < dimensions; kk+=block) {
                for (i = ii; i < ii+block; i++) {
                    for (j = jj; j < jj+block; j++) {
                        sum = 0.0;
                        for (k = kk; k < kk+block; k+=5) {
                        	// loop unrolling
                            sum += (A[i][k]*B[k][j]) + 
                                    (A[i][k+1]*B[k+1][j]) +
                                    (A[i][k+2]*B[k+2][j]) +
                                    (A[i][k+3]*B[k+3][j]) +
                                    (A[i][k+4]*B[k+4][j]);
                        }
                        result[i][j] += sum;
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "Dimensions: "<< dimensions << " Block Loop unroll: " << block << " Time for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    q1_part1 << "Dimensions: "<< dimensions << " Block Loop unroll: " << block << " Time for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
}

int main() {
    // double A[R][C], B[R][C], result[R][C];
    double** A;double** B;double** result;
    int block = 100;
    int n, iterations;
    int i, j, k, ii, jj, kk;
    
    A = generate_matrix(C);
    B = generate_matrix(C);

    q1_part1.open("blocking_normal_vs_optim_with_flags.txt");

    for (iterations = 100; iterations <= R; iterations += 100) {
        block_multiply_normal(A, B, 25, iterations);
        block_multiply_normal(A, B, 50, iterations);
        block_multiply_normal(A, B, 100, iterations);
        block_multiply_optimized(A, B, 25, "loop unroll", iterations);
        block_multiply_optimized(A, B, 50, "loop unroll", iterations);
        block_multiply_optimized(A, B, 100, "loop unroll", iterations);
    }

    // Normal Matrix Multiplication
    for (iterations = 100; iterations <= R; iterations += 100) {
        // intialize result
        result = initialize_matrix(C);
        // start time
        auto start = chrono::high_resolution_clock::now();
        // mat mul normal
        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < iterations; j++) {
                for (int k = 0; k < iterations; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        // end time
        auto end = chrono::high_resolution_clock::now();
        double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        time_taken *= 1e-9;
        // write to out file
        cout << "Dimensions: "<< iterations << " Regular: Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
        q1_part1 << "Dimensions: "<< iterations << " Regular: Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    }
    q1_part1.close();
}