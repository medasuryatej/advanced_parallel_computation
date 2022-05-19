#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>

using namespace std;

#define R 2000 // ROWS
#define C 2000 // COLS
#define ITERATIONS 15

ofstream q1_part1;

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

void mat_mul_normal(double** A, double** B, int dimensions) {
    double** result = new double*[dimensions];
    result = initialize_matrix(dimensions);

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j < dimensions; j++) {
            for (int k = 0; k < dimensions; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;

    cout << "Normal: Dimensions: " << dimensions << " Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    q1_part1 << "Normal: Dimensions: " << dimensions << " Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";

    // print_matrix(result, n);
    // cout <<"\n";
}

void mat_mul_unroll(double** A, double** B, int dimensions) {
    double sum=0.0;
    double** result = new double*[dimensions];
    result = initialize_matrix(dimensions);

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j < dimensions; j++) {
            sum = 0.0;
            // loop unrolling
            for (int k = 0; k < dimensions; k = k+5) {
                sum += (A[i][k] * B[k][j]) + (A[i][k+1] * B[k+1][j]) + 
                        (A[i][k+2] * B[k+2][j]) + (A[i][k+3] * B[k+3][j]) + 
                        (A[i][k+4] * B[k+4][j]);
            }
            result[i][j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;

    cout << "Unroll: Dimensions: " << dimensions << " Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    q1_part1 << "Unroll: Dimensions: " << dimensions << " Time taken for " << " : " << fixed  << time_taken << setprecision(9) <<"\n";
}

int main() {
    q1_part1.open("q1_normal_vs_loop_unroll_db_pointer_funroll_loops.txt");
    double** A;double** B;double** result;
    A = generate_matrix(R);
    B = generate_matrix(C);
    try {
        for (int i=100; i <=2000; i += 100) {
            mat_mul_normal(A, B, i);
            mat_mul_unroll(A, B, i);
        }
    } catch (exception &e) {
        cout << e.what() << '\n';
        q1_part1.close();
    }
    
}