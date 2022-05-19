#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>

using namespace std;

#define R 2000 // ROWS
#define C 2000 // COLS

ofstream q1_part1; // file pointer to write output

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

void mat_mul_generic(double** A, double** B, int n, int p, const char* pattern) {
    double** result = new double*[n];
    result = initialize_matrix(n); //n - dimensions

    auto start = chrono::high_resolution_clock::now();
    // loop reorderings
    switch(p) {
        case 0:
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        case 1:
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    for (int j = 0; j < n; j++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        case 2:
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < n; k++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        case 3:
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    for (int i = 0; i < n; i++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        case 4:
            for (int k = 0; k < n; k++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        case 5:
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            break;
        default :
            printf("unknown pattern");
    }
    
    auto end = chrono::high_resolution_clock::now();

    double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;

    cout << "Dimensions: " << n << " Time taken for " << pattern << " : " << fixed  << time_taken << setprecision(9) <<"\n";
    q1_part1 << "Dim: " << n << " Pattern: " << pattern << " Time: " << fixed  << time_taken << setprecision(9) <<"\n";
}

void mat_mul(int N) {
    double** A; double** B;
    A = generate_matrix(N);
    B = generate_matrix(N);
    // print_matrix(A, N);
    mat_mul_generic(A, B, N, 0, "i, j, k");
    mat_mul_generic(A, B, N, 1, "i, k, j");
    mat_mul_generic(A, B, N, 2, "j, i, k");
    mat_mul_generic(A, B, N, 3, "j, k, i");
    mat_mul_generic(A, B, N, 4, "k, i, j");
    mat_mul_generic(A, B, N, 5, "k, j, i");
}

int main() {
    q1_part1.open("q1_normal_db_pointer_flags.txt");

    try 
    {
        for (int iterations=100; iterations<=2000; iterations+=100) {
            cout << "#-------------------------------------\n";
            mat_mul(iterations);
        }
    }
    catch (exception &e) {
        cout << e.what() << '\n';
        q1_part1.close();
    }
    
    return 0;
}