#include "mpi.h"
#include "stdio.h"
#include <stdlib.h>
#include <time.h>

#define DIM 2100

// Normal Matrices
// double A[DIM][DIM], B[DIM][DIM], C[DIM][DIM];

// Complex Number defintion
struct ComplexNumber {
    double realPart;
    double imgPart;
};

// Adding complex numbers
struct ComplexNumber addition (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = c1.realPart + c2.realPart;
    c3.imgPart = c1.imgPart + c2.imgPart;
    return c3;
};

// Subtract complex numbers
struct ComplexNumber subtract (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = c1.realPart - c2.realPart;
    c3.imgPart = c1.imgPart - c2.imgPart;
    return c3;
};

// Multiply complex numbers
struct ComplexNumber multiplication (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = (c1.realPart * c2.realPart) + (-1)*(c1.imgPart * c2.imgPart);
    c3.imgPart = (c1.realPart * c2.imgPart) + (c1.imgPart * c2.realPart);
    return c3;
}

// multiply complex matrices
void complex_mat_mul(struct ComplexNumber m1[][DIM], struct ComplexNumber m2[][DIM], struct ComplexNumber m3[][DIM], int from, int to) {
    int i,j, k;
    double rPsum, iPsum;
    for (i=from; i<to; i++) {
        for (j=0; j<DIM; j++) {
            // m3[i][j].realPart = 0;
            // m3[i][j].imgPart = 0;
            rPsum = iPsum = 0.0;
            for (k=0; k<DIM; k++) {
                rPsum += (m1[i][k].realPart * m2[k][j].realPart) + (-1)*(m1[i][k].imgPart * m2[k][j].imgPart);
                iPsum += (m1[i][k].realPart * m2[k][j].imgPart) + (m1[i][k].imgPart * m2[k][j].realPart);
            }
            m3[i][j].realPart = rPsum;
            m3[i][j].imgPart = iPsum;
        }
    }
}

// initialize normal double matrices
void initialize_matrix(double m[DIM][DIM], int randN) {
    /**
     * @brief Initialize matrices with random values based on randN seed
     * 
     */
    int i,j,k;
    // printf("\n I am here\n");
    for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
            if (randN == 1) {
                m[i][j] = rand() % randN;
            } else {
                m[i][j] = rand() % randN;
            }
        }
    }
}

void print_matrix(double m[DIM][DIM]) {
    /**
     * @brief Print matrices
     * 
     */
    // printf("printing matrix %s", M);
    int i,j;
    for (i=0; i<DIM; i++) {
        printf("\n\t| ");
        for (j=0; j<DIM; j++) {
            printf("%lf\t", m[i][j]);
        }
    }
}

void print_complex_matrix(struct ComplexNumber m[DIM][DIM]) {
    /**
     * @brief Print the complex matrix
     * 
     */
    int i,j;
    for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
            printf("%.2f + i %.2f\t", m[i][j].realPart, m[i][j].imgPart);
        }
        printf("\n");
    } 
}

struct ComplexNumber m1[DIM][DIM], m2[DIM][DIM], m3[DIM][DIM];
int main(int argc, char *argv[]) {
    int rank, num_proc, scatter_from, scatter_to, i,j,k;
    // struct ComplexNumber m1[DIM][DIM], m2[DIM][DIM], m3[DIM][DIM]; //, m4[DIM][DIM];
    MPI_Status status;
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    MPI_Datatype ComplexNumberType, legacyType[1];
    int blockCounts[1];

    MPI_Aint offsets[1], label, mpiextnt;

    offsets[0] = 0;
    legacyType[0] = MPI_DOUBLE;
    blockCounts[0] = 2;

    MPI_Type_get_extent(MPI_DOUBLE, &label, &mpiextnt);
    MPI_Type_create_struct(1, blockCounts, offsets, legacyType, &ComplexNumberType);
    MPI_Type_commit(&ComplexNumberType);
    /*
    if (DIM % num_proc != 0) {
        if (rank == 0) {
            printf("Incompatible sizes");
        }
        MPI_Finalize();
        exit(-1);
    }
    */

    scatter_from = rank * DIM / num_proc;
    scatter_to = (rank + 1) * DIM / num_proc;
    // printf("\nFrom %d To %d, rank %d\n", from, to, rank);
    // start timer
    double start = MPI_Wtime();

    if (rank == 0) {
        
        /*
        initialize_matrix(A, 23);
        initialize_matrix(B, 29);
        printf("A matrix\n");
        print_matrix(A);
        printf("B matrix\n");
        print_matrix(B);
        printf("\n");
        */
       // initialize complex matrix A & B
       for (i=0; i<DIM; i++) {
           for (j=0; j<DIM; j++) {
               m1[i][j].realPart = rand() % 23;
               m1[i][j].imgPart = rand() % 29;

               m2[i][j].realPart = rand() % 31;
               m2[i][j].imgPart = rand() % 37;
           }
       }
       // print_complex_matrix(m1);
       // print_complex_matrix(m2);
       if (num_proc == 1) {
         // single processor
         complex_mat_mul(m1, m2, m3, 0, DIM);
         double end_time = MPI_Wtime();
         printf("Scatter-Gather: 1 Processor with Dimensions %d took %f seconds for Complex MatMul\n", DIM, (end_time-start));
         MPI_Finalize();
         exit(0);
       }
    }

    // MPI_Bcast(B, (DIM*DIM), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m2, (DIM*DIM), ComplexNumberType, 0, MPI_COMM_WORLD);

    // MPI_Scatter(A, (DIM*DIM/num_proc), MPI_DOUBLE, A[from], (DIM*DIM/num_proc), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(m1, (DIM*DIM/num_proc), ComplexNumberType, m1[scatter_from], (DIM*DIM/num_proc), ComplexNumberType, 0, MPI_COMM_WORLD);

    /*
    // Normal Matmul
    for (i=from; i<to; i++) {
        for (j=0; j<DIM; j++) {
            C[i][j] = 0;
            for (k=0; k<DIM; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    */
    start = MPI_Wtime();
    complex_mat_mul(m1, m2, m3, scatter_from, scatter_to);
    // MPI_Gather(C[from], (DIM*DIM/num_proc), MPI_DOUBLE, C, (DIM*DIM/num_proc), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(m3[scatter_from], (DIM*DIM/num_proc), ComplexNumberType, m3, (DIM*DIM/num_proc), ComplexNumberType, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        // print_matrix(C);
        // print_complex_matrix(m3);
        double end = MPI_Wtime();
        // printf("With %d processors, Complex Mat Mul Dim %d took %f seconds\n", num_proc, DIM, (end-start));
        printf("Scatter-Gather: %d Processors with Dimensions %d took %f seconds for Complex MatMul\n", num_proc, DIM, (end-start));
        /*
        start = MPI_Wtime();
        complex_mat_mul(m1, m2, m3, 0, DIM);
        end = MPI_Wtime();
        printf("Single Proc Complex Mat Mul Dim %d time taken %f\n", DIM, (end-start));
        */
    }
    MPI_Finalize();
    return 0;
}

