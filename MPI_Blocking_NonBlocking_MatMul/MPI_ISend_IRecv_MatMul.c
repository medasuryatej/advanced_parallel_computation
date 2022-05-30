/**
 * References:
 * 
 * https://www.codingame.com/playgrounds/349/introduction-to-mpi/measuring-time
 * // Reference - https://ecomputernotes.com/what-is-c/structure-and-union/complex-numbers-by-structures#:~:text=A%20complex%20number%20is%20also,Cl%20%3D%20x%20%2B%20iy
// Reference - https://www.codingame.com/playgrounds/47058/have-fun-with-mpi-in-c/derived-datatypes
// Reference - http://www.cas.mcmaster.ca/~nedialk/COURSES/mpi/Lectures/lec2_1.pdf
 * 
 */


#include <stdio.h>
#include <math.h>
#include<stdlib.h>
#include <time.h>
#include "mpi.h"

#define DIM 420 // static for testing
#define MAIN_PROC 0
#define TAG_FROM_MAIN 100
#define TAG_FROM_OTHER 200
#define MAX_NUM_PROCS 25

// int DIM = 1024;

// double matA[DIM][DIM], matB[DIM][DIM], matC[DIM][DIM];

struct ComplexNumber {
    double realPart;
    double imgPart;
};

struct ComplexNumber addition (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = c1.realPart + c2.realPart;
    c3.imgPart = c1.imgPart + c2.imgPart;
    return c3;
};

struct ComplexNumber subtract (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = c1.realPart - c2.realPart;
    c3.imgPart = c1.imgPart - c2.imgPart;
    return c3;
};

struct ComplexNumber multiplication (struct ComplexNumber c1, struct ComplexNumber c2) {
    struct ComplexNumber c3;
    c3.realPart = (c1.realPart * c2.realPart) - (c1.imgPart * c2.imgPart);
    c3.imgPart = (c1.realPart * c2.imgPart) + (c1.imgPart * c2.realPart);
    return c3;
}

void complex_mat_mul(struct ComplexNumber m1[][DIM], struct ComplexNumber m2[][DIM], struct ComplexNumber m3[][DIM], int rows) {
    int i,j, k;
    /*
    // Normal Full matrix multiplication
    for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
            m3[i][j].realPart = 0;
            m3[i][j].imgPart = 0;
            for (k=0; k<DIM; k++) {
                m3[i][j].realPart += (m1[i][k].realPart * m2[k][j].realPart) - (m1[i][k].imgPart * m2[k][j].imgPart);
                m3[i][j].imgPart += (m1[i][k].realPart * m2[k][j].imgPart) + (m1[i][k].imgPart * m2[k][j].realPart);
            }
        }
    }
    */
   // Mat Mul based on the rows info sent to processors
   for (k=0; k<DIM; k++) {
        for (i=0; i<rows; i++) {
            m3[i][k].realPart = 0;
            m3[i][k].imgPart = 0;
            for (j=0; j<DIM; j++) {
                m3[i][k].realPart += (m1[i][j].realPart * m2[j][k].realPart) - (m1[i][j].imgPart * m2[j][k].imgPart);
                m3[i][k].imgPart += (m1[i][j].realPart * m2[j][k].imgPart) + (m1[i][j].imgPart * m2[j][k].realPart);
            }
        }
    }
}

void transpose_complex_matrix(struct ComplexNumber m[][DIM]) {
    int i, j;
    struct ComplexNumber transpose[DIM][DIM];
    for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
            transpose[i][j] = m[j][i];
        }
    }
    for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
            m[i][j] = transpose[i][j];
        }
    }
}

struct ComplexNumber m1[DIM][DIM], m2[DIM][DIM], m3[DIM][DIM];

int main(int argc, char* argv[]) {
    // printf("Number of args %d\n",argc);
    // printf("Arg1: %s\n",argv[0]);
    // printf("Arg2: %s\n",argv[1]);
    // if (argc > 1) {
    //   // assigning dimensions from runtime
    //   DIM = atoi(argv[1]);
    // }
    srand(time(NULL));
    // data strutcutre to hold complex numbers
    // struct ComplexNumber m1[DIM][DIM], m2[DIM][DIM], m3[DIM][DIM];

    // custom mpi datatype
    MPI_Datatype ComplexNumberType, legacyType[1];
    int blockCounts[1];
    // getting and assigning address based on MPI_Aint
    MPI_Aint offsets[1], lb, extent;

    // numprocesors, processor ranks
    int total_procs, proc_rank;
    // iterator variables
    int i,j,k;
    // number of parallel processors
    int parallel_procs;
    // the row/col data that is going to be transferred
    int row_transfer, col_transfer;
    // starting offset across the columns
    int starting_offset;
    // tag message number
    int tag;
    // mpi status
    MPI_Status status[MAX_NUM_PROCS], stat;
    // sending to which processor
    int sender_proc;
    // temp sum result
    int psum, count;

    // Initalize MPI
    MPI_Init(&argc, &argv);
    // assign ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    // assign sizes
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    // define offset structure
    offsets[0] = 0;
    // double datatype
    legacyType[0] = MPI_DOUBLE;
    // two types of elements in the structure
    blockCounts[0] = 2;

    // define the custom mpi structure datatype and commit it.
    MPI_Type_get_extent(MPI_DOUBLE, &lb, &extent);
    MPI_Type_create_struct(1, blockCounts, offsets, legacyType, &ComplexNumberType);
    MPI_Type_commit(&ComplexNumberType);

    // num parallel processors
    parallel_procs = total_procs - 1;

    MPI_Request request_send[4*MAX_NUM_PROCS] = {MPI_REQUEST_NULL}, request_recv[4*MAX_NUM_PROCS]= {MPI_REQUEST_NULL}, request= {MPI_REQUEST_NULL};
    int receive_buffer[MAX_NUM_PROCS];

    if (proc_rank == MAIN_PROC && (total_procs < 1 || total_procs > MAX_NUM_PROCS)) {
        printf("Incorrect number of processors asigned value must be between 1 and 25");
        MPI_Finalize();
        exit(0);
    }

    // 0th rank processor
    if (proc_rank == MAIN_PROC) {

        double start = MPI_Wtime();

        // seeding random
        srand(time(NULL));

        // initialize matrices with random values
        /*
        for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                matA[i][j] = rand() % 17;
                matB[i][j] = rand() % 23;
            }
        }
        */
       for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                // matrix one
                m1[i][j].realPart = rand() % 7;
                m1[i][j].imgPart = rand() % 9;
                // matrix two
                m2[i][j].realPart = rand() % 3;
                m2[i][j].imgPart = rand() % 5;
            }
        }
        if (total_procs == 1) {
          //  single processors
          complex_mat_mul(m1,m2,m3,DIM);
          double end_time = MPI_Wtime();
          printf("Send-Recv: 1 Processor with Dimensions %d too %f seconds for Complex MatMul\n", DIM, (end_time - start));
          MPI_Finalize();
          exit(0);
        }

        /*
        // print matrix
        printf("Matrix A From rank: %d\n", proc_rank);
        for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                printf("%.2f + i %.2f\t", m1[i][j].realPart, m1[i][j].imgPart);
            }
            printf("\n");
        }

        // print matrix
        printf("Matrix B From rank: %d\n", proc_rank);
        for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                printf("%.2f + i %.2f\t", m2[i][j].realPart, m2[i][j].imgPart);
            }
            printf("\n");
        }
        */

        /*
        // print matrices
        printf("\n----MatrixA----\n");
        for (i=0; i<DIM; i++) {
            printf("\n\t| ");
            for (j=0; j<DIM; j++) {
                printf("%.2f | ", matA[i][j]);
            }
        }
        printf("\n----MatrixB----\n");
        // print matrices
        for (i=0; i<DIM; i++) {
            printf("\n\t| ");
            for (j=0; j<DIM; j++) {
                printf("%.2f | ", matB[i][j]);
            }
        }
        */

        // number of rows assigned to each of the processors
        row_transfer = DIM / parallel_procs;
        // offset from which data across columns needs to be sent
        starting_offset = 0;
        // tag number for message
        // set tag=100 for forward data transfer from Main processor to other processors
        tag = 100;
        // send the data
        // printf("Reached isend stating\n");
        // int num_reqs = 0;
        for (int proc_num=1; proc_num <= parallel_procs; proc_num++) {
            // printf("num req %d\n", (4*(proc_num-1)));
            // send the offset and number of rows that are going to be transmitted
            MPI_Isend(&starting_offset, 1, MPI_INT, proc_num, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_send[4*(proc_num-1)]);
            MPI_Isend(&row_transfer, 1, MPI_INT, proc_num, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_send[4*(proc_num-1)+1]);
            // send the address of the buffer
            // MPI_Send(&matA[starting_offset][0], row_transfer*DIM, MPI_DOUBLE, proc_num, tag, MPI_COMM_WORLD);
            MPI_Isend(&m1[starting_offset][0], row_transfer*DIM, ComplexNumberType, proc_num, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_send[4*(proc_num-1)+2]);
            // send the snapshot/address of entire matrix B
            // MPI_Send(&matB, DIM*DIM, MPI_DOUBLE, proc_num, tag, MPI_COMM_WORLD);
            // MPI_Send(&m2, DIM*DIM, ComplexNumberType, proc_num, TAG_FROM_MAIN, MPI_COMM_WORLD);
            MPI_Isend(&m2[starting_offset][0], row_transfer*DIM, ComplexNumberType, proc_num, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_send[4*(proc_num-1)+3]);
            // update the offset
            starting_offset += row_transfer; 
            // num_reqs += 1;
        }
        // printf("Total numreqs: %d\n", num_reqs);
        // printf("Reached isend end\n");
        MPI_Waitall(parallel_procs*4, request_send, status);
        // printf("Reached waitall for send\n");

        // receive the data
        // set tag number to be 200 for receiving the computed data from other processors
        tag = 200;
        int num_req_recv = 0;
        for (int proc_num=1; proc_num <= parallel_procs; proc_num++) {
            // printf("num req recv %d\n", (3*(proc_num-1)));
            // which processor are we receiving the data
            sender_proc = proc_num;
            // recieve the offset and row information
            
            MPI_Recv(&starting_offset, 1, MPI_INT, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &stat);
            MPI_Recv(&row_transfer, 1, MPI_INT, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &stat);
            // receive the actual data and store in the matrix C
            // MPI_Recv(&matC[starting_offset][0], row_transfer*DIM, MPI_DOUBLE, sender_proc, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&m3[starting_offset][0], row_transfer*DIM, ComplexNumberType, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &stat);
            
            /*
            MPI_Irecv(&starting_offset, 1, MPI_INT, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_recv[3*(proc_num-1)]);
            num_req_recv += 1;
            MPI_Irecv(&row_transfer, 1, MPI_INT, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_recv[3*(proc_num-1)+1]);
            num_req_recv += 1;
            MPI_Irecv(&m3[starting_offset][0], row_transfer*DIM, ComplexNumberType, sender_proc, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_recv[3*(proc_num-1)+2]);
            num_req_recv += 1;
            */
        }
        // printf("Total numreqs recv: %d\n", num_req_recv);
        // printf("reached wait all recv\n");
        // MPI_Waitall(num_req_recv, request_recv, status);
        // printf("reached wait all recv end\n");

        /*
        printf("\n----MatrixC----\n");
        // print matrices
        for (i=0; i<DIM; i++) {
            printf("\n\t| ");
            for (j=0; j<DIM; j++) {
                printf("%.2f | ", matC[i][j]);
            }
        }
        */
        /*
        printf("Matrix C from rank: %d\n", proc_rank);
        for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                printf("%.2f + i %.2f\t", m3[i][j].realPart, m3[i][j].imgPart);
            }
            printf("\n");
        } 
        */
        
       double end = MPI_Wtime();

       printf("ISend-IRecv: %d Processors with Dimensions %d took %f seconds for Complex MatMul\n",total_procs, DIM, (end-start));
    }
    else {
        // parallel procs
        sender_proc = MAIN_PROC;
        // set tag=100 for receiving the info form main processor
        tag = 100;
        // receive offsets
        MPI_Irecv(&starting_offset, 1, MPI_INT, sender_proc, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_recv[0]);
        MPI_Irecv(&row_transfer, 1, MPI_INT, sender_proc, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_recv[1]);
        // receive the data and store in matA and matB
        // MPI_Recv(&matA, row_transfer*DIM, MPI_DOUBLE, sender_proc, tag, MPI_COMM_WORLD, &status);
        MPI_Irecv(&m1, row_transfer*DIM, ComplexNumberType, sender_proc, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_recv[2]);
        // MPI_Recv(&matB, DIM*DIM, MPI_DOUBLE, sender_proc, tag, MPI_COMM_WORLD, &status);
        // MPI_Recv(&m2, DIM*DIM, ComplexNumberType, sender_proc, TAG_FROM_MAIN, MPI_COMM_WORLD, &status);
        MPI_Irecv(&m2, row_transfer*DIM, ComplexNumberType, sender_proc, TAG_FROM_MAIN, MPI_COMM_WORLD, &request_recv[3]);

        MPI_Waitall(4, request_recv, status);

        // matrix multiplication
        complex_mat_mul(m1, m2, m3, row_transfer);

        /*
        // compute the Actual Matrix Multiplication
        for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
                psum= 0;
                for (k=0; k<DIM; k++) {
                    psum += matA[i][k] * matB[k][j];
                }
                matC[i][j] = psum;
            }
        }
        */

        // set tag=200 for sending back the data
        tag = 200;
        // send back the computed data
        MPI_Isend(&starting_offset, 1, MPI_INT, MAIN_PROC, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_send[0]);
        MPI_Isend(&row_transfer, 1, MPI_INT, MAIN_PROC, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_send[1]);
        // send the computed matix multiplication result
        // MPI_Send(&matC, row_transfer*DIM, MPI_DOUBLE, MAIN_PROC, TAG_FROM_OTHER, MPI_COMM_WORLD);
        MPI_Isend(&m3, row_transfer*DIM, ComplexNumberType, MAIN_PROC, TAG_FROM_OTHER, MPI_COMM_WORLD, &request_send[2]);

        MPI_Waitall(3, request_send, status);
    }

    // MPI_Get_count(status, MPI_CHAR, &count);
    /**
    printf("Task %d: Received %d char(s) from task %d with tag %d\n",
        proc_rank, count, status.MPI_SOURCE, status.MPI_TAG);
    **/

    // end the process
    MPI_Finalize();
    
    return 0;
}
