// references
// https://github.com/marek357/DNS-Matrix-Multiplication
//  For creating the Mesh layout and the topology for DNS Mat Mul
// https://github.com/cstroe/PP-MM-A03

// header files
#include <time.h>
#include <iostream>
#include <bits/stdc++.h>
#include "mpi.h"
#include "string.h"

// namspace
using namespace std;

// CONSTANTS
#define MAT_SIZE 420
#define MAIN_PROC_RANK 0
#define CART_2D_PLANE 2
#define MAX_PROCS 25
#define LEFT 1
#define UP 0
#define DISPLACEMENT 1
#define STRIDE 4
#define TAG 100
#define TRUE 1
#define FALSE 0
#define ROW 0
#define COL 1
#define ORIGIN_ROW 0
#define ORIGIN_COL 0
#define LOWER_BOUND 0

#define iAXIS 0
#define jAXIS 1
#define kAXIS 2

struct DNS_MESH {
    // overall cube communicator
    MPI_Comm cube;
    // per layer communicator needed for broadcasting
    MPI_Comm layer_ik, connector_j;
    MPI_Comm layer_jk, connector_i;
    MPI_Comm layer_ij, connector_k;
    // current processor rank
    int proc_rank;
    // processor coordinates in the 3 dimentions
    int cube_coords3d[3];
    // overall number of processors
    int num_procs;
    // q is the cube root of number of processors
    int q;
    // matrix dimensions
    int mat_size;
    // block dimensions
    int block_dim;
};

int *dims = (int*)malloc(sizeof(int)*3);
int *periods = (int*)malloc(sizeof(int)*3);

double** allocate_space(int row, int col) {
    /**
     * @brief Method to allocate memory and return a 2d matrix of
     * double precision points
     * 
     */
    double** result = new double*[row];
    for (int i=0; i<row; i++) {
        result[i] = new double[col];
    }
    return result;
}

double** initialize_data(int row, int col) {
    /**
     * @brief Method to initialize and return a 2d matrix
     * with random values
     * 
     */
    // random seed
    srand(time(NULL));

    double** temp = new double*[row];
    
    for (int i=0; i<row; i++) {
        temp[i] = new double[col];
        for (int j=0; j<col; j++) {
            temp[i][j] = i * row + j; // rand() % 10;
        }
    }
    return temp;
}

void init_matrix(int row, int col, double **mat, int order) {
    int max_ele = row * col - 1;
    int cur_ele;
    for (int i=0; i<row; i++) {
        for (int j=0; j<col; j++) {
            cur_ele = i * row + j;
            if (order == 1)
                mat[i][j] = cur_ele;
            else
                mat[i][j] = max_ele - cur_ele;
        }
    }
}

void compare_two_matrix(int row, int col, double **m1, double **m2) {
    cout << "Comparing results of traditional matrix multiplication against cannon implementation" <<endl;
    for (int i=0; i<row; i++) {
        for (int j=0; j<col; j++) {
            if (m1[i][j] != m2[i][j]) {
                cout << "RESULT INCORRECT" <<endl;
                return;
            }
        }
    }
    cout << "MATRIX RESULTS ARE CORRECT" <<endl;
}

// reference - how to initialize a 2d array using pointers in cpp
// https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/
void memory_allocation(int row, int col, double ***matrix) {
    int double_size_ptr = sizeof(double *);
    int dimensions = (row * col);
    double *sequential_array = (double *) malloc (dimensions * double_size_ptr);
    // allocate space for matrix
    *matrix = (double **) malloc(row * double_size_ptr);
    for (int i =0; i<row; i++) {
        // convert the sequential array to 2d matrix
        (*matrix)[i] = &(sequential_array[i * col]);
    }
}

void reset_to_zero(int row, int col, double** m) {
    /**
     * @brief Method that sets default value of zero to matrices
     * 
     */
    for(int i=0; i<row; i++) {
        for (int j=0; j<col; j++) {
            m[i][j] = 0.0;
        }
    }
}

/*
// code to print matrix

for (int i=0; i<MAT_SIZE; i++)  {
    for (int j=0; j<MAT_SIZE; j++) {
        printf("%d\t", m1[i][j]);
    }
    printf("\n");
}

*/

void print_matrix(int row, int col, double** m) {
    /**
     * @brief Method to print the matrix
     * 
     */
    for (int i=0; i<row; i++)  {
        for (int j=0; j<col; j++) {
            printf("%f\t", m[i][j]);
        }
        printf("\n");
    }
    printf("#---------------------------------\n");
}

void traditional_matrix_multiply(double** m1, double** m2, double** m3, int row, int col) {
    /**
     * @brief Method to compute matrix multiplication
     * 
     */
    int i, j, k, psum=0;
    for (i=0; i < row; i++) {
        for (j=0; j<col; j++) {
            psum = 0;
            for (k=0; k<col; k++) {
                psum += m1[i][k] * m2[k][j];
            }
            m3[i][j] += psum;
        }
    }
}


int get_dns_cube_dimensions(int num_procs) {
    /**
     * @brief Get the cube root of the num_procs which
     * is the appropriate dimension per DNS cube.
     */
    return (int) cbrt(num_procs);
}

int get_block_dimensions(int msize, int num_procs) {
    /**
     * @brief returns dimension size for each subblock of the matrix
     * 
     */
    return (int) msize / get_dns_cube_dimensions(num_procs);
}

void set_axis(int *dims, int i, int j, int k) {
    dims[iAXIS] = i; dims[jAXIS] = k; dims[kAXIS] = k;
}

void generate_dns_layout(struct DNS_MESH *info, int dim) {
    // assign the processors from command line argument to the mesh
    MPI_Comm_size(MPI_COMM_WORLD, &info->num_procs);
    info->mat_size = dim;
    info->q = get_dns_cube_dimensions(info->num_procs);
    info->block_dim = get_block_dimensions(info->mat_size, info->num_procs);
    cout << "q: " << info->q <<endl;
    cout << "block_dim: " << info->block_dim <<endl;

    if (info->proc_rank == MAIN_PROC_RANK) cout << "q: " << info->q <<endl;
    if (info->proc_rank == MAIN_PROC_RANK) cout << "block_dim: " << info->block_dim <<endl;

    dims[iAXIS] = dims[jAXIS] = dims[kAXIS] = info->q;
    periods[iAXIS] = periods[jAXIS] = periods[kAXIS] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &info->cube);

    MPI_Comm_rank(info->cube, &info->proc_rank);
    MPI_Cart_coords(info->cube, info->proc_rank, 3, info->cube_coords3d);

    // using cartesian to create a topology for each layer in the 3 axis and connecting them
    set_axis(dims, TRUE, FALSE, TRUE);
    MPI_Cart_sub(info->cube, dims, &info->layer_ik);
    set_axis(dims, FALSE, TRUE, FALSE);
    MPI_Cart_sub(info->cube, dims, &info->connector_j);
    // using cartesian to create a topology for each layer in the 3 axis and connecting them
    set_axis(dims, FALSE, TRUE, TRUE);
    MPI_Cart_sub(info->cube, dims, &info->layer_jk);
    set_axis(dims, TRUE, FALSE, FALSE);
    MPI_Cart_sub(info->cube, dims, &info->connector_i);
    // using cartesian to create a topology for each layer in the 3 axis and connecting them
    set_axis(dims, TRUE, TRUE, FALSE);
    MPI_Cart_sub(info->cube, dims, &info->layer_ij);
    set_axis(dims, FALSE, FALSE, TRUE);
    MPI_Cart_sub(info->cube, dims, &info->connector_k);
    if (&info->proc_rank == MAIN_PROC_RANK) cout <<"Layout created" <<endl;
}

void spread_matrix_across_blocks_layers(struct DNS_MESH *info, double** mat, double** block_matrix,
        int elements_per_processor[], int block_offsets[], int connectorDim, int bycolumn,
        int full_matrix_size[], int block_matrix_sizes[], int block_initial_offset[],
        MPI_Datatype block_matrix_type) 
{

    MPI_Comm layer_comm, connector_comm;
    layer_comm = connectorDim == jAXIS ? info->layer_ik : info->layer_jk;
    connector_comm = connectorDim == jAXIS ? info->connector_j : info->connector_i; 
    
    if (info->proc_rank == MAIN_PROC_RANK) cout << "starting scattering" <<endl;

    if (info->cube_coords3d[connectorDim] == MAIN_PROC_RANK) {
        // scatter data into the blocks
        MPI_Scatterv(&(mat[0][0]), elements_per_processor, block_offsets, block_matrix_type, &(block_matrix[0][0]),
            info->block_dim * info->block_dim, MPI_DOUBLE, 0, layer_comm);
    }

    if (info->proc_rank == MAIN_PROC_RANK) cout << "ending scattering" <<endl;
    // broad cast the data across the layer
    MPI_Bcast(&(block_matrix[0][0]), info->block_dim * info->block_dim, MPI_DOUBLE, 0, connector_comm);
    MPI_Type_free(&block_matrix_type);
}

int main(int argc, char* argv[]) {

    int i, j;
    // structure for the DNS Mesh
    struct DNS_MESH info;

    // variables for timing
    double start_time, end_time;
    // matrices to store data
    double **m1, **m2, **m3, **m4;

    // allocate memory for the matrices
    if (info.proc_rank == MAIN_PROC_RANK) cout << "reached matrix initialization" <<endl;
    memory_allocation(MAT_SIZE, MAT_SIZE, &m1);
    memory_allocation(MAT_SIZE, MAT_SIZE, &m2);
    memory_allocation(MAT_SIZE, MAT_SIZE, &m3);
    memory_allocation(MAT_SIZE, MAT_SIZE, &m4);

    // initialize data to matrices m1, m2
    init_matrix(MAT_SIZE, MAT_SIZE, m1, 1);
    init_matrix(MAT_SIZE, MAT_SIZE, m2, 0);
    if (info.proc_rank == MAIN_PROC_RANK) cout << "matrix init completed" <<endl;

    // start computation time
    start_time = MPI_Wtime();

    if (info.proc_rank == MAIN_PROC_RANK) cout << "sub blocks init" <<endl;
    double **block_m1, **block_m2, **block_m3, **block_m3_mpi_reduce_result;
    
    // initialize the arguments into the MPI_Init
    MPI_Init(&argc, &argv);
    // create the Mesh layout for the DNS algorithm
    if (info.proc_rank == MAIN_PROC_RANK) cout << "create the dns layout" <<endl;
    generate_dns_layout(&info, MAT_SIZE);
    if (info.proc_rank == MAIN_PROC_RANK) cout << "generated the dns layout" <<endl;

    // q square processors per layer
    int q_square = info.q * info.q;
    // number of elements allocated per layer
    int size_per_layer = sizeof(int) * q_square;

    int full_matrix_size[2] = {info.mat_size, info.mat_size};
    int block_matrix_sizes[2] = {info.block_dim, info.block_dim};
    int block_initial_offset[2] = {ORIGIN_ROW, ORIGIN_COL};

    MPI_Datatype full_matrix_type, block_matrix_type;
    MPI_Type_create_subarray(2, full_matrix_size, block_matrix_sizes, block_initial_offset, MPI_ORDER_C, MPI_DOUBLE, &full_matrix_type);
    MPI_Type_create_resized(full_matrix_type, 0, info.block_dim * sizeof(double), &block_matrix_type);
    MPI_Type_commit(&block_matrix_type);

    // perform traditional matrix multiplication for single processor
    if (info.proc_rank == MAIN_PROC_RANK) {
        if (info.num_procs == 1) {
            // start time
            start_time = MPI_Wtime();
            // perform mat mul for full matrix size and store the result in m3;
            traditional_matrix_multiply(m1, m2, m3, MAT_SIZE, MAT_SIZE);
            end_time = MPI_Wtime();
            cout <<"Timetaken for traditional MM: %f " << (end_time - start_time) <<endl;
            MPI_Finalize();
            exit(0);
        }
    }

    // allocate memory for sub blocks / layers
    memory_allocation(info.block_dim, info.block_dim, &block_m1);
    memory_allocation(info.block_dim, info.block_dim, &block_m2);
    memory_allocation(info.block_dim, info.block_dim, &block_m3);
    memory_allocation(info.block_dim, info.block_dim, &block_m3_mpi_reduce_result);
    if (info.proc_rank == MAIN_PROC_RANK) cout << "allocated sub blocks" <<endl;

    // elments per layer
    int* elements_per_processor = (int*) malloc(size_per_layer);
    // offsets per block
    int* block_offsets = (int*) malloc(size_per_layer);

    if (info.proc_rank == MAIN_PROC_RANK) for (int i=0; i< info.num_procs; i++) elements_per_processor[i] = 1;

    if (info.proc_rank == MAIN_PROC_RANK) {
        // determining block offsets
        for (int off = 0, i=0; i<info.q; i++, off += (info.block_dim - 1) * info.q) {
            for (int j=0; j<info.q; j++) {
                block_offsets[i * info.q + j] = off;
                off++;
                // cout << "block_offsets: " << block_offsets[i * info.q +  j] <<endl;
            }
        }
        // start the computation time.
        start_time = MPI_Wtime();
    }
    if (info.proc_rank == MAIN_PROC_RANK) cout << "offsets computed" <<endl;
    // spread the matrix across the layer
    /*                  Scatter Content
    Matrix A            Layer 0         Layer 1         Layer 2         Layer 3
    1   2   3   4       1   2   3   4   -   -   -   -   -   -   -   -   -   -   -   -
    5   6   7   8       -   -   -   -   5   6   7   8   -   -   -   -   -   -   -   -
    9   10  11  12      -   -   -   -   -   -   -   -   9   10  11  12  -   -   -   -
    13  14  15  16      -   -   -   -   -   -   -   -   -   -   -   -   13  15  15  16
                        Broad Cast the Content
                        Layer 0         Layer 1         Layer 2         Layer 3
                        1   2   3   4   5   6   7   8   9   10  11  12  13  15  15  16
                        1   2   3   4   5   6   7   8   9   10  11  12  13  15  15  16
                        1   2   3   4   5   6   7   8   9   10  11  12  13  15  15  16
                        1   2   3   4   5   6   7   8   9   10  11  12  13  15  15  16
    */
    if (info.proc_rank == MAIN_PROC_RANK) cout << "scattering matrix m1 started" <<endl;
    spread_matrix_across_blocks_layers(&info, m1, block_m1, elements_per_processor, block_offsets, jAXIS, 0,
            full_matrix_size, block_matrix_sizes, block_initial_offset, block_matrix_type);
    if (info.proc_rank == MAIN_PROC_RANK) cout << "scattering matrix m1 ended" <<endl;

    // spread the matrix across the layer
    /*                  Scatter Content
    Matrix B            Layer 0         Layer 1         Layer 2         Layer 3
    1   2   3   4       1   -   -   -   -   2   -   -   -   -   3   -   -   -   -   4
    5   6   7   8       5   -   -   -   -   6   -   -   -   -   7   -   -   -   -   8
    9   10  11  12      9   -   -   -   -   10  -   -   -   -   11  -   -   -   -   12
    13  14  15  16      13  -   -   -   -   14  -   -   -   -   15  -   -   -   -   16
                        Broad Cast content 
                        Layer 0         Layer 1         Layer 2         Layer 3
                        1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4
                        5   5   5   5   6   6   6   6   7   7   7   7   8   8   8   8
                        9   9   9   9   10  10  10  10  11  11  11  11  12  12  12  12  
                        13  13  13  13  14  14  14  14  15  15  15  15  16  16  16  16  
    */
    if (info.proc_rank == MAIN_PROC_RANK) cout << "scattering matrix m2 started" <<endl;
    spread_matrix_across_blocks_layers(&info, m2, block_m2, elements_per_processor, block_offsets, iAXIS, 0,
        full_matrix_size, block_matrix_sizes, block_initial_offset, block_matrix_type);
    if (info.proc_rank == MAIN_PROC_RANK) cout << "scattering matrix m2 ended" <<endl;

    // perform the matrix multiplication for the blocks
    traditional_matrix_multiply(block_m1, block_m2, block_m3, info.block_dim, info.block_dim);

    // do a reduce operation across the 3d layer and get a single result into one element across the 2d matrix
    // reference - https://rookiehpc.com/mpi/docs/mpi_reduce.php

    MPI_Reduce(&(block_m3[0][0]), // data in each layer
                &(block_m3_mpi_reduce_result[0][0]), // reduce and store in each element of the 2d matrix
                info.block_dim * info.block_dim, // dimensions
                MPI_DOUBLE,     // datatype of the result
                MPI_SUM,        // result operation addition
                MAIN_PROC_RANK,   // rank of the 0th processor
                info.connector_k); // the communicator along with the reduction is needed

    // wait for the operation to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // accumulate the individual blocks across the kAXIS into the single matrix
    if (info.cube_coords3d[kAXIS] == MAIN_PROC_RANK) {

        MPI_Gatherv(&(block_m3_mpi_reduce_result[0][0]), info.block_dim * info.block_dim, MPI_DOUBLE,
        &(m3[0][0]), elements_per_processor, block_offsets, block_matrix_type, 0, info.layer_ij);

        MPI_Type_free(&block_matrix_type);
    }

    if (info.proc_rank == MAIN_PROC_RANK) {
        end_time = MPI_Wtime();
        cout << "Timetaken for: " << end_time - start_time <<endl;
    }
    

    // Finalize and end the Code
    try {
        MPI_Finalize();
    } catch (MPI::Exception failure) {
        cerr << failure.Get_error_string() << endl;
        MPI::COMM_WORLD.Abort(1);
    }

    return 0;

}





















