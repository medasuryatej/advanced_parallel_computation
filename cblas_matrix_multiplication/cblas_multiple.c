 #include <time.h>
 #include <stdlib.h>
 #include <stdio.h>
 #include "cblas.h"
 #include "cblas_f77.h"

 #define SIZE 500
 
 void generate_matrix(double* matrix, int row, int column)
 {
   for (int j = 0; j < column; j++){
     for (int i = 0; i < row; i++){
       matrix[j*row + i] = ((double)rand())/RAND_MAX;
     }
   }
 }
 
void print_matrix(double (*mat), int row, int column) {
    printf("Printing Matrix:\n ");
    for (int j = 0; j < column; j++){
      for (int i = 0; i < row; i++){
			  printf("%f\t", mat[j*row + i]);
    	}
    	printf("\n");
    }
}

void initialize_matrix(double (*mat), int row, int column) {
    for (int j = 0; j < column; j++) {
        for (int i = 0; i < row; i++) {
            mat[j*row + i] = 0.0;
        }
    }
}
 
int main() {

  int rowsA = SIZE, colsB = SIZE, dot = SIZE;
  int i,j,k;
  double A[rowsA * dot]; double B[dot * colsB];
  double CBLAS_RESULT[rowsA * colsB]; double NORMAL[rowsA * colsB];
  char *filename = "cblas_multiple_n.txt";

  CBLAS_LAYOUT order = CblasColMajor;
  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;
  srand(time(NULL));
  clock_t time;
  double time_taken;

  FILE *fp = fopen(filename, "w");
  if (fp == NULL)
   {
        printf("Error opening the file %s", filename);
        return -1;
   }

  for (int size = 100; size <= 500; size += 100) {
      rowsA = size;
      colsB = size;
      dot = size;
      A[rowsA * dot]; 
      B[dot * colsB]; 
      CBLAS_RESULT[rowsA * colsB];
      NORMAL[rowsA * colsB];
  
      generate_matrix(A, rowsA, dot); 
      generate_matrix(B, dot, colsB);
      initialize_matrix(CBLAS_RESULT, rowsA, colsB);
      initialize_matrix(NORMAL, rowsA, colsB);
    
      time = clock();

      cblas_dgemm(order, transA, transB, rowsA, colsB, dot ,1.0,A, 
                  rowsA ,B, dot ,0.0, CBLAS_RESULT, rowsA);

      time = clock() - time;
      
      time_taken = ((double)time) / CLOCKS_PER_SEC; 
      printf("Time with cblas is %.5f seconds for %d dimension\n", time_taken, size);
      fprintf(fp, "Time with cblas is %.5f seconds for %d dimension\n", time_taken, size);
      
      time = clock();
      for(i=0;i<colsB;i++) {
          for(j=0;j<rowsA;j++) {
              NORMAL[i*rowsA+j]=0;
              for(k=0;k<dot;k++) {
                  NORMAL[i*rowsA+j]+=A[k*rowsA+j]*B[k+dot*i];
            }
        }
    }
    time = clock() - time;
    time_taken = ((double)time)/CLOCKS_PER_SEC; 
    printf("Time without using cblas is %.5f seconds for %d dimension\n", time_taken, size);
    fprintf(fp, "Time without using cblas is %.5f seconds for %d dimension\n", time_taken, size);
  }

  // close the file
  fclose(fp);

   return 0;
 }