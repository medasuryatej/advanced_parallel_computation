#include <omp.h>
#include <stdio.h>

#define ARR_SIZE 1000000000
int a[ARR_SIZE];

void get_time() {
    printf("omp_get_wtime() =%g\n", omp_get_wtime());
    printf("omp_get_wtick() =%g\n", omp_get_wtick());
    double t1 = omp_get_wtime();
    for (int i =0; i< 1000000; i++);
    printf("time taken = %g\n", t1-omp_get_wtime());
}

void arr_sum_sequential() {
    /** initialize the array **/
    for (int i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }

    /** sum the array **/
    int sum = 0;
    double start = omp_get_wtime();
    for (int i=0; i<ARR_SIZE; i++) {
        sum += a[i];
    }
    double time_diff = omp_get_wtime() - start;
    printf("Sequential sum = %d; time taken = %g\n", sum, time_diff);

    /**
     * @brief Result: Sequential sum = 1000000000; time taken = 1.85643
     * 
     */
}

void parallelize_attemp1(){
    int i, tid, numt, sum=0;
    double t1, t2;

    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }

    t1 = omp_get_wtime();
    #pragma omp parallel default(shared) private(i, tid)
    {
        tid = omp_get_thread_num();
        numt = omp_get_num_threads();
        for (i=0; i<ARR_SIZE; i++) {
            sum += a[i];
        }
    }
    t2 = omp_get_wtime();
    printf("Parallelize atmp1: sum = %d; time taken = %g\n", sum , t2-t1);

    /**
     * @brief Result: Parallelize atmp1: sum = 1151250307; time taken = 12.0354
     * 
     * First the sum value is wrong and time taken is way to much
     * Reason 1: Race conditions on sum and irregular access to sum varaible
     * Reason 2: All threads are trying to perform sum of billion numbers, there is 
     * no actual parallelization happend.
     * Reason 3: Time went up because of Cache coherency
     * 
     */
}

void parallelize_attempt2() {
    /** Distributing work among theads **/
    int i, tid, numt, sum=0;
    double t1, t2;

    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }

    t1 = omp_get_wtime();
    #pragma omp parallel default(shared) private(i, tid) 
    {
        int from, to;
        tid = omp_get_thread_num();
        numt = omp_get_num_threads();
        from = (ARR_SIZE / numt) * tid;
        to = (ARR_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1) {
            to = ARR_SIZE - 1;
        }
        t1 = omp_get_wtime() - t1;
        printf("\tThead id=%d, range: from=%d to=%d\n", tid, from, to);
        t2 = omp_get_wtime();
        for (i = from; i<to ;i++) {
            sum += a[i];
        }
    }
    t2 = omp_get_wtime() - t2;
    printf("Parallelize atmp2: sum = %d; time taken = %g\n", sum , t2+t1);
    /**
     * @brief Output: Thead id=0, range: from=0 to=249999999
        Thead id=2, range: from=500000000 to=749999999
        Thead id=1, range: from=250000000 to=499999999
        Thead id=3, range: from=750000000 to=999999999
        Parallelize atmp2: sum = 322464490; time taken = 2.90592

        Well the time is fixed, but the output still remains incorrect
     * 
     */
}

void parallelize_attempt3() {
    /** Distributing work among theads **/
    int i, tid, numt, sum=0;
    double t1, t2;

    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }

    t1 = omp_get_wtime();
    #pragma omp parallel default(shared) private(i, tid) 
    {
        int from, to;
        tid = omp_get_thread_num();
        numt = omp_get_num_threads();
        from = (ARR_SIZE / numt) * tid;
        to = (ARR_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1) {
            to = ARR_SIZE - 1;
        }
        t1 = omp_get_wtime() - t1;
        printf("\tThead id=%d, range: from=%d to=%d\n", tid, from, to);
        t2 = omp_get_wtime();
        for (i = from; i<to ;i++) {
            #pragma omp critical
            sum += a[i];
        }
    }
    t2 = omp_get_wtime() - t2;
    printf("Parallelize atmp3: sum = %d; time taken = %g\n", sum , t2+t1);
    /**
     * @brief Result: Thead id=0, range: from=0 to=249999999
        Thead id=2, range: from=500000000 to=749999999
        Thead id=3, range: from=750000000 to=999999999
        Thead id=1, range: from=250000000 to=499999999
        Parallelize atmp3: sum = 999999996; time taken = 6.22936e+06
     * 
     * Pragma omp critical has extreme overhead
     */
}

void parallelize_attempt4() {
    /** Distributing work among theads **/
    int i, tid, numt, sum=0;
    double t1, t2;

    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }

    t1 = omp_get_wtime();
    #pragma omp parallel default(shared) private(i, tid) 
    {
        int from, to, psum=0;
        tid = omp_get_thread_num();
        numt = omp_get_num_threads();
        from = (ARR_SIZE / numt) * tid;
        to = (ARR_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1) {
            to = ARR_SIZE - 1;
        }
        t1 = omp_get_wtime() - t1;
        printf("\tThead id=%d, range: from=%d to=%d\n", tid, from, to);
        t2 = omp_get_wtime();
        for (i=from; i<=to; i++)
        {
            psum += a[i];
        }
        #pragma omp critical
        sum += psum;
    }
    t2 = omp_get_wtime() - t2;
    printf("Parallelize atmp4: sum = %d; time taken = %g\n", sum , t2+t1);
    /**
     * @brief Result: Thead id=0, range: from=0 to=249999999
        Thead id=3, range: from=750000000 to=999999999
        Thead id=1, range: from=250000000 to=499999999
        Thead id=2, range: from=500000000 to=749999999
        Parallelize atmp4: sum = 1000000000; time taken = 0.374321
     * 
     */
}

void parallelize_attemp5() {
    int i, sum = 0;
    double t1, t2;
    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }
    t1= omp_get_wtime();
    #pragma omp parallel default(shared) private(i)
    {
        int psum=0;
        #pragma omp for
        for (i=0; i<ARR_SIZE; i++) {
            psum += a[i];
        }
        #pragma omp critical
        sum += psum;
    }
    t2 = omp_get_wtime();
    printf("Paralleize atm5: sum = %d; time taken = %g\n", sum, t2-t1);

   /**
    * @brief Paralleize atm5: sum = 1000000000; time taken = 0.544633Result: 
    * 
    */
}

void parallelize_attemp6() {
    /* Use omp reduction to implement partial sum*/
    int i, sum = 0;
    double t1, t2;
    for (i=0; i<ARR_SIZE; i++) {
        a[i] = 1;
    }
    t1= omp_get_wtime();
    #pragma omp parallel default(shared) private(i) reduction(+:sum)
    {
        #pragma omp for
        for (i=0; i<ARR_SIZE; i++) {
            sum += a[i];
        }
    }
    t2 = omp_get_wtime();
    printf("Paralleize atm6: sum = %d; time taken = %g\n", sum, t2-t1);

    /**
     * @brief Result: Paralleize atm6: sum = 1000000000; time taken = 0.531436
     * 
     */

}

int main(int *argc, char *argv[]) {
    // get_time();
    omp_set_num_threads(4);
    arr_sum_sequential();
    parallelize_attemp1();
    parallelize_attempt2();
    parallelize_attempt3();
    parallelize_attempt4(); // perfect output
    parallelize_attemp5(); // omp for
    parallelize_attemp6(); // reduction
    return 0;
}

/**
* Command to compile: gcc -fopenmp -o compute_sum.o compute_sum.c
* Command to run: ./compute_sum.o
**/
