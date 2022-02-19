**How to execute?**

| Operation  | Command |
| ------------- | ------------- |
| Compile  | `gcc -fopenmp -o compute_sum.o compute_sum.c`  |
| Run  | `./compute_sum.o`  |

Considered `ARR_SIZE` to be `1 billion`.
Number of threads: `4`.

**Evaluation:**

| Method | Optimization Performed | Sum | Time Taken(secs) | Comments |
| :---         |     :---:      |          ---: |          ---: |          ---: |
| Serial   | None     | 1000000000    | 1.856     | None    |
| Parallel Attempt1   | `#pragma omp parallel`    | 1151250307    | 12.0354    | 1. Incorrect result (Race condition). <br/>2.More time consumed than serial    |
| Parallel Attempt2   | Manually divded ARR_SIZE into chunks for each thread     | 322464490    | 2.90592     | Race condition still exists    |
| Parallel Attempt3   | `#pragma omp critical`     | 1000000000    | 6.22936e+06     | critical region code is executed for 1 billion times, which introduces extreme overhead     |
| Parallel Attempt4   | using local psum variable     | 1000000000    | 0.374321     | Used thread private psum variable to store local sums and use critical to perform final sum     |
| Parallel Attempt5   | `#pragma omp for`     | 1000000000    | 0.544633     | instead of manually dividing the iterations, utilized the omp for and letting compiler take care of it.     |
| Parallel Attempt6   | `#pragma omp for reduction(+:sum)`    | 1000000000    | 0.531436     | Instead of using local thread private psum variable, used omp reduction which does the same internally    |
