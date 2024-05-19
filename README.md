# Parallel Matrix Multiplication

This project implements various methods for matrix multiplication, including sequential, parallel, divide and conquer, and Strassen's algorithm. It uses OpenMP for parallelization.

## Requirements

- C++ Compiler (e.g., g++)
- OpenMP library

## Files

- `main.cpp`: Contains all the implementations and the main function.
- `matrix1.txt`, `matrix2.txt`: Example input matrix files.

## Compilation

To compile the code, use the following command:

```sh
g++ -fopenmp -o matrix_multiplication main.cpp

## Running the Code

### 1. Generating a Random Matrix

First, generate random matrices to use as input for the multiplication algorithms.

1. Uncomment the `mainGenrateMatrix()` function call in the `main()` function.
2. Compile and run the code:

    ```sh
    g++ -fopenmp -o matrix_multiplication main.cpp
    ./matrix_multiplication
    ```

3. You will be prompted to enter the matrix dimension and output filename. Generate at least two matrices (e.g., `matrix1.txt` and `matrix2.txt`).

### 2. Running Matrix Multiplication Algorithms

After generating the input matrices, run the desired matrix multiplication algorithm:

1. Comment out the `mainGenrateMatrix()` function call in the `main()` function.
2. Uncomment the function call for the algorithm you want to run in the `main()` function.
3. Ensure the filenames in the selected function match your generated matrix files (e.g., `matrix1.txt` and `matrix2.txt`).
4. Compile and run the code:

    ```sh
    g++ -fopenmp -o matrix_multiplication main.cpp
    ./matrix_multiplication
    ```

#### Example: Running Parallel Strassen Algorithm

1. Comment out `mainGenrateMatrix()`.
2. Uncomment `runMatrixMultiplicationStrassenParallel()`.
3. Ensure the filenames `filename1` and `filename2` within `runMatrixMultiplicationStrassenParallel()` match your generated matrix files.
4. Compile and run the code:

    ```sh
    g++ -fopenmp -o matrix_multiplication main.cpp
    ./matrix_multiplication
    ```

The output will be written to the specified output files, and the execution time will be printed to the console.
