#include <vector>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <ctime>
#include <string>



// ******************************************************************************************************************** //
// genrate matrix

void generateRandomMatrix(int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::mt19937 generator(static_cast<unsigned int>(time(0))); // Seed the random number generator
    std::uniform_int_distribution<int> distribution(-9, 9);

    // Write the matrix dimension to the file
    file << n << std::endl;

    // Generate and write the matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << distribution(generator) << " ";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Random matrix generated and written to " << filename << std::endl;
}

int mainGenrateMatrix() {
    int n;
    std::string filename;

    std::cout << "Enter matrix dimension (n): ";
    std::cin >> n;
    std::cout << "Enter output filename: ";
    std::cin >> filename;

    generateRandomMatrix(n, filename);

    return 0;
}

// ******************************************************************************************************************** //
//common functions

const int BLOCK_SIZE = 128; // Block size for cache optimization
const int SWITCH_SIZE = 128; // Threshold to switch to block multiplication
const int threshold = 128; // Threshold to switch to naive multiplication
 const int Number_of_cores = omp_get_max_threads(); // Number of cores available on the system


void readMatrix(const std::string& filename, std::vector<std::vector<long>>& matrix, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file >> n;
    matrix.resize(n, std::vector<long>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> matrix[i][j];
        }
    }

    file.close();
}

void writeMatrix(const std::string& filename, const std::vector<std::vector<long>>& matrix, int n) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the matrix dimension to the file
    file << n << std::endl;

    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            file << elem << " ";
        }
        file << std::endl;
    }

    file.close();
}


void writeTime(const std::string& filename, std::chrono::duration<double> duration, int n) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int hours = static_cast<int>(duration.count() / 3600);
    int minutes = static_cast<int>((duration.count() - hours * 3600) / 60);
    double seconds = duration.count() - hours * 3600 - minutes * 60;

    file << "Time taken: " << hours << ":" << minutes << ":" << seconds << std::endl;
    file.close();
}

void addMatrix(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtractMatrix(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void addMatrices(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Function to subtract two matrices
void subtractMatrices(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}



// ******************************************************************************************************************** //



// Function for sequential matrix multiplication

void multiplyMatrices(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void runMatrixMultiplication() {
    std::string filename1 = "10_1";
    std::string filename2 = "matrix_1024";
    std::string outputFilenameBase = "input3_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n));

    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_Sequential.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_Sequential.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}

// ******************************************************************************************************************** //

// parallel matrix multiplication


void multiplyMatricesParallel(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                // Multiply the block sub-matrices
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
                        long sum = 0;
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

void runMatrixMultiplicationParallel() {
    std::string filename1 = "2048";
    std::string filename2 = "2048";
    std::string outputFilenameBase = "input2_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n, 0));

    // Set the number of threads to the maximum available
    omp_set_num_threads(Number_of_cores);

    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatricesParallel(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_SequentialP.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_SequentialP.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Parallel matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}

// ******************************************************************************************************************** //
// Divide and Conquer Matrix Multiplication


// Naive divide and conquer matrix multiplication
void multiplyMatrices2(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Divide-and-conquer matrix multiplication
void multiplyDivideAndConquer(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n, int threshold) {
    if (n <= threshold) {
        multiplyMatrices2(A, B, C, n);
        return;
    }

    int newSize = n / 2;
    std::vector<std::vector<long>> A11(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> A12(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> A21(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> A22(newSize, std::vector<long>(newSize));

    std::vector<std::vector<long>> B11(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B12(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B21(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B22(newSize, std::vector<long>(newSize));

    std::vector<std::vector<long>> C11(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C12(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C21(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C22(newSize, std::vector<long>(newSize));

    std::vector<std::vector<long>> P1(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P2(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P3(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P4(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P5(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P6(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P7(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> P8(newSize, std::vector<long>(newSize));

    // Divide matrices into submatrices
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // P1 = A11 * B11
    multiplyDivideAndConquer(A11, B11, P1, newSize, threshold);

    // P2 = A12 * B21
    multiplyDivideAndConquer(A12, B21, P2, newSize, threshold);

    // P3 = A11 * B12
    multiplyDivideAndConquer(A11, B12, P3, newSize, threshold);

    // P4 = A12 * B22
    multiplyDivideAndConquer(A12, B22, P4, newSize, threshold);

    // P5 = A21 * B11
    multiplyDivideAndConquer(A21, B11, P5, newSize, threshold);

    // P6 = A22 * B21
    multiplyDivideAndConquer(A22, B21, P6, newSize, threshold);

    // P7 = A21 * B12
    multiplyDivideAndConquer(A21, B12, P7, newSize, threshold);

    // P8 = A22 * B22
    multiplyDivideAndConquer(A22, B22, P8, newSize, threshold);

    // C11 = P1 + P2
    addMatrices(P1, P2, C11, newSize);

    // C12 = P3 + P4
    addMatrices(P3, P4, C12, newSize);

    // C21 = P5 + P6
    addMatrices(P5, P6, C21, newSize);

    // C22 = P7 + P8
    addMatrices(P7, P8, C22, newSize);

    // Combine submatrices into result matrix
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

void runMatrixMultiplicationDivideAndConquer() {
    std::string filename1 = "4096";
    std::string filename2 = "4096";
    std::string outputFilenameBase = "input3_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n));

    auto start = std::chrono::high_resolution_clock::now();
    multiplyDivideAndConquer(A, B, C, n , threshold );
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_DivideAndConquer.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_DivideAndConquer.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}

// ******************************************************************************************************************** //
// parallel divide and conquer matrix multiplication

void multiplyMatricesParallel2(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
                        long sum = 0;
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}


void multiplyDivideAndConquerParallel(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    if (n <= SWITCH_SIZE) {
        multiplyMatricesParallel2(A, B, C, n);
        return;
    }

    int newSize = n / 2;
    std::vector<std::vector<long>> A11(newSize, std::vector<long>(newSize)), A12(newSize, std::vector<long>(newSize)), A21(newSize, std::vector<long>(newSize)), A22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B11(newSize, std::vector<long>(newSize)), B12(newSize, std::vector<long>(newSize)), B21(newSize, std::vector<long>(newSize)), B22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C11(newSize, std::vector<long>(newSize)), C12(newSize, std::vector<long>(newSize)), C21(newSize, std::vector<long>(newSize)), C22(newSize, std::vector<long>(newSize));

#pragma omp parallel for schedule(static, BLOCK_SIZE) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    std::vector<std::vector<long>> M1(newSize, std::vector<long>(newSize)), M2(newSize, std::vector<long>(newSize)), M3(newSize, std::vector<long>(newSize)), M4(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> M5(newSize, std::vector<long>(newSize)), M6(newSize, std::vector<long>(newSize)), M7(newSize, std::vector<long>(newSize));

#pragma omp parallel sections
    {
#pragma omp section
        multiplyDivideAndConquerParallel(A11, B11, M1, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A12, B21, M2, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A11, B12, M3, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A12, B22, M4, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A21, B11, M5, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A22, B21, M6, newSize);
#pragma omp section
        multiplyDivideAndConquerParallel(A21, B12, M7, newSize);
    }

#pragma omp parallel for schedule(static, BLOCK_SIZE) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C11[i][j] = M1[i][j] + M2[i][j];
            C12[i][j] = M3[i][j] + M4[i][j];
            C21[i][j] = M5[i][j] + M6[i][j];
            C22[i][j] = M1[i][j] - M5[i][j] + M3[i][j] - M7[i][j];
        }
    }

#pragma omp parallel for schedule(static, BLOCK_SIZE) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

void runMatrixMultiplicationDivideAndConquerParallel() {
    std::string filename1 = "2048";
    std::string filename2 = "2048";
    std::string outputFilenameBase = "input1_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n, 0));

    omp_set_num_threads(Number_of_cores); // Adjust based on your system's high-performance cores

    auto start = std::chrono::high_resolution_clock::now();
    multiplyDivideAndConquerParallel(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_StraightDivAndConqP.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_StraightDivAndConqP.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Parallel Divide and Conquer matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}


// ******************************************************************************************************************** //
// strassen matrix multiplication Parrallel
void addMatricesUnique(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtractMatricesUnique(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void multiplyMatricesParallel3(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                // Multiply the block sub-matrices
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
                        long sum = 0;
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

void strassenMultiplyUnique(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
    if (size <= threshold) {
        multiplyMatricesParallel3(A, B, C, size);
        return;
    }

    int newSize = size / 2;
    std::vector<std::vector<long>> A11(newSize, std::vector<long>(newSize)), A12(newSize, std::vector<long>(newSize)), A21(newSize, std::vector<long>(newSize)), A22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B11(newSize, std::vector<long>(newSize)), B12(newSize, std::vector<long>(newSize)), B21(newSize, std::vector<long>(newSize)), B22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C11(newSize, std::vector<long>(newSize)), C12(newSize, std::vector<long>(newSize)), C21(newSize, std::vector<long>(newSize)), C22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> M1(newSize, std::vector<long>(newSize)), M2(newSize, std::vector<long>(newSize)), M3(newSize, std::vector<long>(newSize)), M4(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> M5(newSize, std::vector<long>(newSize)), M6(newSize, std::vector<long>(newSize)), M7(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> AResult(newSize, std::vector<long>(newSize)), BResult(newSize, std::vector<long>(newSize));

#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            addMatricesUnique(A11, A22, AResult, newSize); // A11 + A22
            addMatricesUnique(B11, B22, BResult, newSize); // B11 + B22
            strassenMultiplyUnique(AResult, BResult, M1, newSize); // M1 = (A11 + A22) * (B11 + B22)
        }
#pragma omp section
        {
            addMatricesUnique(A21, A22, AResult, newSize); // A21 + A22
            strassenMultiplyUnique(AResult, B11, M2, newSize); // M2 = (A21 + A22) * B11
        }
#pragma omp section
        {
            subtractMatricesUnique(B12, B22, BResult, newSize); // B12 - B22
            strassenMultiplyUnique(A11, BResult, M3, newSize); // M3 = A11 * (B12 - B22)
        }
#pragma omp section
        {
            subtractMatricesUnique(B21, B11, BResult, newSize); // B21 - B11
            strassenMultiplyUnique(A22, BResult, M4, newSize); // M4 = A22 * (B21 - B11)
        }
#pragma omp section
        {
            addMatricesUnique(A11, A12, AResult, newSize); // A11 + A12
            strassenMultiplyUnique(AResult, B22, M5, newSize); // M5 = (A11 + A12) * B22
        }
#pragma omp section
        {
            subtractMatricesUnique(A21, A11, AResult, newSize); // A21 - A11
            addMatricesUnique(B11, B12, BResult, newSize); // B11 + B12
            strassenMultiplyUnique(AResult, BResult, M6, newSize); // M6 = (A21 - A11) * (B11 + B12)
        }
#pragma omp section
        {
            subtractMatricesUnique(A12, A22, AResult, newSize); // A12 - A22
            addMatricesUnique(B21, B22, BResult, newSize); // B21 + B22
            strassenMultiplyUnique(AResult, BResult, M7, newSize); // M7 = (A12 - A22) * (B21 + B22)
        }
    }

#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C12[i][j] = M3[i][j] + M5[i][j];
            C21[i][j] = M2[i][j] + M4[i][j];
            C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

void runMatrixMultiplicationStrassenParallel() {
    std::string filename1 = "4096";
    std::string filename2 = "4096";
    std::string outputFilenameBase = "input3_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n, 0));

    omp_set_num_threads(Number_of_cores); // Adjust based on your system's high-performance cores

    auto start = std::chrono::high_resolution_clock::now();
    strassenMultiplyUnique(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_StrassenP.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_StrassenP.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Parallel Strassen matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}
// ******************************************************************************************************************** //
// sequential starassen matrix multiplication
void addMatricesUniqueSeq(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtractMatricesUniqueSeq(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void multiplyMatricesSeq(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int n) {
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                // Multiply the block sub-matrices
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
                        long sum = 0;
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

void strassenMultiplyUniqueSeq(const std::vector<std::vector<long>>& A, const std::vector<std::vector<long>>& B, std::vector<std::vector<long>>& C, int size) {
    if (size <= threshold) {
        multiplyMatricesSeq(A, B, C, size);
        return;
    }

    int newSize = size / 2;
    std::vector<std::vector<long>> A11(newSize, std::vector<long>(newSize)), A12(newSize, std::vector<long>(newSize)), A21(newSize, std::vector<long>(newSize)), A22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> B11(newSize, std::vector<long>(newSize)), B12(newSize, std::vector<long>(newSize)), B21(newSize, std::vector<long>(newSize)), B22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> C11(newSize, std::vector<long>(newSize)), C12(newSize, std::vector<long>(newSize)), C21(newSize, std::vector<long>(newSize)), C22(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> M1(newSize, std::vector<long>(newSize)), M2(newSize, std::vector<long>(newSize)), M3(newSize, std::vector<long>(newSize)), M4(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> M5(newSize, std::vector<long>(newSize)), M6(newSize, std::vector<long>(newSize)), M7(newSize, std::vector<long>(newSize));
    std::vector<std::vector<long>> AResult(newSize, std::vector<long>(newSize)), BResult(newSize, std::vector<long>(newSize));

    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    addMatricesUniqueSeq(A11, A22, AResult, newSize); // A11 + A22
    addMatricesUniqueSeq(B11, B22, BResult, newSize); // B11 + B22
    strassenMultiplyUniqueSeq(AResult, BResult, M1, newSize); // M1 = (A11 + A22) * (B11 + B22)

    addMatricesUniqueSeq(A21, A22, AResult, newSize); // A21 + A22
    strassenMultiplyUniqueSeq(AResult, B11, M2, newSize); // M2 = (A21 + A22) * B11

    subtractMatricesUniqueSeq(B12, B22, BResult, newSize); // B12 - B22
    strassenMultiplyUniqueSeq(A11, BResult, M3, newSize); // M3 = A11 * (B12 - B22)

    subtractMatricesUniqueSeq(B21, B11, BResult, newSize); // B21 - B11
    strassenMultiplyUniqueSeq(A22, BResult, M4, newSize); // M4 = A22 * (B21 - B11)

    addMatricesUniqueSeq(A11, A12, AResult, newSize); // A11 + A12
    strassenMultiplyUniqueSeq(AResult, B22, M5, newSize); // M5 = (A11 + A12) * B22

    subtractMatricesUniqueSeq(A21, A11, AResult, newSize); // A21 - A11
    addMatricesUniqueSeq(B11, B12, BResult, newSize); // B11 + B12
    strassenMultiplyUniqueSeq(AResult, BResult, M6, newSize); // M6 = (A21 - A11) * (B11 + B12)

    subtractMatricesUniqueSeq(A12, A22, AResult, newSize); // A12 - A22
    addMatricesUniqueSeq(B21, B22, BResult, newSize); // B21 + B22
    strassenMultiplyUniqueSeq(AResult, BResult, M7, newSize); // M7 = (A12 - A22) * (B21 + B22)

    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C12[i][j] = M3[i][j] + M5[i][j];
            C21[i][j] = M2[i][j] + M4[i][j];
            C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

void runMatrixMultiplicationStrassenSequential() {
    std::string filename1 = "4096";
    std::string filename2 = "4096";
    std::string outputFilenameBase = "input5_";

    int n;
    std::vector<std::vector<long>> A, B;

    readMatrix(filename1, A, n);
    readMatrix(filename2, B, n);

    std::vector<std::vector<long>> C(n, std::vector<long>(n, 0));

    auto start = std::chrono::high_resolution_clock::now();
    strassenMultiplyUniqueSeq(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ostringstream oss;
    oss << outputFilenameBase << n << "_output_StrassenSeq.txt";
    std::string outputFilename = oss.str();

    writeMatrix(outputFilename, C, n);

    oss.str("");
    oss.clear();
    oss << outputFilenameBase << n << "_info_StrassenSeq.txt";
    std::string timeFilename = oss.str();

    writeTime(timeFilename, duration, n);

    std::cout << "Sequential Strassen matrix multiplication result saved to " << outputFilename << std::endl;
    std::cout << "Time taken saved to " << timeFilename << std::endl;
}


int main() {
    // Uncomment the function you want to run

//    mainGenrateMatrix(); // you will be prompted to enter the matrix size and filename

//    runMatrixMultiplication();
//    runMatrixMultiplicationParallel();
//    runMatrixMultiplicationDivideAndConquer();
//    runMatrixMultiplicationDivideAndConquerParallel();
      // runMatrixMultiplicationStrassenSequential();
//    runMatrixMultiplicationStrassenParallel(); 
    return 0;

}
