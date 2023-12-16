#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

#define NUM_THREADS 8

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);

		// Temporary pointers
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;
        int rows = m1->rows;
        int cols = m1->cols;
		
		// Perform matrix multiplication on the GPU
#		pragma omp target enter data map(to: rows, cols)
#		pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m1->rows * m1->cols])
#		pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m_entries[i][j] = m1_entries[i][j] * m2_entries[i][j];
			}
		}
#		pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
		return m;
	} else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		
		// Temporary pointers
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;
        int rows = m1->rows;
        int cols = m1->cols;
		
		// Perform operation on the GPU
#		pragma omp target enter data map(to: rows, cols)
#		pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m1->rows * m1->cols])
#		pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m_entries[i][j] = m1_entries[i][j] + m2_entries[i][j];
			}
		}
#		pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
		return m;
	} else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);

		// Temporary pointers
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;
        int rows = m1->rows;
        int cols = m1->cols;
		
		// Perform operation on the GPU
#		pragma omp target enter data map(to: rows, cols)
#		pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m1->rows * m1->cols])
#		pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m_entries[i][j] = m1_entries[i][j] - m2_entries[i][j];
			}
		}
#		pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
		return m;
	} else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);
// I don't know how to parallelize this method, due to a function being passed
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);

		// Temporary pointers
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;
        int m1_rows = m1->rows;
        int m1_cols = m1->cols;
		int m2_cols = m2->cols;
        
		// Perform dot product on the GPU
#		pragma omp target enter data map(to: m1_rows,m1_cols,m2_cols)
#		pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols],m2_entries[:m1->cols * m2->cols])
#		pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < m1_rows; i++) {
			for (int j = 0; j < m2_cols; j++) {
				double sum = 0;
				for (int k = 0; k < m1_cols; k++) {
					sum += m1_entries[i][k] * m2_entries[k][j];
				}
				m_entries[i][j] = sum;
			}
		}
#		pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);

	// Temporary pointers
	double **mat_entries = mat->entries;
	int m_rows = m->rows;
	int m_cols = m->cols;

	// Perform scaling on the GPU
#	pragma omp target enter data map(to: m_rows,m_cols,n)
#	pragma omp target enter data map(to: mat_entries[:m->rows * m->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m_rows; i++) {
		for (int j = 0; j < m_cols; j++) {
			mat_entries[i][j] *= n;
		}
	}
#	pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
	return mat;
}

Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);

	// Temporary pointers
	double **mat_entries = mat->entries;
	int m_rows = m->rows;
	int m_cols = m->cols;

	// Perform operation on the GPU
#	pragma omp target enter data map(to: m_rows,m_cols,n)
#	pragma omp target enter data map(to: mat_entries[:m->rows * m->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m_rows; i++) {
		for (int j = 0; j < m_cols; j++) {
			mat_entries[i][j] += n;
		}
	}
#	pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
	return mat;
}

Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);

	// Temporary pointers
	double **mat_entries = mat->entries;
	int m_rows = m->rows;
	int m_cols = m->cols;

	// Perform transpose on the GPU
#	pragma omp target enter data map(to: m_rows,m_cols)
#	pragma omp target enter data map(to: mat_entries[:m->rows * m->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m_rows; i++) {
		for (int j = 0; j < m_cols; j++) {
			mat_entries[j][i] = mat_entries[i][j];
		}
	}
#	pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
	return mat;
}