#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

#define NUM_THREADS 24

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix* m = matrix_create(m1->rows, m1->cols);
		// Temporary pointers for the data mapping
		double** m1_entries = m1->entries;
		double** m2_entries = m2->entries;
		double** m_entries = m->entries;
		// Map the data to the device (GPU)
#pragma omp target enter data map(to: m1[:1], m2[:1])
#pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
#pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
			// Perform matrix multiplication on the GPU
#pragma omp target teams distribute parallel for collapse(2)
			thread_limit(NUM_THREADS)
			for (int i = 0; i < m1->rows; i++) {
				for (int j = 0; j < m2->cols; j++) {
					m_entries[i][j] = m1_entries[i][j] * m2_entries[i][j];
				}
			}
		// Retrieve the results from the device (you can merge this codes to above map to as we
#pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
				return m;
	}
	else {
		printf("Dimension mismatch multiply: %dx%d %dx%d\n",
			m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		// Temporary pointers for the data mapping
		double** m1_entries = m1->entries;
		double** m2_entries = m2->entries;
		double** m_entries = m->entries;
		// Map the data to the device (GPU)
#pragma omp target enter data map(to: m1[:1], m2[:1])
#pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
#pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
			// Perform matrix multiplication on the GPU
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		// Retrieve the results from the device (you can merge this codes to above map to as we
#pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
		return m;
	} else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {

		Matrix *m = matrix_create(m1->rows, m1->cols);
		
		// Temporary pointers for the data mapping
		double** m1_entries = m1->entries;
		double** m2_entries = m2->entries;
		double** m_entries = m->entries;
		// Map the data to the device (GPU)
#pragma omp target enter data map(to: m1[:1], m2[:1])
#pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
#pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
#pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
		return m;
	} else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);
#pragma omp target enter data map(to: m[:1])
#pragma omp target enter data map(to: m_entries[:m->rows * m->cols]
#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
#pragma omp target exit data map(from: mat_entries[:mat->rows * mat->cols])
#pragma omp target exit data map(delete: m_entries[:m->rows * m->cols])
	return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);
		double** m1_entries = m1->entries;
		double** m2_entries = m2->entries;
		double** m_entries = m->entries;
		// Map the data to the device (GPU)
#pragma omp target enter data map(to: m1[:1], m2[:1])
#pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
#pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])
#		pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}
#pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols],
		m2_entries[:m2->rows * m2->cols])
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#pragma omp target enter data map(to: m[:1])
#pragma omp target enter data map(to: m_entries[:m->rows * m->cols]
#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
#pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m_entries[:m->rows * m->cols])

	return mat;
}

Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#pragma omp target enter data map(to: m[:1])
#pragma omp target enter data map(to: m_entries[:m->rows * m->cols]
#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
#pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m_entries[:m->rows * m->cols])
	return mat;
}

Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
	
	double** m_entries = m->entries;
	// Map the data to the device (GPU)
#pragma omp target enter data map(to: m[:1])
#pragma omp target enter data map(to: m_entries[:m->rows * m->cols]
#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
#	pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < m->rows; i++) {
#       pragma omp parallel for num_threads(12) collapse(2)
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
#pragma omp target exit data map(from: mat_entries[:m->rows * m->cols])
#pragma omp target exit data map(delete: m_entries[:m->rows * m->cols])
	return mat;
}