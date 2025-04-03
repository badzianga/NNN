#include <algorithm>
#include <cassert>
#include <iostream>
#include "matrix.hpp"
#include <random>

namespace nnn {

Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(std::make_unique<float[]>(rows * cols)) {
    std::fill_n(data.get(), rows * cols, 0.f);
}

Matrix::Matrix(int rows, int cols, const std::vector<float>& values) : rows(rows), cols(cols) {
    if (values.size() != rows * cols) {
        throw std::runtime_error("Matrix::Matrix: `values.size()` should be the same as `rows * cols`");
    }
    data = std::make_unique<float[]>(rows * cols);
    std::copy_n(values.begin(), rows * cols, data.get());
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    data = std::make_unique<float[]>(rows * cols);
    std::copy_n(other.data.get(), rows * cols, data.get());
}

Matrix::Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (rows != other.rows || cols != other.cols) {
            data.reset(new float[other.rows * other.cols]);
            rows = other.rows;
            cols = other.cols;
        }
        std::copy_n(other.data.get(), rows * cols, data.get());
    }
    return *this;
}

Matrix & Matrix::operator=(Matrix&& other) noexcept {
    rows = other.rows;
    cols = other.cols;
    data = std::move(other.data);

    other.rows = 0;
    other.cols = 0;

    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (cols != other.cols) {
        throw std::runtime_error("Matrix::operator+: matrix columns do not match");
    }

    const int newRows = rows > other.rows ? rows : other.rows;
    Matrix result(newRows, cols);

    for (int i = 0; i < newRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i % rows, j) + other(i % other.rows, j);
        }
    }

    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (cols != other.cols) {
        throw std::runtime_error("Matrix::operator+=: matrix columns do not match");
    }

    if (rows < other.rows) {
        throw std::runtime_error(
            "Matrix::operator+=: Left hand-side rows should be greater or equal than right hand-side rows"
        );
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            (*this)(i, j) += other(i % other.rows, j);
        }
    }

    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator-: matrix dimensions do not match");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }

    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator-=: matrix dimensions do not match");
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            (*this)(i, j) -= other(i, j);
        }
    }

    return *this;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Matrix::operator*: invalid matrix dimensions");
    }

    Matrix result(rows, other.cols);
    const int n = cols;

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result(i, j) = 0.f;
            for (int k = 0; k < n; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }

    return result;
}


float Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= rows) {
        throw std::runtime_error("Matrix::operator(): row index out of range");
    }
    if (col < 0 || col >= cols) {
        throw std::runtime_error("Matrix::operator(): column index out of range");
    }
    assert(row >= 0 && row < rows && col >= 0 && col < cols);
    return data[row * cols + col];
}

float& Matrix::operator()(int row, int col) {
    if (row < 0 || row >= rows) {
        throw std::runtime_error("Matrix::operator(): row index out of range");
    }
    if (col < 0 || col >= cols) {
        throw std::runtime_error("Matrix::operator(): column index out of range");
    }
    return data[row * cols + col];
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

Matrix Matrix::transposed() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::fill(float value) {
    std::fill_n(data.get(), rows * cols, value);
}

void Matrix::randomize(float low, float high) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution dis(low, high);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            (*this)(i, j) = dis(gen);
        }
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << ' ';
        }
        std::cout << '\n';
    }
}

} // nnn
