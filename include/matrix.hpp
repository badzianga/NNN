#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <memory>

namespace nnn {

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    float operator()(int row, int col) const;
    float& operator()(int row, int col);

    [[nodiscard]] int getRows() const;
    [[nodiscard]] int getCols() const;

    void fill(float value);

private:
    int rows;
    int cols;
    std::unique_ptr<float[]> data;
};

} // nnn

#endif //MATRIX_HPP
