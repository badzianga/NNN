#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <memory>
#include <vector>

namespace nnn {

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, const std::vector<float>& values);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    float operator()(int row, int col) const;
    float& operator()(int row, int col);

    [[nodiscard]] int getRows() const;
    [[nodiscard]] int getCols() const;
    [[nodiscard]] Matrix transposed() const;
    [[nodiscard]] Matrix elementwiseMultiply(const Matrix& other) const;

    void fill(float value);
    void randomize(float low, float high);
    void print() const;

private:
    int rows;
    int cols;
    std::unique_ptr<float[]> data;
};

} // nnn

#endif //MATRIX_HPP
