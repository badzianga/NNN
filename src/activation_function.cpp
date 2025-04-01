#include "activation_function.hpp"
#include <cmath>
using namespace nnn;

Matrix ActivationFunction::sigmoid(const Matrix &x) {
    Matrix result(x.getRows(), x.getCols());

    for (int i = 0; i < x.getRows(); i++) {
        for (int j = 0; j < x.getCols(); j++) {
            result(i, j) = 1.f / (1.f + std::exp(-x(i, j)));
        }
    }

    return result;
}
