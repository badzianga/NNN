#include <cmath>
#include "loss_function.hpp"

using namespace nnn;

Matrix LossFunction::meanSquaredError(const Matrix& predictions, const Matrix& targets) {
    if (predictions.getCols() != targets.getCols() || predictions.getRows() != targets.getRows()) {
        throw std::runtime_error("LossFunction::meanSquaredError: matrices' dimensions are not equal");
    }

    Matrix result(predictions.getRows(), 1);
    for (int i = 0; i < predictions.getRows(); ++i) {
        for (int j = 0; j < predictions.getCols(); ++j) {
            result(i, 0) += std::pow(predictions(i, j) - targets(i, j), 2.f);
        }
        result(i, 0) /= static_cast<float>(predictions.getCols());
    }

    return result;
}
