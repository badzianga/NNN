#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP
#include "matrix.hpp"

namespace nnn {

class LossFunction {
public:
    LossFunction() = delete;
    LossFunction(const LossFunction&) = delete;
    LossFunction(LossFunction&&) = delete;
    LossFunction& operator=(const LossFunction&) = delete;
    LossFunction& operator=(LossFunction&&) = delete;

    static Matrix meanSquaredError(const Matrix& predictions, const Matrix& targets);
};

}

#endif //LOSS_FUNCTION_HPP
