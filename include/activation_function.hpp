#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP
#include "matrix.hpp"

namespace nnn {

class ActivationFunction {
public:
    static Matrix sigmoid(const Matrix& x);
};

} // nnn

#endif //ACTIVATION_FUNCTION_HPP
