#ifndef LAYER_HPP
#define LAYER_HPP
#include "matrix.hpp"

namespace nnn {

class Layer {
public:
    Layer();
    Layer(int inputSize, int outputSize);
    Layer(const Layer&) = delete;
    Layer(const Layer&&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer& operator=(const Layer&&) = delete;
    [[nodiscard]] Matrix forward(const Matrix& input) const;

    Matrix weights;
    Matrix biases;
};

} // nnn

#endif //LAYER_HPP
