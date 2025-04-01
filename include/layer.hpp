#ifndef LAYER_HPP
#define LAYER_HPP
#include "matrix.hpp"

namespace nnn {

class Layer {
public:
    Layer();
    Layer(int inputSize, int outputSize);
    Layer(const Layer& other) = delete;
    Layer(Layer&& other) noexcept;
    Layer& operator=(const Layer& other) = delete;
    Layer& operator=(const Layer&& other) = delete;
    [[nodiscard]] Matrix forward(const Matrix& input) const;
    void randomize(float low, float high);

    Matrix weights;
    Matrix biases;
};

} // nnn

#endif //LAYER_HPP
