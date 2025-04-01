#include "activation_function.hpp"
#include "layer.hpp"

using namespace nnn;

Layer::Layer() = default;

Layer::Layer(int inputSize, int outputSize) : weights(inputSize, outputSize), biases(1, outputSize) {}

Layer::Layer(Layer&& other) noexcept : weights(std::move(other.weights)), biases(std::move(other.biases)) {}

Matrix Layer::forward(const Matrix &input) const {
    return ActivationFunction::sigmoid(input * weights + biases);
}

void Layer::randomize(float low, float high) {
    weights.randomize(low, high);
    biases.randomize(low, high);
}
