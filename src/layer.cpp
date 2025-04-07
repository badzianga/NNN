#include "activation_function.hpp"
#include "layer.hpp"

using namespace nnn;

Layer::Layer() = default;

Layer::Layer(int inputSize, int outputSize) : weights(inputSize, outputSize), biases(1, outputSize) {}

Layer::Layer(const Layer& other) {
    weights = other.weights;
    biases = other.biases;
}

Layer::Layer(Layer&& other) noexcept : weights(std::move(other.weights)), biases(std::move(other.biases)) {}

Layer& Layer::operator=(const Layer& other) {
    if (this != &other) {
        weights = other.weights;
        biases = other.biases;
    }

    return *this;
}

Layer& Layer::operator=(Layer&& other) noexcept {
    weights = std::move(other.weights);
    biases = std::move(other.biases);

    return *this;
}

Matrix Layer::forward(const Matrix &input) const {
    return ActivationFunction::sigmoid(input * weights + biases);
}

void Layer::randomize(float low, float high) {
    weights.randomize(low, high);
    biases.randomize(low, high);
}
