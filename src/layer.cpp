#include "layer.hpp"

using namespace nnn;

Layer::Layer() = default;

Layer::Layer(int inputSize, int outputSize) : weights(inputSize, outputSize), biases(1, outputSize) {}

Matrix Layer::forward(const Matrix &input) const {
    return input * weights + biases;
}
