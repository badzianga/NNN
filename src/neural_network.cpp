#include "neural_network.hpp"

namespace nnn {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    if (layerSizes.size() < 2) {
        throw std::runtime_error("NeuralNetwork::NeuralNetwork: there must be at least 2 layers");
    }

    for (int i = 1; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    Matrix output = input;
    for (Layer& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

void NeuralNetwork::randomize(float low, float high) {
    for (Layer& layer : layers) {
        layer.randomize(low, high);
    }
}

} // nnn
