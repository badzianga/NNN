#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP
#include "layer.hpp"
#include "matrix.hpp"
#include <vector>

namespace nnn {

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<int>& layerSizes);
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(const NeuralNetwork&&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&&) = delete;
    Matrix predict(const Matrix& input);
    void randomize(float low, float high);
private:
    std::vector<Layer> layers;
};

} // nnn

#endif //NEURAL_NETWORK_HPP
