#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP
#include "layer.hpp"
#include "matrix.hpp"
#include <vector>

namespace nnn {

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<int>& layerSizes);
    NeuralNetwork(const NeuralNetwork& other);
    NeuralNetwork(NeuralNetwork&& other) noexcept;
    NeuralNetwork& operator=(const NeuralNetwork& other);
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;
    Matrix predict(const Matrix& input);
    void randomize(float low, float high);
    void train(const Matrix& X, const Matrix& Y, int epochs, float learningRate);
private:
    std::vector<Layer> layers;
};

} // nnn

#endif //NEURAL_NETWORK_HPP
