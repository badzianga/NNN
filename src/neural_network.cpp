#include <algorithm>
#include <iostream>
#include "loss_function.hpp"
#include "neural_network.hpp"
#include <numeric>

namespace nnn {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    if (layerSizes.size() < 2) {
        throw std::runtime_error("NeuralNetwork::NeuralNetwork: there must be at least 2 layers");
    }

    for (int i = 1; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
    }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other) : layers(other.layers) {}

NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept {
    layers = std::move(other.layers);
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this != &other) {
        layers = other.layers;
    }

    return *this;
}

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other) noexcept {
    layers = std::move(other.layers);
    return *this;
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

void NeuralNetwork::train(const Matrix &X, const Matrix &Y, int epochs, float learningRate) {
    constexpr int populationSize = 30;
    constexpr float mutationRate = 0.5;

    std::vector<NeuralNetwork> population(populationSize, *this);

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // 1. error for each network
        std::vector<float> scores;
        for (NeuralNetwork& net : population) {
            Matrix output = net.predict(X);
            scores.push_back(LossFunction::meanSquaredError(output, Y));
        }

        // 2. selection
        std::vector<int> sortedIndices(populationSize);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(), [&](int a, int b) {
            return scores[a] < scores[b];
        });

        std::vector<NeuralNetwork> newPopulation;
        for (int i = 0; i < populationSize / 2; ++i) {
            newPopulation.push_back(population[sortedIndices[i]]);
        }

        // 3. mutation
        for (NeuralNetwork& net : newPopulation) {
            for (Layer& layer : net.layers) {
                for (int i = 0; i < layer.weights.getRows(); ++i) {
                    for (int j = 0; j < layer.weights.getCols(); ++j) {
                        if (static_cast<float>(rand()) / RAND_MAX < mutationRate) {
                            layer.weights(i, j) += (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f;
                        }
                    }
                }
                for (int j = 0; j < layer.biases.getCols(); ++j) {
                    if (static_cast<float>(rand()) / RAND_MAX < mutationRate) {
                        layer.biases(0, j) += (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f;
                    }
                }
            }
        }

        // 4. fill
        population = newPopulation;
        while (population.size() < populationSize) {
            population.push_back(population[rand() % (populationSize / 2)]);
        }

        if (epoch % 25 == 0)
            std::cout << "Epoch: " << epoch << " - least loss: " << scores[sortedIndices[0]] << '\n';
    }

    layers = population[0].layers;
}

} // nnn
