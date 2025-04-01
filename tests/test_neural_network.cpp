#define TOASTY_IMPLEMENTATION
extern "C" {
#include "toasty.h"
}
#include "activation_function.hpp"
#include "neural_network.hpp"

using namespace nnn;

struct NeuralNetworkInspector {
    std::vector<Layer> layers;
};

TEST(test_NeuralNetworkShouldBeConstructedProperly) {
    NeuralNetwork nn({ 2, 3, 1 });

    auto inspector = reinterpret_cast<NeuralNetworkInspector*>(&nn);

    TEST_ASSERT_EQUAL(2, inspector->layers.size());

    TEST_ASSERT_EQUAL(2, inspector->layers[0].weights.getRows());
    TEST_ASSERT_EQUAL(3, inspector->layers[0].weights.getCols());

    TEST_ASSERT_EQUAL(3, inspector->layers[1].weights.getRows());
    TEST_ASSERT_EQUAL(1, inspector->layers[1].weights.getCols());
}

TEST(test_NeuralNetworkConstructionShouldFailWhenLayerSizesAreLessThanTwo) {
    try {
        NeuralNetwork valid({ 2, 3 });
    } catch (std::runtime_error& e) {
        (void) e;
        TEST_ASSERT_TRUE(false);
    }

    try {
        NeuralNetwork invalid({ 2 });
    } catch (std::runtime_error& e) {
        (void) e;
        return;
    }
    TEST_ASSERT_TRUE(false);
}

TEST(test_PredictionShouldCalculateOutputMatrixProperly) {
    NeuralNetwork nn({ 2, 2, 1 });

    auto inspector = reinterpret_cast<NeuralNetworkInspector*>(&nn);
    inspector->layers[0].weights.fill(1.f);
    inspector->layers[0].biases.fill(2.f);

    inspector->layers[1].weights.fill(3.f);
    inspector->layers[1].biases.fill(4.f);

    Matrix input(1, 2);
    input.fill(1.f);

    Matrix output = nn.predict(input);

    Matrix withSigmoid(1, 1);
    withSigmoid(0, 0) = 4.f;
    withSigmoid = ActivationFunction::sigmoid(withSigmoid);
    withSigmoid(0, 0) = 2.f * 3.f * withSigmoid(0, 0) + 4.f;
    withSigmoid = ActivationFunction::sigmoid(withSigmoid);

    TEST_ASSERT_EQUAL(1, output.getRows());
    TEST_ASSERT_EQUAL(1, output.getCols());
    TEST_ASSERT_EQUAL_FLOAT(withSigmoid(0, 0), output(0, 0));
}

int main() {
    return RunTests();
}
