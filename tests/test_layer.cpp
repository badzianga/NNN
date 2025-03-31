#define TOASTY_IMPLEMENTATION
extern "C" {
#include "toasty.h"
}
#include "layer.hpp"

using namespace nnn;

TEST(test_LayerDefaultConstructorShouldCreateEmptyMatrices) {
    const Layer layer;

    TEST_ASSERT_EQUAL(0, layer.weights.getCols());
    TEST_ASSERT_EQUAL(0, layer.weights.getRows());

    TEST_ASSERT_EQUAL(0, layer.biases.getCols());
    TEST_ASSERT_EQUAL(0, layer.biases.getRows());
}

TEST(test_LayerShouldBeConstructedProperly) {
    const Layer layer(2, 4);

    TEST_ASSERT_EQUAL(2, layer.weights.getRows());
    TEST_ASSERT_EQUAL(4, layer.weights.getCols());

    TEST_ASSERT_EQUAL(1, layer.biases.getRows());
    TEST_ASSERT_EQUAL(4, layer.biases.getCols());
}

TEST(test_ForwardMethowShouldWorkProperly) {
    Layer layer(2, 4);
    layer.weights.fill(1.f);
    layer.biases.fill(1.f);

    Matrix input(1, 2);
    input.fill(1.f);

    Matrix output = layer.forward(input);

    TEST_ASSERT_EQUAL(1, output.getRows());
    TEST_ASSERT_EQUAL(4, output.getCols());

    for (int j = 0; j < 4; ++j) {
        TEST_ASSERT_EQUAL_FLOAT(3.f, output(0, j));
    }
}

int main() {
    return RunTests();
}
