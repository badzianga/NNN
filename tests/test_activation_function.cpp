#define TOASTY_IMPLEMENTATION
extern "C" {
#include "toasty.h"
}
#include "activation_function.hpp"

using namespace nnn;

TEST(test_SigmoidShouldCalculateProperResults) {
    Matrix input(1, 1);
    Matrix output;

    input(0, 0) = 0.f;
    output = ActivationFunction::sigmoid(input);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, output(0, 0));

    input(0, 0) = 1000.f;
    output = ActivationFunction::sigmoid(input);
    TEST_ASSERT_EQUAL_FLOAT(1.f, output(0, 0));

    input(0, 0) = -1000.f;
    output = ActivationFunction::sigmoid(input);
    TEST_ASSERT_EQUAL_FLOAT(0.f, output(0, 0));
}

int main() {
    return RunTests();
}
