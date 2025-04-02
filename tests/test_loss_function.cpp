#define TOASTY_IMPLEMENTATION
extern "C" {
#include "toasty.h"
}
#include "loss_function.hpp"

using namespace nnn;

TEST(test_MeanSquaredErrorShouldCalculateProperResults) {
    const Matrix predicted(4, 2, {
        0.f,   0.f,
        1.f,   1.f,
        0.5f,  0.5f,
        0.75f, 0.75f
    });
    const Matrix target(4, 2, {
        0.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        1.f, 1.f
    });

    const Matrix expected(4, 1, {
        0.f,
        0.5f,
        0.25f,
        0.0625f,
    });

    const Matrix actual = LossFunction::meanSquaredError(predicted, target);

    TEST_ASSERT_EQUAL(expected.getRows(), actual.getRows());
    TEST_ASSERT_EQUAL(expected.getCols(), actual.getCols());

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(expected(i, 0), actual(i, 0));
    }
}

int main() {
    return RunTests();
}
