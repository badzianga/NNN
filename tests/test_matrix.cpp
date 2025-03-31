#define TOASTY_IMPLEMENTATION
extern "C" {
#include "toasty.h"
}
#include "matrix.hpp"

using namespace nnn;

struct MatrixInspector {
    int rows;
    int cols;
    std::unique_ptr<float[]> data;
};

TEST(test_DefaultMatrixConstructorShouldCreateEmptyMatrix) {
    const Matrix matrix;

    const auto inspector = reinterpret_cast<const MatrixInspector*>(&matrix);

    TEST_ASSERT_EQUAL(0, inspector->rows);
    TEST_ASSERT_EQUAL(0, inspector->cols);
    TEST_ASSERT_NULL(inspector->data.get());
}

TEST(test_ConstructorShouldCreateMatrixFilledWithZeros) {
    const Matrix matrix(3, 2);

    const auto inspector = reinterpret_cast<const MatrixInspector*>(&matrix);

    TEST_ASSERT_EQUAL(3, inspector->rows);
    TEST_ASSERT_EQUAL(2, inspector->cols);
    for (int i = 0; i < inspector->rows * inspector->cols; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(0.f, inspector->data[i]);
    }
}

TEST(test_ConstParenthesisOperatorShouldReturnProperValues) {
    Matrix matrix(3, 2);
    const auto inspector = reinterpret_cast<MatrixInspector*>(&matrix);

    for (int i = 0; i < 3 * 2; ++i) {
        inspector->data[i] = static_cast<float>(i);
    }

    for (int i = 0; i < 3 * 2; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(static_cast<float>(i), inspector->data[i]);
    }
}

TEST(test_ParenthesisOperatorShouldSetValues) {
    Matrix matrix(3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            matrix(i, j) = static_cast<float>(i * 2 + j);
        }
    }

    const auto inspector = reinterpret_cast<MatrixInspector*>(&matrix);

    for (int i = 0; i < 3 * 2; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(static_cast<float>(i), inspector->data[i]);
    }
}

TEST(test_SizeGettersShouldReturnCorrectValues) {
    const Matrix matrix(3, 2);

    TEST_ASSERT_EQUAL(3, matrix.getRows());
    TEST_ASSERT_EQUAL(2, matrix.getCols());
}

TEST(test_CopyConstructorShouldCreateExactCopyOfMatrix) {
    Matrix original(1, 2);
    original(0, 0) = 1.f;
    original(0, 1) = 2.f;

    Matrix copy(original);

    TEST_ASSERT_EQUAL(original.getRows(), copy.getRows());
    TEST_ASSERT_EQUAL(original.getCols(), copy.getCols());

    for (int i = 0; i < original.getRows(); ++i) {
        for (int j = 0; j < original.getCols(); ++j) {
            TEST_ASSERT_EQUAL_FLOAT(original(i, j), copy(i, j));
        }
    }
}

TEST(test_MoveConstructorShouldMoveToNewMatrix) {
    Matrix original(1, 2);
    original(0, 0) = 1.f;
    original(0, 1) = 2.f;

    Matrix moved(std::move(original));

    TEST_ASSERT_EQUAL(1, moved.getRows());
    TEST_ASSERT_EQUAL(2, moved.getCols());

    TEST_ASSERT_EQUAL_FLOAT(1.f, moved(0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, moved(0, 1));
}

TEST(test_CopyOperatorShouldCreateExactCopyOfMatrix) {
    Matrix original(1, 2);
    original(0, 0) = 1.f;
    original(0, 1) = 2.f;

    Matrix copy = original;

    TEST_ASSERT_EQUAL(original.getRows(), copy.getRows());
    TEST_ASSERT_EQUAL(original.getCols(), copy.getCols());

    for (int i = 0; i < original.getRows(); ++i) {
        for (int j = 0; j < original.getCols(); ++j) {
            TEST_ASSERT_EQUAL_FLOAT(original(i, j), copy(i, j));
        }
    }
}

TEST(test_MoveOperatorShouldMoveToNewMatrix) {
    Matrix original(1, 2);
    original(0, 0) = 1.f;
    original(0, 1) = 2.f;

    Matrix moved = std::move(original);

    TEST_ASSERT_EQUAL(1, moved.getRows());
    TEST_ASSERT_EQUAL(2, moved.getCols());

    TEST_ASSERT_EQUAL_FLOAT(1.f, moved(0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, moved(0, 1));
}

TEST(test_AdditionOperatorShouldSumMatrices) {
    Matrix a(1, 2);
    a(0, 0) = 1.f;
    a(0, 1) = 2.f;
    Matrix b(1, 2);
    b(0, 0) = 3.f;
    b(0, 1) = 4.f;

    Matrix c = a + b;

    TEST_ASSERT_EQUAL(1, c.getRows());
    TEST_ASSERT_EQUAL(2, c.getCols());

    TEST_ASSERT_EQUAL_FLOAT(a(0, 0) + b(0, 0), c(0, 0));
    TEST_ASSERT_EQUAL_FLOAT(a(0, 1) + b(0, 1), c(0, 1));
}

TEST(test_AdditionOperatorShouldThrowErrorIfDimensionsDoNotMatch) {
    Matrix a(1, 2);
    Matrix b(3, 4);

    try {
        Matrix c = a + b;
    } catch (std::runtime_error& e) {
        // if error is caught, end test with success, else fail at next assertion
        (void) e;
        return;
    }
    TEST_ASSERT_TRUE(false);
}

TEST(test_AdditionAssignmentOperatorShouldAddToMatrix) {
    Matrix a(1, 2);

    Matrix b(1, 2);
    b(0, 0) = 1.f;
    b(0, 1) = 2.f;

    a += b;

    TEST_ASSERT_EQUAL(1, a.getRows());
    TEST_ASSERT_EQUAL(2, a.getCols());

    TEST_ASSERT_EQUAL_FLOAT(b(0, 0), a(0, 0));
    TEST_ASSERT_EQUAL_FLOAT(b(0, 1), a(0, 1));
}

TEST(test_AdditionAssignmentOperatorShouldThrowErrorIfDimensionsDoNotMatch) {
    Matrix a(1, 2);
    Matrix b(3, 4);

    try {
        a += b;
    } catch (std::runtime_error& e) {
        // if error is caught, end test with success, else fail at next assertion
        (void) e;
        return;
    }
    TEST_ASSERT_TRUE(false);
}

TEST(test_MultiplicationOperatorShouldMultiplyMatrices) {
    Matrix a(1, 2);
    a(0, 0) = 1.f;
    a(0, 1) = 2.f;
    Matrix b(2, 3);
    b(0, 0) = 3.f;
    b(0, 1) = 4.f;
    b(0, 2) = 5.f;
    b(1, 0) = 6.f;
    b(1, 1) = 7.f;
    b(1, 2) = 8.f;

    Matrix c = a * b;

    TEST_ASSERT_EQUAL(1, c.getRows());
    TEST_ASSERT_EQUAL(3, c.getCols());

    TEST_ASSERT_EQUAL_FLOAT(15.f, c(0, 0));
    TEST_ASSERT_EQUAL_FLOAT(18.f, c(0, 1));
    TEST_ASSERT_EQUAL_FLOAT(21.f, c(0, 2));
}

TEST(test_MultiplicationOperatorShouldThrowErrorIfDimensionsAreInvalid) {
    Matrix a(1, 2);
    Matrix b(3, 2);

    try {
        Matrix c = a * b;
    } catch (std::runtime_error& e) {
        // if error is caught, end test with success, else fail at next assertion
        (void) e;
        return;
    }
    TEST_ASSERT_TRUE(false);
}

TEST(test_FillMethodShouldFillMatrixWithValue) {
    Matrix a(2, 4);
    a.fill(3.34f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            TEST_ASSERT_EQUAL_FLOAT(3.34f, a(i, j));
        }
    }
}

int main() {
    return RunTests();
}
