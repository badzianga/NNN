#ifndef TOASTY_H
#define TOASTY_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*TestFunc)();

typedef struct {
    const char* name;
    TestFunc func;
} TestCase;

extern void (*toasty__SetUp)();
extern void(*toasty__TearDown)();

void toasty__RegisterTest(const char* name, TestFunc func);
void toasty__IncrementFail();
const char* toasty__GetCurrentTestName();
void toasty__SegfaultHandler(int signum);
int toasty__RunTests(const char* fileName);
float toasty__AbsFloat(float x);
double toasty__AbsDouble(double x);

#define TOASTY_EPS_FLOAT 1e-6f
#define TOASTY_EPS_DOUBLE 1e-9

#define RunTests() toasty__RunTests(__FILE__)

#define TEST(name) \
    void name(); \
    __attribute__((constructor)) void register_##name() { \
        toasty__RegisterTest(#name, name); \
    } \
    void name()

#define FAIL(msg, file, line) do { \
    printf("\033[1;91m[FAIL]\033[m %s %s:%d: %s\n", toasty__GetCurrentTestName(), file, line, msg); \
    toasty__IncrementFail(); \
    return; \
} while(0)

#define TEST_ASSERT_TRUE(condition) if (!(condition)) { \
    FAIL("Condition is false", __FILE__, __LINE__); \
}

#define TEST_ASSERT_FALSE(condition) if ((condition)) { \
    FAIL("Condition is true", __FILE__, __LINE__); \
}

#define TEST_ASSERT_EQUAL(expected, actual) if ((expected) != (actual)) { \
    char msg[64]; \
    snprintf(msg, sizeof(msg), "Expected %d, but got %d", (int)(expected), (int)(actual)); \
    FAIL(msg, __FILE__, __LINE__); \
}

#define TEST_ASSERT_NULL(pointer) if ((pointer)) { \
    FAIL("Pointer is not null", __FILE__, __LINE__); \
}

#define TEST_ASSERT_NOT_NULL(pointer) if (!(pointer)) { \
    FAIL("Pointer is null", __FILE__, __LINE__); \
}

#define TEST_ASSERT_EQUAL_INT(expected, actual) do { \
    int expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %d, but got %d", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_INT8(expected, actual) do { \
    int8_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %d, but got %d", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_INT16(expected, actual) do { \
    int16_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %hd, but got %hd", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_INT32(expected, actual) do { \
    int32_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %d, but got %d", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_INT64(expected, actual) do { \
    int64_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %ld, but got %ld", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_UINT(expected, actual) do { \
    unsigned int expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %u, but got %u", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_UINT8(expected, actual) do { \
    uint8_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %u, but got %u", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_UINT16(expected, actual) do { \
    uint16_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %hu, but got %hu", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_UINT32(expected, actual) do { \
    uint32_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %u, but got %u", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_UINT64(expected, actual) do { \
    uint64_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %lu, but got %lu", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_HEX(expected, actual) do { \
    unsigned int expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected 0x%08X, but got 0x%08X", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_HEX8(expected, actual) do { \
    uint8_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected 0x%02X, but got 0x%02X", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_HEX16(expected, actual) do { \
    uint16_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected 0x%04X, but got 0x%04X", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_HEX32(expected, actual) do { \
    uint32_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected 0x%08X, but got 0x%08X", expected_, actual_); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_HEX64(expected, actual) do { \
    uint64_t expected_ = (expected), actual_ = (actual); \
    if (expected_ != actual_) { \
        char msg[64]; \
        snprintf( \
            msg, sizeof(msg), "Expected 0x%08X%08X, but got 0x%08X%08X", \
            (uint32_t)(expected_ >> 32), (uint32_t)(expected_ & 0xFFFFFFFF), \
            (uint32_t)(actual_ >> 32), (uint32_t)(actual_ & 0xFFFFFFFF) \
        ); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_FLOAT(expected, actual) do { \
    if (toasty__AbsFloat((expected) - (actual)) > TOASTY_EPS_FLOAT) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %.6f, but got %.6f", expected, actual); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#define TEST_ASSERT_EQUAL_DOUBLE(expected, actual) do { \
    if (toasty__AbsDouble((expected) - (actual)) > TOASTY_EPS_DOUBLE) { \
        char msg[64]; \
        snprintf(msg, sizeof(msg), "Expected %.9lf, but got %.9lf", expected, actual); \
        FAIL(msg, __FILE__, __LINE__); \
    } \
} while (0)

#endif // TOASTY_H

#ifdef TOASTY_IMPLEMENTATION

#ifndef TOASTY_IGNORE_SEGFAULTS
#include <setjmp.h>
#include <signal.h>
#endif // TOASTY_IGNORE_SEGFAULTS

#ifndef TOASTY_MAX_TESTS
#define TOASTY_MAX_TESTS 100
#endif // TOASTY_MAX_TESTS

static TestCase toasty__tests[TOASTY_MAX_TESTS];
static size_t toasty__testCount = 0;
static size_t toasty__testsPassed = 0;
static size_t toasty__testsFailed = 0;
static const char* toasty__currentTestName = NULL;
static const char* toasty__fileName = NULL;

#ifndef TOASTY_IGNORE_SEGFAULTS
static jmp_buf toasty__jmpBuf;
static int toasty__segfaultCaught = 0;
#endif // TOASTY_IGNORE_SEGFAULTS


void (*toasty__SetUp)() = NULL;
void (*toasty__TearDown)() = NULL;
void SetUp() __attribute__((weak));
void TearDown() __attribute__((weak));

__attribute__((constructor)) static void toasty__Initialize() {
    if (SetUp != NULL) toasty__SetUp = SetUp;
    if (TearDown != NULL) toasty__TearDown = TearDown;
}

void toasty__RegisterTest(const char *name, TestFunc func) {
    if (toasty__testCount < TOASTY_MAX_TESTS) {
        toasty__tests[toasty__testCount++] = (TestCase){ name, func };
    }
    else {
        printf(
            "Max number of tests exceeded. (current amount: %zu)\n" \
            "Increase the max number using the following macro in the file with implementation:\n" \
            "#define TOASTY_MAX_TESTS n\n", \
            toasty__testCount \
        );
        exit(EXIT_FAILURE);
    }
}

void toasty__IncrementFail() {
    ++toasty__testsFailed;
}

const char* toasty__GetCurrentTestName() {
    return toasty__currentTestName;
}

#ifndef TOASTY_IGNORE_SEGFAULTS
void toasty__SegfaultHandler(int signum) {
    (void)signum;
    printf("\033[1;91m[FAIL]\033[m %s %s: Caught SEGFAULT!\n", toasty__currentTestName, toasty__fileName);
    toasty__segfaultCaught = 1;
    longjmp(toasty__jmpBuf, 1);
}
#endif // TOASTY_IGNORE_SEGFAULTS

int toasty__RunTests(const char* fileName) {
    toasty__fileName = fileName;
    printf("\033[1;96mRunning %zu tests from %s:\033[m\n", toasty__testCount, toasty__fileName);
    
#ifndef TOASTY_IGNORE_SEGFAULTS
    struct sigaction sa;
    sa.sa_handler = toasty__SegfaultHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, NULL);
#endif // TOASTY_IGNORE_SEGFAULTS

    for (size_t i = 0; i < toasty__testCount; ++i) {
        if (toasty__SetUp) toasty__SetUp();

        toasty__currentTestName = toasty__tests[i].name;
        size_t beforeFailCount = toasty__testsFailed;
        
#ifndef TOASTY_IGNORE_SEGFAULTS
        toasty__segfaultCaught = 0;
        if (setjmp(toasty__jmpBuf) == 0) {
            toasty__tests[i].func();
        } else {
            ++toasty__testsFailed;
        }
#else
        toasty__tests[i].func();
#endif
        
        if (beforeFailCount == toasty__testsFailed) {
            printf("\033[1;92m[PASS]\033[m %s\n", toasty__currentTestName);
            ++toasty__testsPassed;
        }

        if (toasty__TearDown) toasty__TearDown();
    }

    printf("\033[1;96m========== TEST SUMMARY ==========\033[m\n");
    printf(
        "\033[1mTotal:\033[m %zu | \033[1mPassed:\033[m %zu | \033[1mFailed:\033[m %zu\n",
        toasty__testCount, toasty__testsPassed, toasty__testsFailed
    );
    if (toasty__testsFailed > 0) {
        printf("\033[1;91mSome tests failed!\033[m\n\n");
        return EXIT_FAILURE;
    } else {
        printf("\033[1;92mAll tests passed!\033[m\n\n");
        return EXIT_SUCCESS;
    }
}

float toasty__AbsFloat(float x) {
    return x > 0.f ? x : -x;
}

double toasty__AbsDouble(double x) {
    return x > 0.0 ? x : -x;
}

#endif // TOASTY_IMPLEMENTATION
