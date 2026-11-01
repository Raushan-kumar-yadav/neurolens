#include <doctest/doctest.h>
#include <neurolens/tensor/ops.hpp>

using namespace neurolens;

TEST_CASE("matmul 2x2") {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor c = matmul(a, b);
    CHECK(c.at(0, 0) == 19);
    CHECK(c.at(0, 1) == 22);
    CHECK(c.at(1, 0) == 43);
    CHECK(c.at(1, 1) == 50);
}

TEST_CASE("matmul non-square (2x3 times 3x2)") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = matmul(a, b);
    CHECK(c.dim(0) == 2);
    CHECK(c.dim(1) == 2);
    CHECK(c.at(0, 0) == 58);
    CHECK(c.at(1, 1) == 154);
}

TEST_CASE("matmul rejects mismatched shapes") {
    CHECK_THROWS(matmul(Tensor({2, 3}), Tensor({2, 3})));  // inner dims 3 != 2
}

TEST_CASE("add / mul / scale") {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    CHECK(add(a, b)[3] == 12);
    CHECK(mul(a, b)[1] == 12);
    CHECK(scale(a, 2.0f)[2] == 6);
}

TEST_CASE("add / mul reject different shapes") {
    Tensor a({2, 2});
    CHECK_THROWS(add(a, Tensor({3, 2})));
    CHECK_THROWS(mul(a, Tensor({3, 2})));
}

TEST_CASE("add does not modify its inputs") {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    add(a, b);
    CHECK(a[0] == 1);
    CHECK(b[0] == 5);
}

TEST_CASE("Transpose") {
    Tensor t = Transpose(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
    CHECK(t.dim(0) == 3);
    CHECK(t.dim(1) == 2);
    CHECK(t.at(0, 1) == 4);
    CHECK(t.at(2, 0) == 3);
}

TEST_CASE("sum_rows") {
    Tensor s = sum_rows(Tensor({2, 2}, {1, 2, 3, 4}));
    CHECK(s[0] == 4);
    CHECK(s[1] == 6);
}

TEST_CASE("addBias adds the bias to every row") {
    Tensor r = addBias(Tensor({2, 2}, {1, 2, 3, 4}), Tensor({2}, {10, 20}));
    CHECK(r.at(0, 0) == 11);
    CHECK(r.at(0, 1) == 22);
    CHECK(r.at(1, 0) == 13);
    CHECK(r.at(1, 1) == 24);
}
