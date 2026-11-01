#include <doctest/doctest.h>
#include <neurolens/tensor/tensor.hpp>

using neurolens::Tensor;

TEST_CASE("Tensor(shape) is all zeros") {
    Tensor t({2, 3});
    CHECK(t.size() == 6);
    CHECK(t.rank() == 2);
    CHECK(t.dim(0) == 2);
    CHECK(t.dim(1) == 3);
    for (std::size_t k = 0; k < t.size(); ++k)
        CHECK(t[k] == 0.0f);
}

TEST_CASE("Tensor(shape, values) and at(i, j)") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(t.at(0, 0) == 1);
    CHECK(t.at(0, 2) == 3);
    CHECK(t.at(1, 0) == 4);
    CHECK(t.at(1, 2) == 6);
}

TEST_CASE("at() can write") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    t.at(1, 2) = 9.0f;
    CHECK(t[5] == 9.0f);
}

TEST_CASE("at() rejects bad indices") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS(t.at(2, 0));
    CHECK_THROWS(t.at(0, 3));
    CHECK_THROWS(t.at(-1, 0));
}

TEST_CASE("dim() rejects bad axes") {
    Tensor t({2, 3});
    CHECK_THROWS(t.dim(2));
    CHECK_THROWS(t.dim(-1));
}

TEST_CASE("bad shapes throw") {
    CHECK_THROWS(Tensor(std::vector<int>{2, 0}));
    CHECK_THROWS(Tensor(std::vector<int>{}));
    CHECK_THROWS(Tensor({2, 3}, {1, 2, 3}));  // 3 values for 6 slots
}

TEST_CASE("reShape keeps the numbers, changes the view") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    t.reShape({3, 2});
    CHECK(t.at(1, 0) == 3);
    CHECK_THROWS(t.reShape({4, 2}));  // 8 != 6
}

TEST_CASE("zeros, full, fill") {
    Tensor z = Tensor::zeros({2, 2});
    CHECK(z.size() == 4);
    CHECK(z[3] == 0.0f);

    Tensor f({2, 2});
    f.fill(7.0f);
    CHECK(f[0] == 7.0f);
    CHECK(f[3] == 7.0f);
}

TEST_CASE("sameShape and shape_str") {
    Tensor a({2, 3}), b({2, 3}), c({3, 2});
    CHECK(a.sameShape(b));
    CHECK_FALSE(a.sameShape(c));
    CHECK(a.shape_str() == "(2, 3)");
}
