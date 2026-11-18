#include <doctest/doctest.h>

#include <neurolens/ann/activations.hpp>
#include <neurolens/ann/linear.hpp>
#include <neurolens/ann/sequential.hpp>
#include <neurolens/loss/mse.hpp>

using namespace neurolens;

// ---------- Linear ----------

TEST_CASE("Linear forward with known weights") {
    Linear fc(2, 2);
    fc.weight() = Tensor({2, 2}, {1, 2, 3, 4});
    fc.bias() = Tensor({1, 2}, {1, 1});

    Tensor y = fc.forward(Tensor({1, 2}, {1, 1}));
    CHECK(y.dim(0) == 1);
    CHECK(y.dim(1) == 2);
    CHECK(y.at(0, 0) == 5);  // 1*1 + 1*3 + 1
    CHECK(y.at(0, 1) == 7);  // 1*2 + 1*4 + 1
}

TEST_CASE("Linear forward handles a batch") {
    Linear fc(2, 2);
    fc.weight() = Tensor({2, 2}, {1, 2, 3, 4});
    fc.bias() = Tensor({1, 2}, {0, 0});

    Tensor y = fc.forward(Tensor({2, 2}, {1, 0, 0, 1}));  // picks row 0 and row 1 of W
    CHECK(y.at(0, 0) == 1);
    CHECK(y.at(0, 1) == 2);
    CHECK(y.at(1, 0) == 3);
    CHECK(y.at(1, 1) == 4);
}

TEST_CASE("Linear rejects bad sizes") {
    CHECK_THROWS(Linear(0, 2));
    CHECK_THROWS(Linear(2, 0));
    Linear fc(2, 2);
    CHECK_THROWS(fc.forward(Tensor({1, 3})));  // 3 features, layer expects 2
}

TEST_CASE("Linear: same seed gives same weights, different seed differs") {
    Linear a(3, 4, 7), b(3, 4, 7), c(3, 4, 8);
    bool differs = false;
    for (std::size_t k = 0; k < a.weight().size(); ++k) {
        CHECK(a.weight()[k] == b.weight()[k]);
        if (a.weight()[k] != c.weight()[k]) differs = true;
    }
    CHECK(differs);
}

TEST_CASE("Linear backward with known numbers") {
    Linear fc(2, 2);
    fc.weight() = Tensor({2, 2}, {1, 2, 3, 4});
    fc.bias() = Tensor({1, 2}, {1, 1});

    fc.forward(Tensor({1, 2}, {1, 1}));
    Tensor grad_in = fc.backward(Tensor({1, 2}, {1, 1}));

    // dW = x^T @ g = [[1,1],[1,1]], db = [1,1], dx = g @ W^T = [[1+2, 3+4]]
    CHECK(fc.grad_weight().at(0, 0) == 1);
    CHECK(fc.grad_weight().at(1, 1) == 1);
    CHECK(fc.grad_bias()[0] == 1);
    CHECK(fc.grad_bias()[1] == 1);
    CHECK(grad_in.at(0, 0) == 3);
    CHECK(grad_in.at(0, 1) == 7);
}

TEST_CASE("Linear step moves weights against the gradient") {
    Linear fc(2, 2);
    fc.weight() = Tensor({2, 2}, {1, 2, 3, 4});
    fc.bias() = Tensor({1, 2}, {1, 1});
    fc.forward(Tensor({1, 2}, {1, 1}));
    fc.backward(Tensor({1, 2}, {1, 1}));
    fc.step(0.1f);

    CHECK(fc.weight().at(0, 0) == doctest::Approx(0.9f));
    CHECK(fc.weight().at(1, 1) == doctest::Approx(3.9f));
    CHECK(fc.bias()[0] == doctest::Approx(0.9f));
}

// ---------- activations ----------

TEST_CASE("ReLU forward and backward") {
    ReLU relu;
    Tensor y = relu.forward(Tensor({1, 3}, {-1, 0, 2}));
    CHECK(y.dim(1) == 3);
    CHECK(y[0] == 0);
    CHECK(y[1] == 0);
    CHECK(y[2] == 2);

    Tensor g = relu.backward(Tensor({1, 3}, {1, 1, 1}));
    CHECK(g[0] == 0);  // input was negative
    CHECK(g[1] == 0);  // input was exactly 0
    CHECK(g[2] == 1);  // input was positive
}

TEST_CASE("Tanh forward and backward") {
    Tanh act;
    Tensor y = act.forward(Tensor({1, 2}, {0, 1}));
    CHECK(y[0] == doctest::Approx(0.0f));
    CHECK(y[1] == doctest::Approx(0.761594f));

    Tensor g = act.backward(Tensor({1, 2}, {1, 1}));
    CHECK(g[0] == doctest::Approx(1.0f));  // 1 - 0^2
    CHECK(g[1] == doctest::Approx(1.0f - 0.761594f * 0.761594f));
}

TEST_CASE("Sigmoid forward and backward") {
    Sigmoid act;
    Tensor y = act.forward(Tensor({1, 2}, {0, 2}));
    CHECK(y[0] == doctest::Approx(0.5f));
    CHECK(y[1] == doctest::Approx(0.880797f));

    Tensor g = act.backward(Tensor({1, 2}, {1, 1}));
    CHECK(g[0] == doctest::Approx(0.25f));  // 0.5 * (1 - 0.5)
}

// ---------- loss ----------

TEST_CASE("MSELoss value and gradient") {
    MSELoss mse;
    float loss = mse.forward(Tensor({1, 2}, {1, 2}), Tensor({1, 2}, {1, 4}));
    CHECK(loss == doctest::Approx(2.0f));  // (0^2 + 2^2) / 2

    Tensor g = mse.backward();  // 2/2 * (pred - target)
    CHECK(g[0] == doctest::Approx(0.0f));
    CHECK(g[1] == doctest::Approx(-2.0f));
}

TEST_CASE("MSELoss rejects different sizes") {
    MSELoss mse;
    CHECK_THROWS(mse.forward(Tensor({1, 2}), Tensor({1, 3})));
}

// ---------- Sequential ----------

TEST_CASE("Sequential matches chaining the layers by hand") {
    Sequential net;
    net.add<Linear>(2, 3, 5);
    net.add<Tanh>();
    net.add<Linear>(3, 1, 7);

    Linear a(2, 3, 5);
    Tanh t;
    Linear b(3, 1, 7);

    Tensor x({2, 2}, {0.5f, -1.0f, 1.5f, 2.0f});
    Tensor via_net = net.forward(x);
    Tensor by_hand = b.forward(t.forward(a.forward(x)));

    CHECK(via_net.sameShape(by_hand));
    for (std::size_t k = 0; k < by_hand.size(); ++k)
        CHECK(via_net[k] == doctest::Approx(by_hand[k]));
}
