// Gradient checks: compare backward() with a numerical estimate.
//   numeric = ( loss(v + eps) - loss(v - eps) ) / (2 * eps)
// If backward() is right, the two agree. This is what catches wrong derivatives.
#include <doctest/doctest.h>

#include <neurolens/ann/activations.hpp>
#include <neurolens/ann/linear.hpp>
#include <neurolens/ann/sequential.hpp>
#include <neurolens/loss/mse.hpp>

using namespace neurolens;

namespace {

const float EPS = 1e-3f;
const double TOL = 0.01;  // doctest::Approx: |a - n| < TOL * (1 + |n|)

// loss of module(x) against target; no gradients involved
float loss_of(Module& m, const Tensor& x, const Tensor& target) {
    MSELoss mse;
    return mse.forward(m.forward(x), target);
}

// forward, loss, backward; returns dLoss/dInput
Tensor run_backward(Module& m, const Tensor& x, const Tensor& target) {
    MSELoss mse;
    mse.forward(m.forward(x), target);
    return m.backward(mse.backward());
}

// nudge every input value and compare with the gradient backward() returned
void check_input_grad(Module& m, Tensor x, const Tensor& target) {
    Tensor analytic = run_backward(m, x, target);
    REQUIRE(analytic.sameShape(x));

    for (std::size_t k = 0; k < x.size(); ++k) {
        float original = x[k];
        x[k] = original + EPS;
        float loss_plus = loss_of(m, x, target);
        x[k] = original - EPS;
        float loss_minus = loss_of(m, x, target);
        x[k] = original;

        float numeric = (loss_plus - loss_minus) / (2.0f * EPS);
        INFO("input index " << k);
        CHECK(analytic[k] == doctest::Approx(numeric).epsilon(TOL));
    }
}

// nudge every value of one parameter tensor of a Linear layer
void check_param_grad(Linear& fc, Tensor& param, const Tensor& analytic, const Tensor& x, const Tensor& target) {
    REQUIRE(analytic.sameShape(param));
    for (std::size_t k = 0; k < param.size(); ++k) {
        float original = param[k];
        param[k] = original + EPS;
        float loss_plus = loss_of(fc, x, target);
        param[k] = original - EPS;
        float loss_minus = loss_of(fc, x, target);
        param[k] = original;

        float numeric = (loss_plus - loss_minus) / (2.0f * EPS);
        INFO("parameter index " << k);
        CHECK(analytic[k] == doctest::Approx(numeric).epsilon(TOL));
    }
}

Tensor make_x() {
    return Tensor({4, 3}, {0.5f, -1.2f, 0.8f, 1.5f, 0.3f, -0.7f, -0.4f, 0.9f, 1.1f, 0.2f, -0.6f, -1.3f});
}

}  // namespace

TEST_CASE("gradcheck: Linear weights") {
    Linear fc(3, 2, 7);
    Tensor x = make_x();
    Tensor target({4, 2}, {1, 0, 0, 1, 1, 1, 0, 0});
    run_backward(fc, x, target);
    Tensor analytic = fc.grad_weight();  // copy: later forwards must not change it
    check_param_grad(fc, fc.weight(), analytic, x, target);
}

TEST_CASE("gradcheck: Linear bias") {
    Linear fc(3, 2, 7);
    Tensor x = make_x();
    Tensor target({4, 2}, {1, 0, 0, 1, 1, 1, 0, 0});
    run_backward(fc, x, target);
    Tensor analytic = fc.grad_bias();
    check_param_grad(fc, fc.bias(), analytic, x, target);
}

TEST_CASE("gradcheck: Linear input") {
    Linear fc(3, 2, 7);
    check_input_grad(fc, make_x(), Tensor({4, 2}, {1, 0, 0, 1, 1, 1, 0, 0}));
}

TEST_CASE("gradcheck: Tanh input") {
    Tanh act;
    check_input_grad(act, make_x(), Tensor({4, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 0, -1}));
}

TEST_CASE("gradcheck: Sigmoid input") {
    Sigmoid act;
    check_input_grad(act, make_x(), Tensor({4, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 0, 1}));
}

TEST_CASE("gradcheck: ReLU input (values are away from the kink at 0)") {
    ReLU act;
    check_input_grad(act, make_x(), Tensor({4, 3}, {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1}));
}

TEST_CASE("gradcheck: MSELoss gradient matches its own formula") {
    Tensor pred({2, 2}, {0.3f, -0.8f, 1.2f, 0.5f});
    Tensor target({2, 2}, {1.0f, 0.0f, 0.5f, -0.5f});
    MSELoss mse;
    mse.forward(pred, target);
    Tensor analytic = mse.backward();

    for (std::size_t k = 0; k < pred.size(); ++k) {
        float original = pred[k];
        MSELoss a, b;
        pred[k] = original + EPS;
        float loss_plus = a.forward(pred, target);
        pred[k] = original - EPS;
        float loss_minus = b.forward(pred, target);
        pred[k] = original;
        float numeric = (loss_plus - loss_minus) / (2.0f * EPS);
        CHECK(analytic[k] == doctest::Approx(numeric).epsilon(TOL));
    }
}

TEST_CASE("gradcheck: whole Sequential network input") {
    Sequential net;
    net.add<Linear>(3, 4, 11);
    net.add<Tanh>();
    net.add<Linear>(4, 2, 13);
    check_input_grad(net, make_x(), Tensor({4, 2}, {1, 0, 0, 1, 1, 1, 0, 0}));
}
