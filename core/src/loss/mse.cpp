#include <neurolens/loss/mse.hpp>


#include <stdexcept>

float neurolens::MSELoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.size() != target.size()) {
        throw std::invalid_argument("MSELoss: pred and target must have the same size.");
    }

    m_pred = pred;
    m_target = target;

    float sum = 0.0f;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        float diff = pred[i] - target[i];

        // sum of all difference
        sum += diff * diff;
    }

    // finding means
    return sum / static_cast<float>(pred.size());
}


neurolens::Tensor neurolens::MSELoss::backward() {
    Tensor grad_out(m_pred.shape());
    float scale = 2.0f / static_cast<float>(m_pred.size());

    for (std::size_t i = 0; i < m_pred.size(); ++i) {
        grad_out[i] = scale * (m_pred[i] - m_target[i]);
    }

    return grad_out;
}