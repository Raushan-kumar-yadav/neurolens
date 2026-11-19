#include <neurolens/ann/sequential.hpp>

void neurolens::Sequential::add(std::unique_ptr<Module> layer) {
    m_layers.push_back(std::move(layer));
}

neurolens::Tensor neurolens::Sequential::forward(const Tensor& x) {
    Tensor current = x;
    for (auto& layer : m_layers) {
        current = layer->forward(current);
    }
    return current;
}

neurolens::Tensor neurolens::Sequential::backward(const Tensor& grad_output) {
    Tensor current_grad = grad_output;
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
    return current_grad;
}

void neurolens::Sequential::step(float lr) {
    for (auto& layer : m_layers) {
        layer->step(lr);
    }
}
