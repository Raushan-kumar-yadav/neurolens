#pragma once 
#include "../include/neurolens/module.hpp"
#include <memory>
#include <utility>
#include <vector>


namespace neurolens {


class Sequential : public Module{

public:

    // create directly
    template<typename LayerType , typename... Args>
    void add(Args&&... args){
        m_layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }
    // pass it as ptr
    void add(std::unique_ptr<Module> layer) {
        m_layers.push_back(std::move(layer));
    }

    // run fwd of all layers 
    Tensor forward(const Tensor& x) override {
        Tensor current = x;
        for (auto& layer : m_layers) {
            current = layer->forward(current);
        }
        return current;
    }

    // run backward for all layers 
    Tensor backward(const Tensor& grad_output) override {
        Tensor current_grad = grad_output;
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
        return current_grad;
    }

    // run steps calc for all layers 
    void step(float lr) override {
        for (auto& layer : m_layers) {
            layer->step(lr);
        }
    }

private:
std::vector<std::unique_ptr<Module>> m_layers;

};}