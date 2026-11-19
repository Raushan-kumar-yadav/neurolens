#pragma once
#include <neurolens/module.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace neurolens {

// A list of modules run one after another.
class Sequential : public Module {
public:
    
    template <typename LayerType, typename... Args>
    void add(Args&&... args) {
        m_layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }

    // add a layer you already created
    void add(std::unique_ptr<Module> layer);

    // run forward through all layers, in order
    Tensor forward(const Tensor& x) override;

    // run backward through all layers, in reverse order
    Tensor backward(const Tensor& grad_output) override;

    // update every layer
    void step(float lr) override;

private:
    std::vector<std::unique_ptr<Module>> m_layers;
};

}  // namespace neurolens
