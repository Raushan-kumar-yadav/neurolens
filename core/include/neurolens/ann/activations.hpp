#pragma once 
#include <neurolens/tensor/tensor.hpp>
#include <neurolens/module.hpp>

namespace neurolens { 
class ReLU :  public Module {
    public:
        ReLU() = default;

    neurolens::Tensor forward(const neurolens::Tensor& x) override;
    neurolens::Tensor backward(const Tensor& grad_output) override;

    private:
    Tensor m_input;
};


class Tanh : public Module{
    public:
    Tanh() = default;

    neurolens::Tensor forward(const neurolens::Tensor& x) override;
    neurolens::Tensor backward(const Tensor& grad_output) override;

    private:
    Tensor m_output;
};

class Sigmoid : public Module{

    public:
    Sigmoid() = default;

    neurolens::Tensor forward(const neurolens::Tensor& x) override;
    neurolens::Tensor backward(const Tensor& grad_output) override;

    private:
    Tensor m_output;
};


}