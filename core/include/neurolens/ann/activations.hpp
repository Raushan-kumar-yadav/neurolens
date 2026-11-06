#pragma once 
#include "../include/neurolens/tensor/tensor.hpp"

namespace neurolens { 
class ReLU {
    public:
        ReLU() = default;

    neurolens::Tensor forward(const neurolens::Tensor& x);

    private:
    Tensor m_input;
};


class Tanh{
    public:
    Tanh() = default;

    neurolens::Tensor forward(const neurolens::Tensor& x);

    private:

    Tensor m_input;
    Tensor m_output;
};

}