#pragma once 
#include "../include/neurolens/tensor/tensor.hpp"

namespace neurolens {

class MSELoss {
public:
    float forward(const Tensor& pred, const Tensor& target);
    Tensor backward();  

private:
    Tensor m_pred;
    Tensor m_target;
};

}
