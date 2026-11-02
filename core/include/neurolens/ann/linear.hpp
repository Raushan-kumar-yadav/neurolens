#pragma once 
#include "cstdint"
#include "../include/neurolens/tensor/tensor.hpp"

namespace neurolens {
class Liner{

    public:
    // in = numer of intput features , out = number of output features 
    Liner(int in , int out , std::uint32_t seed = 32);

    // forward pass method its will handle calculation 
    Tensor forward(const Tensor& x);

    Tensor& weight();
    const Tensor& weight() const;

    Tensor& bias();
    const Tensor& bias() const ;


    private:  // datas 
    int m_in;
    int m_out;
    Tensor m_weights; // (in, out)
    Tensor m_bias; // (1, out)
    Tensor m_input; // copy of the last x 
};

} // namespace
