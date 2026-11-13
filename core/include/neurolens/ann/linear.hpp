#pragma once 
#include <cstdint>
#include <neurolens/tensor/tensor.hpp>

namespace neurolens {
class Linear{

    public:
    // in = numer of intput features , out = number of output features 
    Linear(int in , int out , std::uint32_t seed = 32);

    // forward pass method its will handle calculation 
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    // optimizer steps 

    void step(float lr);
    
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

    Tensor m_grad_weights;
    Tensor m_grad_bias;
    
};

} // namespace
