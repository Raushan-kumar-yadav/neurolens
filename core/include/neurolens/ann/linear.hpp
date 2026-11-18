#pragma once 
#include <cstdint>
#include <neurolens/tensor/tensor.hpp>
#include <neurolens/module.hpp>


namespace neurolens {
class Linear : public Module{

    public:
    // in = numer of intput features , out = number of output features 
    Linear(int in , int out , std::uint32_t seed = 32);

    // forward pass method its will handle calculation 
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output) override;

    // optimizer steps 

    void step(float lr) override;
    
    Tensor& weight();
    const Tensor& weight() const;

    Tensor& bias();
    const Tensor& bias() const ;

    // gradients from the last backward() call (read-only, used by tests and optimizers)
    const Tensor& grad_weight() const;
    const Tensor& grad_bias() const;


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
