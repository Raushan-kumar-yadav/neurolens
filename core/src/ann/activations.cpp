#include <neurolens/ann/activations.hpp>
#include <neurolens/tensor/tensor.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>



// the base formula is std::max(0,x[i])
neurolens::Tensor neurolens::ReLU::forward(const neurolens::Tensor& x){

    m_input = x;
    Tensor c(x.shape());

    for(size_t i = 0 ; i < x.size()  ; ++i){
        c[i] = std::max(0.0f,x[i]);
    }
    return c;
};



// tanh(x[i])
neurolens::Tensor neurolens::Tanh::forward(const neurolens::Tensor& x){

   
    Tensor c(x.shape());

    for(size_t i = 0 ; i < x.size()  ; ++i){
        c[i] = std::tanh(x[i]);
    }

    m_output = c;
    return c;
};



// 1/1 + e^-x
neurolens::Tensor neurolens::Sigmoid::forward(const neurolens::Tensor& x){

    
    Tensor c(x.shape());

    for(size_t i = 0 ; i < x.size()  ; ++i){
        c[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    m_output = c;
    return c;
};  

// Backward pass uses cached m_output (y)
neurolens::Tensor neurolens::Sigmoid::backward(const Tensor& grad_out) {
    Tensor grad_in(grad_out.shape());
    for (std::size_t i = 0; i < grad_out.size(); ++i) {
        float y = m_output[i];                      
        float local_deriv = y * (1.0f - y); // dy/dx = y * (1 - y)
        grad_in[i] = grad_out[i] * local_deriv; // Chain rule: dL/dy * dy/dx
    }
    return grad_in;
}

// Backward pass uses cached m_output (y)
neurolens::Tensor neurolens::Tanh::backward(const Tensor& grad_out) {
    Tensor grad_in(grad_out.shape());
    for (std::size_t i = 0; i < grad_out.size(); ++i) {
        float y = m_output[i];                      
        float local_deriv = 1.0f - y * y; // dy/dx = 1 - y^2
        grad_in[i] = grad_out[i] * local_deriv; // Chain rule: dL/dy * dy/dx
    }
    return grad_in;
}

neurolens::Tensor neurolens::ReLU::backward(const Tensor& grad_out) {
    Tensor grad_in(grad_out.shape());
    for (std::size_t i = 0; i < grad_out.size(); ++i) {
                             
        grad_in[i] = (m_input[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
    return grad_in;
}