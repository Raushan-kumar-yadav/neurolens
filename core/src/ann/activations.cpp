#include "../include/neurolens/ann/activations.hpp"
#include "neurolens/tensor/tensor.hpp"
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

    m_input = x;
    Tensor c(x.shape());

    for(size_t i = 0 ; i < x.size()  ; ++i){
        c[i] = std::tanh(x[i]);
    }
    m_output = c;
    return c;
};