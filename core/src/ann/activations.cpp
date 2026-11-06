#include "../include/neurolens/ann/activations.hpp"
#include "neurolens/tensor/tensor.hpp"
#include <algorithm>
#include <cstddef>



// the base formula is std::max(0,x[i])
neurolens::Tensor neurolens::ReLU::forward(const neurolens::Tensor& x){

    m_input = x;
    Tensor c(x.shape());

    for(size_t i = 0 ; i < x.size()  ; ++i){
        c[i] = std::max<size_t>(0,x[i]);
    }
    return c;
};