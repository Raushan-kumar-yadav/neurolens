#pragma once 

#include <neurolens/tensor/tensor.hpp>

// Tensor opration here 
namespace neurolens {

    Tensor add(const Tensor& a , const Tensor& b );
    Tensor mul(const Tensor& a , const Tensor& b);
    Tensor scale(const Tensor& a , float s);
    Tensor Transpose(const Tensor& a );
    Tensor addBias(const Tensor& x , const Tensor& b);
    Tensor sum_rows(const Tensor& a);
}

// Matrix multiplication here 
namespace neurolens {
    Tensor matmul(const Tensor& a, const Tensor& b);
}