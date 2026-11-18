#pragma once 
#include <neurolens/tensor/tensor.hpp>


namespace neurolens {
class Module{
public:
virtual ~Module()=default;

virtual Tensor forward(const Tensor& x ) = 0;
virtual Tensor backward(const Tensor& grad_output) = 0;

// step method 
virtual void step(float lr){};
};}