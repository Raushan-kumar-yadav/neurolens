#include "../include/neurolens/ann/linear.hpp"
#include "neurolens/tensor/ops.hpp"
#include "neurolens/tensor/tensor.hpp"
#include <cmath>
#include <random>
#include <stdexcept>


neurolens::Liner::Liner(int in , int out , std::uint32_t seed ) : m_in(in) , m_out(out),m_weights({in , out}),m_bias({1 , out }),m_input({1, in}){

    if(in <= 0 || out <=0){
        throw std::invalid_argument("Liner: 'in' and 'out' dimensions must be > 0");
    }

    std::mt19937 rng(seed);
    float limit  = std::sqrt(6.0f / static_cast<float>(in + out));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (std::size_t i = 0; i < m_weights.size(); ++i) {
        m_weights[i] = dist(rng);
    }
  
};

neurolens::Tensor& neurolens::Liner::weight(){
    return m_weights;
};


const neurolens::Tensor& neurolens::Liner::weight() const {
    return m_weights;
};

neurolens::Tensor& neurolens::Liner::bias(){
    return m_bias;
};


const neurolens::Tensor& neurolens::Liner::bias()const {
    return m_bias;
};


// formulas (x*w + x*w + bias)
 neurolens::Tensor neurolens::Liner::forward(const Tensor& x){

    m_input = x; // Cache input eg: previos layer output

    Tensor xy = neurolens::matmul(x, m_weights);
    Tensor y = neurolens::addBias(xy, m_bias);

    return y;

 };

 
// formulas ( x*w + bias)
neurolens::Tensor neurolens::Liner::backward(const Tensor& grad_output) {

 
    m_grad_weights = neurolens::matmul(m_input.transpose(), grad_output);

    // Bias gradient 
    std::size_t rows = grad_output.shape()[0];
    std::size_t cols = grad_output.shape()[1];

    m_grad_bias = Tensor({1, static_cast<int>(cols)});

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            m_grad_bias[c] += grad_output[r * cols + c];
        }
    }

    // Input gradient to pass to the previous layer 
    Tensor grad_in = neurolens::matmul(grad_output, m_weights.transpose());

    return grad_in;
}
