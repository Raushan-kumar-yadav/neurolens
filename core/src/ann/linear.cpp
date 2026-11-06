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
