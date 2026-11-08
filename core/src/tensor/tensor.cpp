#include "../include/neurolens/tensor/tensor.hpp"
#include <stdexcept>
#include <vector>

namespace {
std::size_t numel(const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor: shape must have at least one dimension");
    }
    std::size_t n = 1;
    for (int d : shape) {
        if (d <= 0) {
            throw std::invalid_argument("Tensor: every dimension must be > 0");
        }
        n *= static_cast<std::size_t>(d);  
    }
    return n;
}
}

neurolens::Tensor::Tensor(std::vector<int> shape, std::vector<float> values)
    : m_shape(std::move(shape)),
      m_data(std::move(values))
{
    if (m_data.size() != numel(m_shape))
        throw std::invalid_argument("Tensor: values.size() does not match shape");
}

neurolens::Tensor::Tensor(std::vector<int>shape):
m_shape(std::move(shape)),
m_data(numel(m_shape)){}      

neurolens::Tensor neurolens::Tensor::zeros(std::vector<int> shape){
    return neurolens::Tensor(std::move(shape));
}

neurolens::Tensor neurolens::Tensor::full(std::vector<int> shape, float v){
    Tensor t(std::move(shape));
    t.fill(v);
    return t;
}

const std::vector<int>& neurolens::Tensor::shape() const{
    return m_shape;
}

 neurolens::Tensor neurolens::Tensor::transpose() const{

    if (m_shape.size() != 2) {
        throw std::invalid_argument("transpose() currently supports 2D tensors only.");
    }

    std::size_t rows = m_shape[0];
    std::size_t cols = m_shape[1];

    Tensor result({static_cast<int>(cols),static_cast<int>(rows)});

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            // Element at (r, c)   to (c, r)  
            result[c * rows + r] = m_data[r * cols + c];
        }
    }

    return result;


 } ;


int neurolens::Tensor::rank() const{
    return static_cast<int>(m_shape.size());
}


std::size_t neurolens::Tensor::size()const {
    return m_data.size();
}


int neurolens::Tensor::dim(int axis)const{
    if(axis < 0 || axis >= rank()){
        throw std::invalid_argument("axis out of range");
    }
    return m_shape[axis];
}



// it will return first element from vector 
float* neurolens::Tensor::data(){
    return  m_data.data();
  };



const float* neurolens::Tensor::data() const {
    return  m_data.data();
  };



std::vector<float>& neurolens::Tensor::values(){
    return m_data;
};


const std::vector<float>& neurolens::Tensor::values() const{
return m_data;
}

 float& neurolens::Tensor::operator[](std::size_t k)
{
    return m_data[k];
}

const float& neurolens::Tensor::operator[](std::size_t k) const
{
    return m_data[k];
}

neurolens::Tensor& neurolens::Tensor::reShape(std::vector<int> new_shape)
{
    if (numel(new_shape) != m_data.size())
        throw std::invalid_argument("Tensor::reshape: element count must not change");

    m_shape = std::move(new_shape);
    return *this;
}

 void neurolens::Tensor::fill(float v)
{
    for (float& x : m_data)
        x = v;
}

bool neurolens::Tensor::sameShape(const Tensor& other) const
{
    return m_shape == other.m_shape;
}

std::string neurolens::Tensor::shape_str() const
{
    std::string s = "(";
    for (std::size_t k = 0; k < m_shape.size(); ++k) {
        if (k > 0) s += ", ";
        s += std::to_string(m_shape[k]);
    }
    s += ")";
    return s;
}

float& neurolens::Tensor::at(int i, int j)
{
     if (rank() != 2)
        throw std::invalid_argument("Tensor::at(i, j) needs a 2-D tensor");

     if (i < 0 || i >= dim(0) || j < 0 || j >= dim(1))
        throw std::out_of_range("Tensor::at: index out of range");

     return m_data[static_cast<std::size_t>(i) * dim(1) + j];
}

const float& neurolens::Tensor::at(int i, int j) const
{
    if (rank() != 2)
        throw std::invalid_argument("Tensor::at(i, j) needs a 2-D tensor");

    if (i < 0 || i >= dim(0) || j < 0 || j >= dim(1))
        throw std::out_of_range("Tensor::at: index out of range");

    return m_data[static_cast<std::size_t>(i) * dim(1) + j];
}