#pragma once 
#include <cstddef>
#include <string>
#include <vector>

namespace neurolens {

class Tensor{
    public:

    Tensor(){};
    Tensor(std::vector<int>shape , std::vector<float>value);
    explicit Tensor(std::vector<int>shape);
    static Tensor zeros(std::vector<int> shape);
    static Tensor full(std::vector<int> shape, float v);



    // shape    
    const std::vector<int>& shape() const;
    int rank() const;

    // size and dim
    std::size_t size()const ;
    int dim(int axis)const;

    // data access 
    float* data();
    const float* data()const;

    // value access 
    std::vector<float>& values();
    const std::vector<float>& values() const;

    // Element access 
    float& at(int i , int j );
    const float& at(int i  , int j ) const ;
    float& operator[](std::size_t k);
    const float& operator[](std::size_t k) const ;  

    // Re-shape 
    Tensor& reShape(std::vector<int> newShape);

    // utils 
    void fill(float v);
    bool sameShape(const Tensor& other) const ;
    std::string shape_str() const;

    private:

    // store 
    std::vector<int> m_shape;
    std::vector<float> m_data ;


};
};