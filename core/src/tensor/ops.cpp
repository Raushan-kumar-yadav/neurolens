#pragma once 
#include "../include/neurolens/tensor/ops.hpp"
#include "neurolens/tensor/tensor.hpp"
#include <cstddef>
#include <stdexcept>



// addition of two tensor 
neurolens::Tensor neurolens::add( const Tensor &a, const Tensor &b){
    if(!a.sameShape(b)){
        throw std::invalid_argument("Tensor must be same size");
    }

    Tensor c(a.shape());
    for (std::size_t i = 0;  i < a.size() ; ++i) {

        c[i] = a[i] + b[i];
    }
    return c;
}

// adding bias to a tensor with bias  tensor 
neurolens::Tensor neurolens::addBias(const Tensor& a, const Tensor& b) {
    std::size_t cols = static_cast<std::size_t>(a.dim(a.rank() - 1));

    if (cols != b.size() || static_cast<std::size_t>(b.dim(b.rank() - 1)) != cols) {
        throw std::invalid_argument("Bias size does not match last dimension of tensor");
    }

    Tensor c(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i % cols];
    }
    return c;
}

// multiplication of two tensor 
neurolens::Tensor neurolens::mul( const Tensor &a, const Tensor &b){
    if(!a.sameShape(b)){
        throw std::invalid_argument("Tensor must be same size");
    };

    Tensor c(a.shape());
    for (std::size_t i = 0;  i < a.size() ; ++i) {

        c[i] = a[i] * b[i];
    }
    return c;
}

    neurolens::Tensor neurolens::scale(const Tensor& a , float s){
        Tensor c(a.shape());

        for (std::size_t i = 0 ; i < a.size() ; ++i) {
            c[i] = a[i] * s;
        }

        return c;
    };

   // Transpose a 2D Tensor 
neurolens::Tensor neurolens::Transpose(const Tensor& a) {
    if (a.rank() != 2) {
        throw std::invalid_argument("transpose: input tensor must be 2D");
    }

    int rows = a.dim(0);
    int cols = a.dim(1);

    Tensor out({cols, rows});

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(c * rows + r)] = a[static_cast<std::size_t>(r * cols + c)];
        }
    }

    return out;
}

//  Sum across rows  
neurolens::Tensor neurolens::sum_rows(const Tensor& a) {
    if (a.rank() != 2) {
        throw std::invalid_argument("sum_rows: input tensor must be 2D");
    }

    int rows = a.dim(0);
    int cols = a.dim(1);

    Tensor out({1, cols});

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(c)] += a[static_cast<std::size_t>(r * cols + c)];
        }
    }

    return out;
}

//  2D Matrix Multiplication 
neurolens::Tensor neurolens::matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("matmul: both tensors must be 2D");
    }

    int M = a.dim(0);
    int K = a.dim(1);
    int K2 = b.dim(0);
    int N = b.dim(1);

    if (K != K2) {
        throw std::invalid_argument("matmul: inner dimensions must match (a.cols != b.rows)");
    }

    Tensor out({M, N});

    // i -> k -> j loop order for sequential CPU cache reads
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = a[static_cast<std::size_t>(i * K + k)];
            for (int j = 0; j < N; ++j) {
                out[static_cast<std::size_t>(i * N + j)] += a_ik * b[static_cast<std::size_t>(k * N + j)];
            }
        }
    }

    return out;
}


