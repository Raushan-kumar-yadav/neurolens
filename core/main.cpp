#include <iostream>
#include <neurolens/ann/activations.hpp>
#include <neurolens/ann/linear.hpp>
#include <neurolens/ann/sequential.hpp>
#include <neurolens/loss/mse.hpp>

int main() {
    using namespace neurolens;

    Tensor x({4, 2}, {0,0,  0,1,  1,0,  1,1});
    Tensor target({4, 1}, {0, 1, 1, 0});

    //build the model: 2 inputs -> 4 neurons -> Tanh -> 1 output
    Sequential net;
    net.add<Linear>(2, 4);     
    net.add<Tanh>();
    net.add<Linear>(4, 1);

    MSELoss mse;

    // train it
    for (int epoch = 0; epoch < 2000; ++epoch) {
        Tensor pred = net.forward(x);         
        float loss = mse.forward(pred, target);

        net.backward(mse.backward());          
        net.step(0.1f);                  

        if (epoch % 200 == 0) std::cout << "Epoch " << epoch << " | Loss: " << loss << "\n";
    }

    // use it
    Tensor out = net.forward(x);
    for (std::size_t i = 0; i < 4; ++i)
        std::cout << x[i*2] << " XOR " << x[i*2+1] << " = " << out[i] << "\n";
}