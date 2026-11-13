#include <iostream>
#include "include/neurolens/tensor/tensor.hpp"
#include "include/neurolens/ann/linear.hpp"
#include "include/neurolens/ann/activations.hpp"
#include "include/neurolens/loss/mse.hpp"
 
int main() {
    using namespace neurolens;

    // XOR Dataset: 4 samples, 2 input features -> shape {4, 2}
    Tensor x({4, 2}, {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    });

    // XOR Targets: 4 samples, 1 output feature -> shape {4, 1}
    Tensor target({4, 1}, {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    });

    // Build Network: 2 -> 4 (Tanh) -> 1
    Liner layer1(2, 4);
    Tanh act1;
    Liner layer2(4, 1);
    MSELoss mse;

    float lr = 0.1f;
    int epochs = 2000;

    // Training Loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward Pass  
        Tensor h1 = layer1.forward(x);
        Tensor a1 = act1.forward(h1);
        Tensor pred = layer2.forward(a1);

        float loss = mse.forward(pred, target);

        if (epoch % 200 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss << "\n";
        }

        // Backward Pass 
        Tensor grad_loss = mse.backward();
        Tensor grad_a1 = layer2.backward(grad_loss);
        Tensor grad_h1 = act1.backward(grad_a1);
        Tensor grad_x = layer1.backward(grad_h1);

        // Update Weights 
        layer1.step(lr);
        layer2.step(lr);
    }

    // Final Predictions Check
    Tensor final_pred = layer2.forward(act1.forward(layer1.forward(x)));
    std::cout << "\nFinal Predictions:\n";
    for (std::size_t i = 0; i < 4; ++i) {
        std::cout << x[i * 2] << " XOR " << x[i * 2 + 1] 
                  << " = " << final_pred[i] 
                  << " (Target: " << target[i] << ")\n";
    }

    return 0;
}