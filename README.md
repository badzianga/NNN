# NNN

Nice Neural Network is a small library for creating simple artificial neural networks in C++.
Unlike possible implementation that use explicit `Neuron` class, this library is based on matrix operations,
resulting in faster computations and reduced memory usage.

## Build Instructions
A `CMakeLists.txt` file is provided for building the project.
To build the library, run the following commands:

```bash
mkdir build && cd build
cmake ..
make NNN
```

## Example
The code below shows an example of training a model to behave like an XOR gate.
```C++
#include "neural_network.hpp"

int main() {
    // Define a network with 4 layers:
    // input (2 neurons), 1st hidden (5 neurons), 2nd hidden (4 neurons), output (3 neurons)
    nnn::NeuralNetwork nn({ 2, 5, 4, 3 });
    
    // Randomize weights and biases (range (-1; 1))
    nn.randomize(-1.f, 1.f);

    // Create an input matrix (1x2)
    nnn::Matrix input(1, 2);
    
    // Fill input matrix with values
    input(0, 0) = 0.5f;
    input(0, 1) = 0.8f;

    // Perform a forward pass
    nnn::Matrix output = nn.predict(input);
    
    return 0;
}
```

## Example
The code below show 
```C++
#include "neural_network.hpp"

int main() {
    srand(time(nullptr));
    
    // Define a network with 4 layers:
    // input (2 neurons), 1st hidden (5 neurons), 2nd hidden (4 neurons), output (3 neurons)
    NeuralNetwork nn({ 2, 3, 1 });
    
    // Randomize weights and biases (range (-1; 1))
    nn.randomize(-1.f, 1.f);
    
    // Create an input matrix (4 samples, 2 inputs each)
    Matrix inputs(4, 2, {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    });
    
    // Create a target matrix (4 samples, 1 output each)
    Matrix outputs(4, 1, {
        0,
        1,
        1,
        0,
    });
    
    // Print behavior of the model before training
    std::cout << "Before training:\n";
    Matrix predictions = nn.predict(inputs);
    for (int i = 0; i < inputs.getRows(); ++i) {
        std::cout << inputs(i, 0) << " ^ " << inputs(i, 1) << " = " << predictions(i, 0) << '\n';
    }
    
    // Train the model using genetic algorithm (250 epochs, 0.1 mutation chance)
    nn.train(inputs, outputs, 250, 0.1);

    // Print behavior of the model after training    
    std::cout << "After training:\n";
    predictions = nn.predict(inputs);
    for (int i = 0; i < inputs.getRows(); ++i) {
        std::cout << inputs(i, 0) << " ^ " << inputs(i, 1) << " = " << predictions(i, 0) << '\n';
    }
    
    return 0;
}
```
## Components

### Matrix (`matrix.hpp`)

A fundamental data structure for handling numerical operations.

```C++
nnn:Matrix mat1(3, 3);
nnn:Matrix mat2(3, 3);
nnn:Matrix result = mat1 + mat2;
```

### Layer (`layer.hpp`)

Represents a single layer in the neural network, containing weights and biases.

### Activation Functions (`activation_function.hpp`)

Includes a static methods for applying activation functions.

```C++
nnn:Matrix activated = nnn::ActivationFunction::sigmoid(mat1);
```

### Neural Network (`neural_network.hpp`)

A simple feedforward neural network implementation.

## Future Improvements

Currently, the library is work-in-progress.
Below is the list of improvements which will be added soon:

- Implement backpropagation
- Add more activation functions
- Apply different activation function for each layer
- Write full documentation
