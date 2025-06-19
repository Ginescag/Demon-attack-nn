#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>

class Perceptron {
private:
    std::vector<float> weights;  // Pesos del perceptrón
    float bias;                  // Sesgo
    float learningRate;          // Tasa de aprendizaje

public:
    Perceptron(int numInputs, float learningRate = 0.01);

    // Predecir la salida basada en las entradas
    int predict(const std::vector<float>& inputs);

    // Entrenar el perceptrón con una entrada y salida esperada
    void train(const std::vector<float>& inputs, int target);

    // Establecer pesos entrenados desde un archivo o vector
    void setWeights(const std::vector<float>& newWeights);

    // Obtener los pesos actuales (para guardarlos en un archivo)
    std::vector<float> getWeights() const;
};

#endif
