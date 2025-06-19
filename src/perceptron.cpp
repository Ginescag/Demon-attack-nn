#include "perceptron.hpp"
#include <numeric>  // std::inner_product
#include <stdexcept>  // Para std::invalid_argument

// Constructor
Perceptron::Perceptron(int numInputs, float learningRate)
    : learningRate(learningRate), bias(0.0f) {
    weights.resize(numInputs, 0.0f);  // Inicializa los pesos en 0
}

// Predicción
int Perceptron::predict(const std::vector<float>& inputs) {
    // Producto interno entre pesos y entradas, más el sesgo
    float sum = std::inner_product(weights.begin(), weights.end(), inputs.begin(), bias);

    // Activación binaria: devuelve 1 si la suma es mayor o igual a 0, de lo contrario, 0
    return (sum >= 0.0f) ? 1 : 0;
}

// Entrenamiento
void Perceptron::train(const std::vector<float>& inputs, int target) {
    int prediction = predict(inputs);   // Predicción actual
    int error = target - prediction;   // Error entre predicción y objetivo

    // Ajuste de pesos basado en el error
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += learningRate * error * inputs[i];
    }
    bias += learningRate * error;  // Ajuste del sesgo
}

// Establecer nuevos pesos
void Perceptron::setWeights(const std::vector<float>& newWeights) {
    if (newWeights.size() == weights.size()) {
        weights = newWeights;
    } else {
        throw std::invalid_argument("El tamaño del vector de pesos no coincide con el número de entradas del perceptrón.");
    }
}

// Obtener los pesos actuales
std::vector<float> Perceptron::getWeights() const {
    return weights;
}
