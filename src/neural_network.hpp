#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>

class NeuralNetwork {
public:
    // Constructor: define la arquitectura de la red.
    // numInputs: Tamaño del vector de extractFeatures.
    // numHidden: Número de neuronas en la capa oculta (ej. 64, 128).
    // numOutputs: Número de acciones posibles (ej. 4: IZQ, DER, FUEGO, NADA).
    NeuralNetwork(int numInputs, int numHidden, int numOutputs, double learningRate);

    // Propagación hacia adelante: predice los Q-values para un estado dado.
    std::vector<double> predict(const std::vector<double>& inputs) const;

    // Retropropagación: entrena la red usando el error entre la predicción y el objetivo.
    void train(const std::vector<double>& inputs, const std::vector<double>& targets);

    // Guarda y carga los pesos de la red.
    void saveWeights(const std::string& filename) const;
    void loadWeights(const std::string& filename);

    // --- NUEVOS MÉTODOS PARA EL ALGORITMO GENÉTICO ---
    std::vector<double> getWeightsAsVector() const;
    void setWeightsFromVector(const std::vector<double>& weights);

private:
    int numInputs;
    int numHidden;
    int numOutputs;
    double learningRate;

    // Pesos y sesgos
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<double> bias_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_output;

    // Función de activación y su derivada
    static double relu(double x);
    static double relu_derivative(double x);
};

#endif // NEURAL_NETWORK_HPP