#include "neural_network.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Constructor
NeuralNetwork::NeuralNetwork(int numInputs, int numHidden, int numOutputs, double learningRate)
    : numInputs(numInputs), numHidden(numHidden), numOutputs(numOutputs), learningRate(learningRate) {

    // Inicialización de pesos y sesgos con valores aleatorios pequeños
    std::mt19937 gen(std::random_device{}());
    // Usamos la inicialización más conservadora para evitar valores extremos
    std::normal_distribution<> dist_ih(0.0, std::sqrt(1.0 / numInputs));
    std::normal_distribution<> dist_ho(0.0, std::sqrt(1.0 / numHidden));

    weights_input_hidden.resize(numHidden, std::vector<double>(numInputs));
    for (auto& row : weights_input_hidden) {
        for (auto& val : row) {
            val = dist_ih(gen);
            // Asegurar que ningún peso inicial sea extremo
            val = std::max(-0.5, std::min(0.5, val));
        }
    }

    bias_hidden.resize(numHidden, 0.0);

    weights_hidden_output.resize(numOutputs, std::vector<double>(numHidden));
    for (auto& row : weights_hidden_output) {
        for (auto& val : row) {
            val = dist_ho(gen);
            // Asegurar que ningún peso inicial sea extremo
            val = std::max(-0.5, std::min(0.5, val));
        }
    }

    bias_output.resize(numOutputs, 0.0);
}

// Función de activación ReLU
double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

// Derivada de ReLU
double NeuralNetwork::relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Predicción (propagación hacia adelante)
std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) const {
    // Verificar entrada NaN
    for (const auto& val : inputs) {
        if (std::isnan(val) || std::isinf(val)) {
            std::cerr << "¡Alerta! Entrada NaN o Inf detectada en predict()." << std::endl;
            return std::vector<double>(numOutputs, 0.0); // Devolver ceros
        }
    }

    // Capa oculta
    std::vector<double> hidden(numHidden);
    for (int i = 0; i < numHidden; ++i) {
        double sum = bias_hidden[i];
        for (int j = 0; j < numInputs; ++j) {
            sum += inputs[j] * weights_input_hidden[i][j];
            if (std::isnan(sum) || std::isinf(sum)) {
                std::cerr << "¡Alerta! Sum NaN/Inf en capa oculta de predict()." << std::endl;
                return std::vector<double>(numOutputs, 0.0);
            }
        }
        hidden[i] = relu(sum);
    }

    // Capa de salida (lineal, sin activación para Q-values)
    std::vector<double> outputs(numOutputs);
    for (int i = 0; i < numOutputs; ++i) {
        double sum = bias_output[i];
        for (int j = 0; j < numHidden; ++j) {
            sum += hidden[j] * weights_hidden_output[i][j];
            if (std::isnan(sum) || std::isinf(sum)) {
                std::cerr << "¡Alerta! Sum NaN/Inf en capa de salida de predict()." << std::endl;
                return std::vector<double>(numOutputs, 0.0);
            }
        }
        outputs[i] = sum;
    }

    return outputs;
}

// Entrenamiento (retropropagación)
void NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& targets) {

    std::vector<double> hidden_sums(numHidden, 0.0);
    std::vector<double> hidden_outputs(numHidden, 0.0);
    
    // Verificar entradas NaN
    for (const auto& val : inputs) {
        if (std::isnan(val) || std::isinf(val)) {
            std::cerr << "¡Alerta! Entrada NaN o Inf detectada." << std::endl;
            return; // Saltarse esta actualización
        }
    }
    
    for (const auto& val : targets) {
        if (std::isnan(val) || std::isinf(val)) {
            std::cerr << "¡Alerta! Target NaN o Inf detectado." << std::endl;
            return; // Saltarse esta actualización
        }
    }
    
    // Continuar con forward pass
    for (int i = 0; i < numHidden; ++i) {
        hidden_sums[i] = bias_hidden[i];
        for (int j = 0; j < numInputs; ++j) {
            hidden_sums[i] += inputs[j] * weights_input_hidden[i][j];
            
            // Verificar si el valor se vuelve NaN o Inf
            if (std::isnan(hidden_sums[i]) || std::isinf(hidden_sums[i])) {
                std::cerr << "¡Alerta! NaN/Inf en hidden_sums. Input: " << inputs[j] 
                          << ", Weight: " << weights_input_hidden[i][j] << std::endl;
                return; // Saltarse esta actualización
            }
        }
        hidden_outputs[i] = relu(hidden_sums[i]);
    }

    std::vector<double> final_sums(numOutputs, 0.0);
    std::vector<double> final_outputs(numOutputs, 0.0);
    for (int i = 0; i < numOutputs; ++i) {
        final_sums[i] = bias_output[i];
        for (int j = 0; j < numHidden; ++j) {
            final_sums[i] += hidden_outputs[j] * weights_hidden_output[i][j];
            
            // Verificar si el valor se vuelve NaN o Inf
            if (std::isnan(final_sums[i]) || std::isinf(final_sums[i])) {
                std::cerr << "¡Alerta! NaN/Inf en final_sums." << std::endl;
                return; // Saltarse esta actualización
            }
        }
        final_outputs[i] = final_sums[i]; // Salida lineal
    }

    // 1. Calcular error de la capa de salida (delta de salida)
    std::vector<double> output_deltas(numOutputs);
    for (int i = 0; i < numOutputs; ++i) {
        output_deltas[i] = targets[i] - final_outputs[i]; // Error MSE

        // Protección contra gradientes explosivos
        output_deltas[i] = std::max(-1.0, std::min(1.0, output_deltas[i])); // Clipeo más estricto

        // Protección contra NaN
        if (std::isnan(output_deltas[i]) || std::isinf(output_deltas[i])) {
            std::cerr << "¡Alerta! Se detectó NaN/Inf en los deltas de salida." << std::endl;
            return; // Saltarse esta actualización en lugar de terminar
        }
    }

    // 2. Calcular error de la capa oculta (delta oculto)
    std::vector<double> hidden_deltas(numHidden);
    for (int i = 0; i < numHidden; ++i) {
        double error = 0.0;
        for (int j = 0; j < numOutputs; ++j) {
            error += output_deltas[j] * weights_hidden_output[j][i];
        }
        hidden_deltas[i] = error * relu_derivative(hidden_sums[i]);
        
        hidden_deltas[i] = std::max(-1.0, std::min(1.0, hidden_deltas[i]));
        
        // Protección contra NaN/Inf
        if (std::isnan(hidden_deltas[i]) || std::isinf(hidden_deltas[i])) {
            std::cerr << "¡Alerta! Se detectó NaN/Inf en los deltas ocultos." << std::endl;
            return; // Saltarse esta actualización
        }
    }

    // 3. Actualizar pesos y sesgos de la capa de salida
    for (int i = 0; i < numOutputs; ++i) {
        double effective_lr = learningRate * 0.1; 
        bias_output[i] += effective_lr * output_deltas[i];
        
        for (int j = 0; j < numHidden; ++j) {
            double delta_weight = effective_lr * output_deltas[i] * hidden_outputs[j];
            
            // Clipeo adicional en la actualización de pesos
            delta_weight = std::max(-0.1, std::min(0.1, delta_weight));
            
            weights_hidden_output[i][j] += delta_weight;
            
            // Verificar después de la actualización
            if (std::isnan(weights_hidden_output[i][j]) || std::isinf(weights_hidden_output[i][j])) {
                std::cerr << "¡Alerta! NaN/Inf después de actualizar weights_hidden_output." << std::endl;
                weights_hidden_output[i][j] = 0.0; // Restablecer peso
                std::cerr << "¡Alerta! Se detectó NaN en los pesos. Reiniciando 1." << std::endl;
            }
        }
    }

    // 4. Actualizar pesos y sesgos de la capa de entrada
    for (int i = 0; i < numHidden; ++i) {
        double effective_lr = learningRate * 0.1;
        bias_hidden[i] += effective_lr * hidden_deltas[i];
        
        for (int j = 0; j < numInputs; ++j) {
            double delta_weight = effective_lr * hidden_deltas[i] * inputs[j];
            
            // Clipeo adicional
            delta_weight = std::max(-0.1, std::min(0.1, delta_weight));
            
            weights_input_hidden[i][j] += delta_weight;
            
            // Verificar después de la actualización
            if (std::isnan(weights_input_hidden[i][j]) || std::isinf(weights_input_hidden[i][j])) {
                std::cerr << "¡Alerta! NaN/Inf después de actualizar weights_input_hidden." << std::endl;
                weights_input_hidden[i][j] = 0.0; // Restablecer peso
                std::cerr << "¡Alerta! Se detectó NaN en los pesos. Reiniciando 2." << std::endl;
            }
        }
    }
}

// Guardar pesos en un archivo
void NeuralNetwork::saveWeights(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo para guardar pesos: " << filename << std::endl;
        return;
    }

    auto write_vector = [&](const std::vector<double>& vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            // Verificar que el valor no sea NaN o Inf antes de guardarlo
            if (std::isnan(vec[i]) || std::isinf(vec[i])) {
                file << "0.0" << (i == vec.size() - 1 ? "" : " ");
            } else {
                file << vec[i] << (i == vec.size() - 1 ? "" : " ");
            }
        }
        file << "\n";
    };

    auto write_matrix = [&](const std::vector<std::vector<double>>& matrix) {
        for (const auto& row : matrix) {
            write_vector(row);
        }
    };

    write_matrix(weights_input_hidden);
    write_vector(bias_hidden);
    write_matrix(weights_hidden_output);
    write_vector(bias_output);
}

// Cargar pesos desde un archivo
void NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Info: No se encontró el archivo de pesos: " << filename << ". Se usarán pesos aleatorios." << std::endl;
        return;
    }

    auto read_vector = [&](std::vector<double>& vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            if (!(file >> vec[i])) throw std::runtime_error("Error al leer vector del archivo de pesos.");
            
            // Protección contra valores corruptos
            if (std::isnan(vec[i]) || std::isinf(vec[i]) || vec[i] > 100 || vec[i] < -100) {
                vec[i] = 0.0;
            }
        }
    };

    auto read_matrix = [&](std::vector<std::vector<double>>& matrix) {
        for (auto& row : matrix) {
            read_vector(row);
        }
    };

    try {
        read_matrix(weights_input_hidden);
        read_vector(bias_hidden);
        read_matrix(weights_hidden_output);
        read_vector(bias_output);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << " El archivo " << filename << " podría estar corrupto o no coincidir con la arquitectura de la red." << std::endl;
    }
}

// --- IMPLEMENTACIÓN DE NUEVOS MÉTODOS PARA EL AG ---

// Convierte todos los pesos y sesgos en un único vector (cromosoma)
std::vector<double> NeuralNetwork::getWeightsAsVector() const {
    std::vector<double> vec;
    vec.reserve(weights_input_hidden.size() * weights_input_hidden[0].size() +
                bias_hidden.size() +
                weights_hidden_output.size() * weights_hidden_output[0].size() +
                bias_output.size());

    for (const auto& row : weights_input_hidden) vec.insert(vec.end(), row.begin(), row.end());
    vec.insert(vec.end(), bias_hidden.begin(), bias_hidden.end());
    for (const auto& row : weights_hidden_output) vec.insert(vec.end(), row.begin(), row.end());
    vec.insert(vec.end(), bias_output.begin(), bias_output.end());
    
    return vec;
}

// Establece los pesos y sesgos de la red a partir de un único vector
void NeuralNetwork::setWeightsFromVector(const std::vector<double>& weights) {
    size_t index = 0;
    auto read = [&](double& val) {
        if (index < weights.size()) val = weights[index++];
    };

    for (auto& row : weights_input_hidden) for (auto& val : row) read(val);
    for (auto& val : bias_hidden) read(val);
    for (auto& row : weights_hidden_output) for (auto& val : row) read(val);
    for (auto& val : bias_output) read(val);
}

