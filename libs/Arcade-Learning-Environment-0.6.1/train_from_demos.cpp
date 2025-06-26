#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "src/ale_interface.hpp"
#include "../src/neural_network.hpp"

struct GameplayFrame {
    std::vector<uint8_t> ram_state;
    Action action;
};

// Convierte la RAM en características para la red (igual que en minimal_agent.cpp)
std::vector<double> extractFeatures(const std::vector<uint8_t>& ram_data) {
    std::vector<double> features;
    
    double player_x = static_cast<double>(ram_data[16]);
    features.push_back(player_x / 160.0f);

    // CARACTERÍSTICAS DEL ENEMIGO MÁS CERCANO
    double min_dist = 999.0;
    double closest_enemy_x = 0.0;
    bool enemy_present = false;
    for (int i = 32; i <= 39; i++) {
        if (ram_data[i] > 0) {
            enemy_present = true;
            double enemy_x = static_cast<double>(ram_data[i]);
            double dist = std::abs(player_x - enemy_x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_enemy_x = enemy_x;
            }
        }
    }
    if (enemy_present) {
        features.push_back(min_dist / 160.0);
        features.push_back((closest_enemy_x - player_x) / 160.0);
    } else {
        features.push_back(1.0);
        features.push_back(0.0);
    }
    
    // DETECCIÓN DE BALAS
    double imminent_threat = 0.0;
    double threat_relative_pos = 0.0;
    for (int i = 0x25; i <= 0x2D; ++i) {
        int bullet_value = ram_data[i];
        if (bullet_value > 0) {
            double threat_level = 1.0 - ((i - 0x25) / 8.0);
            if (threat_level > imminent_threat) {
                imminent_threat = threat_level;
                threat_relative_pos = (closest_enemy_x - player_x) / 160.0;
            }
        }
    }
    features.push_back(imminent_threat);
    features.push_back(threat_relative_pos);
    
    features.push_back((ram_data[28] == 0x01) ? 1.0f : 0.0f);
    features.push_back(static_cast<double>(ram_data[114]) / 5.0f);
    
    while (features.size() < 16) {
        features.push_back(0.0);
    }
    return features;
}

// Convierte acción ALE a índice
int getIndexFromAction(Action action) {
    switch (action) {
        case PLAYER_A_LEFT: return 0;
        case PLAYER_A_RIGHT: return 1;
        case PLAYER_A_FIRE: return 2;
        case PLAYER_A_LEFTFIRE: return 3;
        case PLAYER_A_RIGHTFIRE: return 4;
        default: return 2; // FIRE como default
    }
}

int main() {
    // Cargar datos de partidas grabadas
    std::ifstream infile("demon_gameplay_data.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Error: No se pudo abrir el archivo de datos de juego." << std::endl;
        return 1;
    }
    
    size_t num_frames;
    infile.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
    
    std::vector<GameplayFrame> gameplay_data;
    for (size_t i = 0; i < num_frames; ++i) {
        GameplayFrame frame;
        size_t ram_size;
        infile.read(reinterpret_cast<char*>(&ram_size), sizeof(ram_size));
        
        frame.ram_state.resize(ram_size);
        infile.read(reinterpret_cast<char*>(frame.ram_state.data()), ram_size);
        infile.read(reinterpret_cast<char*>(&frame.action), sizeof(frame.action));
        
        gameplay_data.push_back(frame);
    }
    infile.close();
    
    std::cout << "Datos cargados: " << gameplay_data.size() << " frames." << std::endl;
    
    // Inicializar red neuronal
    const int NUM_FEATURES = 16;
    const int NUM_HIDDEN_NEURONS = 128;
    const int NUM_ACTIONS = 5;
    const double LEARNING_RATE = 0.0001;
    NeuralNetwork model(NUM_FEATURES, NUM_HIDDEN_NEURONS, NUM_ACTIONS, LEARNING_RATE);
    
    // Hiperparámetros de entrenamiento
    const int NUM_EPOCHS = 100;
    const int BATCH_SIZE = 64;
    
    // Generador de números aleatorios para mezclar los datos
    std::mt19937 gen(std::random_device{}());
    
    // Entrenamiento supervisado
    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        // Mezclar los datos para cada época
        std::shuffle(gameplay_data.begin(), gameplay_data.end(), gen);
        
        double total_loss = 0.0;
        int correct_predictions = 0;
        
        // Entrenamiento por lotes
        for (size_t i = 0; i < gameplay_data.size(); i += BATCH_SIZE) {
            size_t batch_end = std::min(i + BATCH_SIZE, gameplay_data.size());
            
            for (size_t j = i; j < batch_end; ++j) {
                auto features = extractFeatures(gameplay_data[j].ram_state);
                int action_idx = getIndexFromAction(gameplay_data[j].action);
                
                // Predecir
                auto predictions = model.predict(features);
                if (std::distance(predictions.begin(), 
                                  std::max_element(predictions.begin(), predictions.end())) == action_idx) {
                    correct_predictions++;
                }
                
                // Crear vector objetivo (one-hot encoding)
                std::vector<double> targets(NUM_ACTIONS, 0.0);
                targets[action_idx] = 1.0;
                
                // Entrenar
                model.train(features, targets);
                
                // Calcular pérdida manualmente si es necesario
                auto predictions_after_training = model.predict(features); // CAMBIADO: nombre diferente
                double loss = 0.0;
                for (size_t k = 0; k < targets.size(); ++k) {
                    loss += std::pow(targets[k] - predictions_after_training[k], 2);
                }
                total_loss += loss;
            }
        }
        
        // Calcular métricas
        double avg_loss = total_loss / gameplay_data.size();
        double accuracy = static_cast<double>(correct_predictions) / gameplay_data.size();
        
        std::cout << "Época " << epoch << "/" << NUM_EPOCHS 
                  << ", Pérdida: " << avg_loss 
                  << ", Precisión: " << accuracy * 100.0 << "%" << std::endl;
        
        // Guardar pesos periódicamente
        if (epoch % 10 == 0 || epoch == NUM_EPOCHS) {
            model.saveWeights("demon_bot_imitation_weights.txt");
            std::cout << "Pesos guardados en demon_bot_imitation_weights.txt" << std::endl;
        }
    }
    
    return 0;
}