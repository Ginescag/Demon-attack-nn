#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>

#include "src/ale_interface.hpp"
#include "neural_network.hpp"

// --- HIPERPARÁMETROS DEL ALGORITMO GENÉTICO ---
const int POPULATION_SIZE = 50;      // Número de individuos por generación
const int NUM_GENERATIONS = 1000;    // Número de generaciones a entrenar
const int TOURNAMENT_SIZE = 5;       // Tamaño del torneo para selección
const double MUTATION_RATE = 0.2;   // Probabilidad de que un gen (peso) mute
const double MUTATION_STRENGTH = 0.3; // Magnitud de la mutación

// --- HIPERPARÁMETROS DE LA RED NEURONAL ---
const int NUM_FEATURES = 16;
const int NUM_TOTAL_FEATURES = 128; // Total de características extraídas de la RAM
const int NUM_HIDDEN_NEURONS = 32; // Red más pequeña para un entrenamiento más rápido
const int NUM_ACTIONS = 2;
const double LEARNING_RATE = 0.0; // No se usa en el AG

// Estructura para un individuo de la población
struct Individual {
    NeuralNetwork net;
    double fitness = 0.0;

    Individual() : net(NUM_FEATURES, NUM_HIDDEN_NEURONS, NUM_ACTIONS, LEARNING_RATE) {}
};

std::vector<double> extractAllFeatures(ALEInterface& alei) {
    const auto& ram = alei.getRAM();
    std::vector<double> features;
    for (int i = 0; i < 128; ++i) {
        features.push_back(static_cast<double>(ram.get(i)) / 255.0); // Normalizar entre 0 y 1
    }
    return features;
}

// --- FUNCIONES AUXILIARES (Copiadas de minimal_agent.cpp) ---
std::vector<double> extractFeatures(ALEInterface& alei) {
    const auto& ram = alei.getRAM();
    std::vector<double> features;

    // Posición del jugador (normalizada)
    double player_x = static_cast<double>(ram.get(16));
    features.push_back(player_x / 160.0f);

    // Información sobre enemigos más cercanos
    double min_dist = 999.0;
    double closest_enemy_x = 0.0;
    bool enemy_present = false;
    for (int i = 32; i <= 39; i++) {
        if (ram.get(i) > 0) {
            enemy_present = true;
            double enemy_x = static_cast<double>(ram.get(i));
            double dist = std::abs(player_x - enemy_x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_enemy_x = enemy_x;
            }
        }
    }
    if (enemy_present) {
        features.push_back(min_dist / 160.0); // Distancia mínima normalizada
        features.push_back((closest_enemy_x - player_x) / 160.0); // Dirección hacia el enemigo
    } else {
        features.push_back(1.0); // Sin enemigos presentes
        features.push_back(0.0);
    }

    // Información adicional (por ejemplo, si el jugador está disparando)
    features.push_back((ram.get(28) == 0x01) ? 1.0f : 0.0f);

    // Normalizar el número de vidas restantes
    features.push_back(static_cast<double>(alei.lives()) / 5.0f);

    // Rellenar con ceros si faltan características
    while (features.size() < NUM_FEATURES) {
        features.push_back(0.0);
    }

    return features;
}

Action getActionFromIndex(int index) {
    // Solo permitir las acciones de moverse y disparar
    const static Action actions[] = {PLAYER_A_RIGHTFIRE, PLAYER_A_LEFTFIRE};
    return actions[index % 2]; // Asegurarse de que el índice esté dentro del rango
}

void usage(char const* pname) {
   std::printf("Uso: %s <ruta_a_la_rom> [train|eval]\n", pname);
}

// --- FUNCIONES DEL ALGORITMO GENÉTICO ---

// Evalúa la aptitud de un individuo haciéndolo jugar
void evaluateFitness(Individual& individual, ALEInterface& alei) {
    alei.reset_game();
    reward_t total_score = 0;
    int max_steps = 18000; // Límite de frames por episodio
    for(int step = 0; step < max_steps && !alei.game_over(); ++step) {
        auto state = extractFeatures(alei);
        auto q_values = individual.net.predict(state);
        int action_idx = std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
        total_score += alei.act(getActionFromIndex(action_idx));
    }
    individual.fitness = total_score;
}

// Selección por torneo
Individual tournamentSelection(const std::vector<Individual>& population, int tournament_size) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, population.size() - 1);

    Individual best = population[dist(gen)];
    for (int i = 1; i < tournament_size; ++i) {
        Individual competitor = population[dist(gen)];
        if (competitor.fitness > best.fitness) {
            best = competitor;
        }
    }
    return best;
}

// Muta los genes (pesos) de un individuo
void mutate(Individual& individual) {
    auto weights = individual.net.getWeightsAsVector();
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> rate_dist(0.0, 1.0);
    std::normal_distribution<> mutation_dist(0.0, MUTATION_STRENGTH);

    for (auto& weight : weights) {
        if (rate_dist(gen) < MUTATION_RATE) {
            weight += mutation_dist(gen);
        }
    }
    individual.net.setWeightsFromVector(weights);
}

// --- PROGRAMA PRINCIPAL ---
int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    bool training_mode = (argc >= 3 && std::string(argv[2]) == "train");
    bool evaluation_mode = (argc < 3 || std::string(argv[2]) == "eval");

    ALEInterface alei{};
    alei.setBool("sound", false);
    alei.setInt("frame_skip", 4);
    alei.setBool("display_screen", !training_mode);
    alei.loadROM(argv[1]);

    if (training_mode) {
        std::cout << "--- Iniciando Entrenamiento con Algoritmo Genético ---" << std::endl;
        
        std::vector<Individual> population(POPULATION_SIZE);

        for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
            // 1. Evaluar la población
            double total_fitness = 0;
            for (auto& individual : population) {
                evaluateFitness(individual, alei);
                total_fitness += individual.fitness;
            }

            // 2. Ordenar por aptitud (de mayor a menor)
            std::sort(population.begin(), population.end(), [](const auto& a, const auto& b) {
                return a.fitness > b.fitness;
            });

            // Imprimir estadísticas de la generación
            std::cout << "Generación: " << std::setw(4) << gen
                      << " | Mejor Fitness: " << std::setw(6) << population[0].fitness
                      << " | Fitness Promedio: " << std::setw(8) << std::fixed << std::setprecision(2) << (total_fitness / POPULATION_SIZE)
                      << std::endl;

            // 3. Crear la nueva generación
            std::vector<Individual> new_population;
            new_population.reserve(POPULATION_SIZE);

            // 3.1. Elitismo: El mejor individuo pasa directamente
            new_population.push_back(population[0]);

            // 3.2. Selección por torneo y mutación para el resto
            while (new_population.size() < POPULATION_SIZE) {
                Individual parent = tournamentSelection(population, TOURNAMENT_SIZE);
                Individual child = parent; // Copiar al padre
                mutate(child); // Aplicar mutación
                new_population.push_back(child);
            }

            population = new_population;

            // Guardar el mejor modelo de la generación
            population[0].net.saveWeights("demon_bot_genetic_weights.txt");
        }

    } else if (evaluation_mode) {
        std::cout << "--- Iniciando Modo de Evaluación ---" << std::endl;
        Individual best_agent;
        best_agent.net.loadWeights("demon_bot_genetic_weights.txt");

        for (int episode = 1; episode <= 10; ++episode) {
            evaluateFitness(best_agent, alei);
            std::cout << "Evaluación - Episodio " << episode << ", Puntuacion: " << best_agent.fitness << std::endl;
        }
    }

    std::cout << "Proceso finalizado." << std::endl;
    return 0;
}