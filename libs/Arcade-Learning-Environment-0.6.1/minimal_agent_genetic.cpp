#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>

#include "src/ale_interface.hpp"
#include "neural_network.hpp"

// --- HIPERPARÁMETROS DEL ALGORITMO GENÉTICO ---
const int POPULATION_SIZE = 60;      // Mayor población para más diversidad
const int NUM_GENERATIONS = 2000;    // Más generaciones
const int TOURNAMENT_SIZE = 3;       // Torneo más pequeño para menos presión selectiva
const double MUTATION_RATE = 0.15;   // Tasa de mutación más conservadora
const double MUTATION_STRENGTH = 0.2; // Mutación más suave
const double ELITE_RATIO = 0.1;      // 10% de élite
const double CROSSOVER_RATE = 0.7;   // Probabilidad de crossover

// --- HIPERPARÁMETROS DE LA RED NEURONAL ---
const int NUM_FEATURES = 25;         // Más características para mejor representación
const int NUM_TOTAL_FEATURES = 128;
const int NUM_HIDDEN_NEURONS = 64;   // Red más grande para comportamientos complejos
const int NUM_ACTIONS = 6;           // Más acciones disponibles
const double LEARNING_RATE = 0.0;

// Estructura para un individuo de la población
struct Individual {
    NeuralNetwork net;
    double fitness = 0.0;
    double survival_time = 0.0;
    double damage_dealt = 0.0;
    double movement_score = 0.0;
    int generation_born = 0;

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

// --- FUNCIONES AUXILIARES MEJORADAS ---
std::vector<double> extractFeatures(ALEInterface& alei) {
    const auto& ram = alei.getRAM();
    std::vector<double> features;

    // 1. Posición del jugador (normalizada)
    double player_x = static_cast<double>(ram.get(16));
    features.push_back(player_x / 160.0);

    // 2. Información sobre enemigos (hasta 8 enemigos)
    std::vector<std::pair<double, double>> enemies;
    for (int i = 32; i <= 39; i++) {
        if (ram.get(i) > 0) {
            double enemy_x = static_cast<double>(ram.get(i));
            double enemy_y = static_cast<double>(ram.get(i + 8)); // Y positions at 40-47
            enemies.push_back({enemy_x, enemy_y});
        }
    }
    
    // Distancia y dirección al enemigo más cercano
    if (!enemies.empty()) {
        double min_dist = 999.0;
        double closest_enemy_x = 0.0;
        double closest_enemy_y = 0.0;
        for (const auto& enemy : enemies) {
            double dist = std::sqrt(std::pow(player_x - enemy.first, 2) + std::pow(150 - enemy.second, 2));
            if (dist < min_dist) {
                min_dist = dist;
                closest_enemy_x = enemy.first;
                closest_enemy_y = enemy.second;
            }
        }
        features.push_back(min_dist / 200.0); // Distancia normalizada
        features.push_back((closest_enemy_x - player_x) / 160.0); // Dirección X
        features.push_back(closest_enemy_y / 200.0); // Altura del enemigo
    } else {
        features.push_back(1.0); // Sin enemigos
        features.push_back(0.0);
        features.push_back(0.0);
    }

    // 3. Información detallada sobre proyectiles enemigos
    std::vector<std::pair<double, double>> enemy_bullets;
    for (int i = 0; i < 8; ++i) {
        int bullet_y_addr = 0x50 + i;
        int bullet_x_addr = 0x58 + i;
        int bullet_y = ram.get(bullet_y_addr);
        int bullet_x = ram.get(bullet_x_addr);
        
        if (bullet_y > 0 && bullet_y < 210) {
            enemy_bullets.push_back({static_cast<double>(bullet_x), static_cast<double>(bullet_y)});
        }
    }

    // Proyectil más peligroso (más cerca del jugador)
    if (!enemy_bullets.empty()) {
        double min_threat_dist = 999.0;
        double threat_x = 0.0;
        double threat_y = 0.0;
        for (const auto& bullet : enemy_bullets) {
            double x_dist = std::abs(player_x - bullet.first);
            double threat_level = (210 - bullet.second) / 210.0; // Más peligroso cuanto más bajo
            double weighted_dist = x_dist * (1.0 - threat_level);
            
            if (weighted_dist < min_threat_dist) {
                min_threat_dist = weighted_dist;
                threat_x = bullet.first;
                threat_y = bullet.second;
            }
        }
        
        features.push_back(1.0); // Hay amenaza
        features.push_back((threat_x - player_x) / 160.0); // Dirección relativa
        features.push_back(threat_y / 210.0); // Altura de la amenaza
        features.push_back(min_threat_dist / 200.0); // Distancia ponderada
    } else {
        features.push_back(0.0); // Sin amenazas
        features.push_back(0.0);
        features.push_back(0.0);
        features.push_back(0.0);
    }

    // 4. Información sobre proyectiles propios
    int player_bullets_count = 0;
    for (int i = 0; i < 4; ++i) {
        if (ram.get(0x60 + i) > 0) player_bullets_count++;
    }
    features.push_back(static_cast<double>(player_bullets_count) / 4.0);

    // 5. Estado del jugador
    features.push_back((ram.get(28) == 0x01) ? 1.0 : 0.0); // Puede disparar
    features.push_back(static_cast<double>(alei.lives()) / 5.0); // Vidas restantes
    
    // 6. Información espacial adicional
    double left_space = player_x / 160.0; // Espacio a la izquierda
    double right_space = (160.0 - player_x) / 160.0; // Espacio a la derecha
    features.push_back(left_space);
    features.push_back(right_space);
    
    // 7. Densidad de enemigos por zona
    int left_enemies = 0, center_enemies = 0, right_enemies = 0;
    for (const auto& enemy : enemies) {
        if (enemy.first < 53) left_enemies++;
        else if (enemy.first < 107) center_enemies++;
        else right_enemies++;
    }
    features.push_back(static_cast<double>(left_enemies) / 8.0);
    features.push_back(static_cast<double>(center_enemies) / 8.0);
    features.push_back(static_cast<double>(right_enemies) / 8.0);

    // 8. Información temporal/contextual
    features.push_back(static_cast<double>(alei.getEpisodeFrameNumber()) / 18000.0); // Progreso temporal

    // Rellenar hasta NUM_FEATURES si es necesario
    while (features.size() < NUM_FEATURES) {
        features.push_back(0.0);
    }

    return features;
}

Action getActionFromIndex(int index) {
    // Conjunto más amplio de acciones para comportamientos más complejos
    const static Action actions[] = {
        PLAYER_A_LEFT,       // 0: Moverse izquierda
        PLAYER_A_RIGHT,      // 1: Moverse derecha
        PLAYER_A_FIRE,       // 2: Disparar
        PLAYER_A_LEFTFIRE,   // 3: Moverse izquierda y disparar
        PLAYER_A_RIGHTFIRE,  // 4: Moverse derecha y disparar
        PLAYER_A_NOOP        // 5: No hacer nada
    };
    return actions[index % 6];
}

void usage(char const* pname) {
   std::printf("Uso: %s <ruta_a_la_rom> [train|eval]\n", pname);
}

// --- FUNCIONES DEL ALGORITMO GENÉTICO MEJORADAS ---

// Evalúa la aptitud de un individuo con métricas múltiples
void evaluateFitness(Individual& individual, ALEInterface& alei) {
    alei.reset_game();
    reward_t total_score = 0;
    int initial_lives = alei.lives();
    int survival_frames = 0;
    int damage_dealt = 0;
    int movement_diversity = 0;
    int last_action = -1;
    int same_action_count = 0;
    double total_threat_avoidance = 0.0;
    int threat_encounters = 0;
    
    std::vector<double> position_history;
    const int max_steps = 18000; // Límite de frames por episodio
    
    for(int step = 0; step < max_steps && !alei.game_over(); ++step) {
        auto state = extractFeatures(alei);
        auto q_values = individual.net.predict(state);
        int action_idx = std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
        
        // Registrar diversidad de movimiento
        if (action_idx != last_action) {
            if (same_action_count > 10) { // Penalizar quedarse inmóvil demasiado tiempo
                movement_diversity -= same_action_count / 10;
            }
            movement_diversity++;
            same_action_count = 0;
        } else {
            same_action_count++;
        }
        last_action = action_idx;
        
        // Evaluar evasión de amenazas
        bool has_threat = state[7] > 0.5; // Feature que indica amenaza
        if (has_threat) {
            threat_encounters++;
            double threat_direction = state[8]; // Dirección relativa de la amenaza
            
            // Premiar movimiento en dirección opuesta a la amenaza
            if ((threat_direction < -0.1 && (action_idx == 1 || action_idx == 4)) || // amenaza a la izq, moverse derecha
                (threat_direction > 0.1 && (action_idx == 0 || action_idx == 3))) {  // amenaza a la der, moverse izquierda
                total_threat_avoidance += 1.0;
            }
        }
        
        // Registrar posición para calcular cobertura del mapa
        const auto& ram = alei.getRAM();
        double player_x = static_cast<double>(ram.get(16));
        position_history.push_back(player_x);
        
        reward_t frame_reward = alei.act(getActionFromIndex(action_idx));
        total_score += frame_reward;
        
        // Contar daño infligido (incremento en puntuación que no sea por tiempo)
        if (frame_reward > 10) damage_dealt += frame_reward;
        
        survival_frames++;
    }
    
    // Calcular cobertura del espacio
    double space_coverage = 0.0;
    if (!position_history.empty()) {
        std::sort(position_history.begin(), position_history.end());
        double min_pos = position_history.front();
        double max_pos = position_history.back();
        space_coverage = (max_pos - min_pos) / 160.0; // Normalizado
    }
    
    // Función de fitness multiobjetivo
    double base_score = static_cast<double>(total_score);
    double survival_bonus = static_cast<double>(survival_frames) / max_steps * 1000.0;
    double lives_penalty = (initial_lives - alei.lives()) * 500.0;
    double movement_bonus = std::min(static_cast<double>(movement_diversity) / 100.0, 1.0) * 300.0;
    double damage_bonus = static_cast<double>(damage_dealt) * 2.0;
    double coverage_bonus = space_coverage * 200.0;
    double threat_avoidance_bonus = threat_encounters > 0 ? 
        (total_threat_avoidance / threat_encounters) * 400.0 : 0.0;
    
    // Penalizar comportamiento estático severo
    double static_penalty = 0.0;
    if (movement_diversity < 10 && survival_frames > 1000) {
        static_penalty = 1000.0; // Penalización severa por quedarse inmóvil
    }
    
    individual.fitness = base_score + survival_bonus + movement_bonus + 
                        damage_bonus + coverage_bonus + threat_avoidance_bonus - 
                        lives_penalty - static_penalty;
    
    // Guardar métricas adicionales para análisis
    individual.survival_time = static_cast<double>(survival_frames);
    individual.damage_dealt = static_cast<double>(damage_dealt);
    individual.movement_score = static_cast<double>(movement_diversity);
}

// Selección por torneo con diversidad
Individual tournamentSelection(const std::vector<Individual>& population, int tournament_size) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, population.size() - 1);

    Individual best = population[dist(gen)];
    for (int i = 1; i < tournament_size; ++i) {
        Individual competitor = population[dist(gen)];
        // Considerar tanto fitness como diversidad comportamental
        double fitness_diff = competitor.fitness - best.fitness;
        double diversity_bonus = std::abs(competitor.movement_score - best.movement_score) * 10.0;
        
        if (fitness_diff + diversity_bonus > 0) {
            best = competitor;
        }
    }
    return best;
}

// Crossover uniforme entre dos padres
Individual crossover(const Individual& parent1, const Individual& parent2) {
    Individual child;
    auto weights1 = parent1.net.getWeightsAsVector();
    auto weights2 = parent2.net.getWeightsAsVector();
    auto child_weights = weights1; // Copia del primer padre
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    // Crossover uniforme: cada gen tiene 50% probabilidad de venir de cada padre
    for (size_t i = 0; i < child_weights.size(); ++i) {
        if (dist(gen) < 0.5) {
            child_weights[i] = weights2[i];
        }
    }
    
    child.net.setWeightsFromVector(child_weights);
    return child;
}

// Mutación adaptativa
void mutate(Individual& individual, int generation) {
    auto weights = individual.net.getWeightsAsVector();
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> rate_dist(0.0, 1.0);
    
    // Mutación adaptativa: reduce con las generaciones
    double adaptive_rate = MUTATION_RATE * (1.0 - static_cast<double>(generation) / NUM_GENERATIONS * 0.5);
    double adaptive_strength = MUTATION_STRENGTH * (1.0 - static_cast<double>(generation) / NUM_GENERATIONS * 0.3);
    
    std::normal_distribution<> mutation_dist(0.0, adaptive_strength);

    for (auto& weight : weights) {
        if (rate_dist(gen) < adaptive_rate) {
            double old_weight = weight;
            weight += mutation_dist(gen);
            
            // Límites para evitar explosión de gradientes
            weight = std::max(-5.0, std::min(5.0, weight));
            
            // Mutación creep adicional para exploración fina
            if (rate_dist(gen) < 0.1) {
                weight += (rate_dist(gen) - 0.5) * 0.01;
            }
        }
    }
    individual.net.setWeightsFromVector(weights);
    individual.generation_born = generation;
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
        std::cout << "--- Iniciando Entrenamiento con Algoritmo Genético Mejorado ---" << std::endl;
        
        std::vector<Individual> population(POPULATION_SIZE);
        std::vector<double> fitness_history;
        std::mt19937 gen(std::random_device{}());

        for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
            // 1. Evaluar la población
            double total_fitness = 0;
            double max_fitness = -999999;
            double min_fitness = 999999;
            double total_survival = 0;
            double total_damage = 0;
            double total_movement = 0;
            
            for (auto& individual : population) {
                evaluateFitness(individual, alei);
                total_fitness += individual.fitness;
                max_fitness = std::max(max_fitness, individual.fitness);
                min_fitness = std::min(min_fitness, individual.fitness);
                total_survival += individual.survival_time;
                total_damage += individual.damage_dealt;
                total_movement += individual.movement_score;
            }

            // 2. Ordenar por aptitud (de mayor a menor)
            std::sort(population.begin(), population.end(), [](const auto& a, const auto& b) {
                return a.fitness > b.fitness;
            });

            // Imprimir estadísticas detalladas
            double avg_fitness = total_fitness / POPULATION_SIZE;
            fitness_history.push_back(avg_fitness);
            
            std::cout << "Gen: " << std::setw(4) << generation
                      << " | Best: " << std::setw(8) << std::fixed << std::setprecision(1) << max_fitness
                      << " | Avg: " << std::setw(8) << avg_fitness
                      << " | Survival: " << std::setw(6) << (total_survival / POPULATION_SIZE)
                      << " | Movement: " << std::setw(5) << (total_movement / POPULATION_SIZE)
                      << " | Damage: " << std::setw(6) << (total_damage / POPULATION_SIZE)
                      << std::endl;

            // Detectar estancamiento y aplicar medidas
            if (generation > 50 && generation % 25 == 0) {
                double recent_avg = 0;
                for (int i = std::max(0, (int)fitness_history.size() - 25); i < fitness_history.size(); ++i) {
                    recent_avg += fitness_history[i];
                }
                recent_avg /= std::min(25, (int)fitness_history.size());
                
                double old_avg = 0;
                int start_idx = std::max(0, (int)fitness_history.size() - 50);
                int end_idx = std::max(0, (int)fitness_history.size() - 25);
                for (int i = start_idx; i < end_idx; ++i) {
                    old_avg += fitness_history[i];
                }
                if (end_idx > start_idx) old_avg /= (end_idx - start_idx);
                
                double improvement = recent_avg - old_avg;
                std::cout << "    -> Mejora en últimas 25 gen: " << improvement << std::endl;
                
                // Si el progreso es muy lento, introducir diversidad
                if (improvement < 50.0) {
                    std::cout << "    -> Introduciendo diversidad por estancamiento..." << std::endl;
                    for (int i = POPULATION_SIZE / 2; i < POPULATION_SIZE - 5; ++i) {
                        // Reiniciar algunos individuos aleatorios
                        population[i] = Individual();
                        population[i].generation_born = generation;
                    }
                }
            }

            // 3. Crear la nueva generación
            std::vector<Individual> new_population;
            new_population.reserve(POPULATION_SIZE);

            // 3.1. Elitismo: Los mejores individuos pasan directamente
            int elite_size = static_cast<int>(POPULATION_SIZE * ELITE_RATIO);
            for (int i = 0; i < elite_size; ++i) {
                new_population.push_back(population[i]);
            }

            // 3.2. Crossover y mutación para el resto
            std::uniform_real_distribution<> crossover_dist(0.0, 1.0);
            while (new_population.size() < POPULATION_SIZE) {
                if (crossover_dist(gen) < CROSSOVER_RATE && new_population.size() < POPULATION_SIZE - 1) {
                    // Crossover
                    Individual parent1 = tournamentSelection(population, TOURNAMENT_SIZE);
                    Individual parent2 = tournamentSelection(population, TOURNAMENT_SIZE);
                    Individual child = crossover(parent1, parent2);
                    mutate(child, generation);
                    new_population.push_back(child);
                } else {
                    // Solo mutación
                    Individual parent = tournamentSelection(population, TOURNAMENT_SIZE);
                    Individual child = parent;
                    mutate(child, generation);
                    new_population.push_back(child);
                }
            }

            population = new_population;

            // Guardar el mejor modelo periódicamente
            if (generation % 50 == 0 || generation == NUM_GENERATIONS - 1) {
                population[0].net.saveWeights("demon_bot_genetic_weights.txt");
                std::cout << "    -> Modelo guardado en generación " << generation << std::endl;
            }
        }

    } else if (evaluation_mode) {
        std::cout << "--- Iniciando Modo de Evaluación ---" << std::endl;
        Individual best_agent;
        best_agent.net.loadWeights("demon_bot_genetic_weights.txt");

        for (int episode = 1; episode <= 10; ++episode) {
            evaluateFitness(best_agent, alei);
            std::cout << "Eval " << episode << " - Fitness: " << best_agent.fitness 
                      << ", Survival: " << best_agent.survival_time
                      << ", Movement: " << best_agent.movement_score
                      << ", Damage: " << best_agent.damage_dealt << std::endl;
        }
    }

    std::cout << "Proceso finalizado." << std::endl;
    return 0;
}