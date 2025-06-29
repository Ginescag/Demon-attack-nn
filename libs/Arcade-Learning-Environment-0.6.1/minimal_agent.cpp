
#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iterator>
#include <deque>
#include <random>
#include <unordered_set>

#include "src/ale_interface.hpp"
#include <SDL/SDL.h>
#include "neural_network.hpp"

///////////////////////////////////////////////////////////////////////////////
/// UTILITY FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

void clearScreen() {
   std::printf("\033[H\033[J");
}

void printRAM(ALEInterface& alei){
   auto* RAM {alei.getRAM().array() };
   size_t i{};
   std::printf("\033[H");
   std::printf("\nADDR || 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F");
   std::printf("\n-------------------------------------------------------------");
   for(uint16_t row{}; row < 8; ++row) {
      std::printf("\n %02X  ||", row * 16);
      for (uint16_t col{}; col < 16; ++col, ++i) {
         std::printf(" %02X", RAM[i]);
      }
   }
   std::printf("\n-------------------------------------------------------------");
}

///////////////////////////////////////////////////////////////////////////////
/// RL AGENT FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

// Estructura para almacenar una transición (experiencia)
struct Transition {
    std::vector<double> state;
    int action;
    reward_t reward;
    std::vector<double> next_state;
    bool is_done;
    double priority;
};

// Convierte las lecturas de la RAM en un vector de características normalizadas
std::vector<double> extractFeatures(ALEInterface& alei) {
    const auto& ram = alei.getRAM();
    std::vector<double> features;

    double player_x = static_cast<double>(ram.get(16));
    features.push_back(player_x / 160.0f);

    // CARACTERÍSTICAS DEL ENEMIGO MÁS CERCANO
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
        features.push_back(min_dist / 160.0); // Distancia normalizada
        features.push_back((closest_enemy_x - player_x) / 160.0); // Posición relativa X
    } else {
        features.push_back(1.0); // Distancia máxima si no hay enemigos
        features.push_back(0.0); // Posición relativa neutral
    }

    double imminent_threat = 0.0;
    double threat_relative_pos = 0.0;
    
    // Buscar balas activas, priorizando las más cercanas al jugador (valores bajos, cerca de 0x25)
    for (int i = 0x25; i <= 0x2D; ++i) {
        int bullet_value = ram.get(i);
        if (bullet_value > 0) {
            double threat_level = 1.0 - ((i - 0x25) / 8.0); // 1.0 para 0x25, 0.0 para 0x2D
            
            if (threat_level > imminent_threat) {
                imminent_threat = threat_level;
                threat_relative_pos = (closest_enemy_x - player_x) / 160.0;
            }
        }
    }
    
    features.push_back(imminent_threat);
    features.push_back(threat_relative_pos);

    // Mantener características antiguas útiles
    features.push_back((ram.get(28) == 0x01) ? 1.0f : 0.0f); // Puede disparar
    features.push_back(static_cast<double>(ram.get(114)) / 5.0f); // Vidas

    // Rellenar con ceros para mantener el tamaño del vector de entrada
    while (features.size() < 16) {
        features.push_back(0.0);
    }
    return features;
}

Action getActionFromIndex(int index) {
    switch (index) {
        case 0: return PLAYER_A_LEFT;
        case 1: return PLAYER_A_RIGHT;
        case 2: return PLAYER_A_FIRE;
        case 3: return PLAYER_A_LEFTFIRE;
        case 4: return PLAYER_A_RIGHTFIRE;
        default: return PLAYER_A_NOOP;
    }
}

void usage(char const* pname) {
   std::printf("Uso: %s <ruta_a_la_rom> [train|eval|manual]\n", pname);
   std::printf("  train: entrenar sin visualización\n");
   std::printf("  eval: evaluar el modelo entrenado (epsilon=0, sin entrenamiento)\n");
   std::printf("  manual: visualizar RAM en tiempo real (para depuración)\n");
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   if (argc < 2) {
      usage(argv[0]);
      return 1;
   }

   // --- Parámetros del modo de ejecución ---
   bool training_mode = (argc >= 3 && std::string(argv[2]) == "train");
   bool evaluation_mode = (argc >= 3 && std::string(argv[2]) == "eval");
   bool manual_mode = (argc >= 3 && std::string(argv[2]) == "manual");
   
   // --- Hiperparámetros de RL ---
   const int NUM_EPISODES = (evaluation_mode || manual_mode) ? 100 : 35000;
   const size_t REPLAY_MEMORY_SIZE = 10000;
   const size_t BATCH_SIZE = 64;
   const double GAMMA = 0.99;
   double epsilon = (evaluation_mode || manual_mode) ? 0.0 : 1.0;
   const double EPSILON_MIN = 0.1; // Epsilon mínimo más alto para mantener exploración
   const double EPSILON_DECAY = 0.9995;
   const int TRAIN_FREQUENCY = 4;

   // --- Inicialización de la Red Neuronal ---
   const int NUM_FEATURES = 16;
   const int NUM_HIDDEN_NEURONS = 128;
   const int NUM_ACTIONS = 5; 
   const double LEARNING_RATE = 0.0001;
   NeuralNetwork model(NUM_FEATURES, NUM_HIDDEN_NEURONS, NUM_ACTIONS, LEARNING_RATE);
   model.loadWeights("demon_bot_weights.txt");

   // --- Inicialización de ALE ---
   ALEInterface alei{};
   
   if (training_mode) {
      alei.setBool("display_screen", false);
      alei.setInt("frame_skip", 4); // Reducido de 8 a 4 para mejor reactividad
      std::cout << "Iniciando entrenamiento rápido..." << std::endl;
   } else if (evaluation_mode) {
      alei.setBool("display_screen", true);
      std::cout << "Iniciando modo de EVALUACIÓN (epsilon=0, sin entrenamiento)..." << std::endl;
   } else if (manual_mode) {
      alei.setBool("display_screen", true);
      alei.setInt("frame_skip", 1); // Sin saltar frames para mejor visualización
      std::cout << "Iniciando modo MANUAL con visualización de RAM..." << std::endl;
   } else {
      alei.setBool("display_screen", true);
      std::cout << "Iniciando entrenamiento con visualización..." << std::endl;
   }
   
   alei.setBool("sound", false);
   alei.loadROM(argv[1]);

   // --- Modo Manual ---
   if (manual_mode) {
      clearScreen();
      std::cout << "Controles: Flechas para mover, Espacio para disparar, ESC para salir" << std::endl;
      
      alei.reset_game();
      while (true) {
         printRAM(alei);
         
         SDL_Event event;
         while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
               return 0;
            }
         }
         
         Uint8* keystates = SDL_GetKeyState(NULL);
         Action action = PLAYER_A_NOOP;
         
         if (keystates[SDLK_LEFT] && keystates[SDLK_SPACE]) action = PLAYER_A_LEFTFIRE;
         else if (keystates[SDLK_RIGHT] && keystates[SDLK_SPACE]) action = PLAYER_A_RIGHTFIRE;
         else if (keystates[SDLK_LEFT]) action = PLAYER_A_LEFT;
         else if (keystates[SDLK_RIGHT]) action = PLAYER_A_RIGHT;
         else if (keystates[SDLK_SPACE]) action = PLAYER_A_FIRE;
         
         alei.act(action);
         
         if (alei.game_over()) {
            alei.reset_game();
         }
         
         SDL_Delay(1000/60);
      }
   }

   std::deque<Transition> replay_memory;
   std::mt19937 gen(std::random_device{}());

   for (int episode = 1; episode <= NUM_EPISODES; ++episode) {
      alei.reset_game();
      auto state = extractFeatures(alei);
      reward_t total_score = 0;
      int lives = alei.lives();
      int step_counter = 0;
      
      // Variables para rastrear comportamiento cobarde
      int frames_sin_disparar = 0;
      int frames_sin_moverse = 0;
      double last_pos_x = state[0] * 160.0; // Convertir posición normalizada a valor real
      bool done = false; // Declarar 'done' aquí, fuera del bucle

      while (!done) {
         step_counter++;
         int action_idx;
         
         // Exploración más inteligente
         if (!evaluation_mode && static_cast<double>(gen()) / gen.max() < epsilon) {
            // 50% del tiempo de exploración, forzar movimientos (no solo disparar)
            if (static_cast<double>(gen()) / gen.max() < 0.5) {
                action_idx = 2 + gen() % 3; // Forzar LEFT o RIGHT (índices 0 y 1)
            } else {
                action_idx = gen() % NUM_ACTIONS; // Acción completamente aleatoria
            }
         } else {
            auto q_values = model.predict(state);
            action_idx = std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
         }

         Action ale_action = getActionFromIndex(action_idx);
         reward_t game_reward = alei.act(ale_action);

         auto next_state = extractFeatures(alei);
         done = alei.game_over();
         const auto& ram = alei.getRAM();


         double player_x = next_state[0] * 160.0;
         bool ha_disparado = (action_idx == 2 || action_idx == 3 || action_idx == 4);
         bool se_ha_movido = std::abs(player_x - last_pos_x) > 2.0;
         
         if (!ha_disparado) {
             frames_sin_disparar++;
         } else {
             frames_sin_disparar = 0;
         }
         
         if (!se_ha_movido) {
             frames_sin_moverse++;
         } else {
             frames_sin_moverse = 0;
         }
         
         last_pos_x = player_x;

         //INGENIERÍA DE RECOMPENSAS
         double shaped_reward = 0;

         shaped_reward += 5.0;

         // Penalización fuerte por morir
         if (alei.lives() < lives) {
             shaped_reward -= 25; 
             lives = alei.lives();
         }

         // 1. PENALIZAR POR NO DISPARAR
         if (frames_sin_disparar > 60) {
             shaped_reward -= 5.0;
         }

         // 2. PENALIZAR POR QUEDARSE QUIETO
         if (frames_sin_moverse > 60) {
             shaped_reward -= 5.0;
         }

         // 3. PENALIZAR POR QUEDARSE EN LOS EXTREMOS DEL MAPA
         double normalized_x = next_state[0];
         if (normalized_x < 0.15 || normalized_x > 0.85) {
             shaped_reward -= 10.0; // Penalización constante y más fuerte
         }

         // 4. RECOMPENSA MASIVA POR MATAR
         if (ram.get(126) == 0x4E) {
             shaped_reward += 200.0;
         }

         // 5. FINALIZAR EPISODIO SI ES EXTREMADAMENTE COBARDE
         bool extremadamente_cobarde = (frames_sin_disparar > 240) || (frames_sin_moverse > 240);
         if (extremadamente_cobarde && !done) { // Comprobar si no hemos terminado ya
             shaped_reward -= 1000;  // Enorme penalización final
             done = true;
             std::cout << "¡Episodio terminado por comportamiento cobarde!" << std::endl;
         }
         
         reward_t final_reward = game_reward + shaped_reward;
         total_score += final_reward;

         if (!evaluation_mode) {
            double priority = std::abs(final_reward) + 1.0;
            // La transición se guarda con el valor correcto de 'done'
            replay_memory.push_back({state, action_idx, final_reward, next_state, done, priority});
            if (replay_memory.size() > REPLAY_MEMORY_SIZE) {
               replay_memory.pop_front();
            }

            if (step_counter % TRAIN_FREQUENCY == 0 && replay_memory.size() >= BATCH_SIZE) {
               std::vector<Transition> batch;
               std::vector<double> probabilities(replay_memory.size());
               double total_priority = 0.0;
               for(const auto& t : replay_memory) total_priority += t.priority;
               for(size_t i = 0; i < replay_memory.size(); ++i) probabilities[i] = replay_memory[i].priority / total_priority;
               
               std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
               std::unordered_set<size_t> selected_indices;
               while (selected_indices.size() < std::min(BATCH_SIZE, replay_memory.size())) {
                  selected_indices.insert(dist(gen));
               }
               for (auto idx : selected_indices) batch.push_back(replay_memory[idx]);

               for (const auto& transition : batch) {
                  auto q_current = model.predict(transition.state);
                  auto q_next = model.predict(transition.next_state);
                  double q_target = transition.reward;
                  if (!transition.is_done) {
                     q_target += GAMMA * (*std::max_element(q_next.begin(), q_next.end()));
                  }
                  std::vector<double> targets = q_current;
                  targets[transition.action] = q_target;
                  model.train(transition.state, targets);
               }
            }
         }
         
         state = next_state;
      }

      if (!evaluation_mode && epsilon > EPSILON_MIN) {
         epsilon *= EPSILON_DECAY;
      }

      if (evaluation_mode) {
         std::cout << "Evaluación - Episodio " << episode << ", Puntuacion: " << total_score << std::endl;
      } else {
         std::cout << "Episodio: " << episode << ", Puntuacion: " << total_score << ", Epsilon: " << epsilon << std::endl;
      }
      
      if (!evaluation_mode && episode % 50 == 0) {
         std::cout << "--- Guardando pesos del modelo ---" << std::endl;
         model.saveWeights("demon_bot_weights.txt");
      }
   }

   std::cout << "Proceso finalizado." << std::endl;
   return 0;
}

//=======================================================================================
//ESTE CODIGO ES PARA VER LA RAM
//=======================================================================================

// #include <iostream>
// #include <cmath>
// #include <cstdint>
// #include "src/ale_interface.hpp"
// #include <SDL/SDL.h>
// #include <vector>

// // Constants
// constexpr uint32_t maxSteps = 7500;

// ///////////////////////////////////////////////////////////////////////////////
// /// Get info from RAM
// ///////////////////////////////////////////////////////////////////////////////
// int32_t getPlayerX(ALEInterface& alei) {
//    return alei.getRAM().get(72) + ((rand() % 3) - 1);
// }

// int32_t getBallX(ALEInterface& alei) {
//    return alei.getRAM().get(99) + ((rand() % 3) - 1);
// }


// void clearScreen() {
//    // Clear the console screen
//    std::printf("\033[H\033[J");
// }
// void printRAM(ALEInterface& alei){
//    auto* RAM {alei.getRAM().array() };

//    size_t i{};
//    std::printf("\033[H");
//    std::printf("\nADDR || 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F");
//    std::printf("\n-------------------------------------------------------------");

//    for(uint16_t row{}; row < 8; ++row) {
//       std::printf("\n %02X  ||", row * 16);
//       for (uint16_t col{}; col < 16; ++col, ++i) {
//          std::printf(" %02X", RAM[i]);
//       }
//    }
//    std::printf("\n-------------------------------------------------------------");

// }

// ///////////////////////////////////////////////////////////////////////////////
// /// Do Next Manual Step
// ///////////////////////////////////////////////////////////////////////////////
// reward_t manualStep(ALEInterface& alei) {
//    Action action = PLAYER_A_NOOP;
//    Uint8* keystates = SDL_GetKeyState(NULL);

//    if (keystates[SDLK_LEFT]) {
//       action = PLAYER_A_LEFT;
//    }
//    else if (keystates[SDLK_RIGHT]) {
//       action = PLAYER_A_RIGHT;
//    }
//    else if (keystates[SDLK_SPACE]) {
//       action = PLAYER_A_FIRE;
//    }
//    else if (keystates[SDLK_ESCAPE]) {
//       std::cout << "Exiting..." << std::endl;
//       exit(0);
//    }

//    return alei.act(action);
// }

// ///////////////////////////////////////////////////////////////////////////////
// /// Print usage and exit
// ///////////////////////////////////////////////////////////////////////////////
// void usage(char const* pname) {
//    std::cerr
//       << "\nUSAGE:\n" 
//       << "   " << pname << " <romfile>\n";
//    exit(-1);
// }

// ///////////////////////////////////////////////////////////////////////////////
// /// MAIN PROGRAM
// ///////////////////////////////////////////////////////////////////////////////
// int main(int argc, char **argv) {
//    reward_t totalReward{};
//    ALEInterface alei{};

//    // Check input parameter
//    if (argc != 2)
//       usage(argv[0]);

//    // Configure alei object.
//    alei.setInt  ("random_seed", 0);
//    alei.setFloat("repeat_action_probability", 0);
//    alei.setBool ("display_screen", true);
//    alei.setBool ("sound", true);
//    alei.loadROM (argv[1]);

//    // Init
//    std::srand(static_cast<uint32_t>(std::time(0)));

//    // Main loop
//    {
//       uint32_t step{};
//       bool manualMode{false};
//       SDL_Event ev;
//       int32_t lives { alei.lives() };
//       std::cout << "Juego iniciado. Pulsa 'M' para cambiar a modo manual." << std::endl;
//       clearScreen();
//       while ( !alei.game_over() && step < maxSteps ) {
//          // Check for keypresses to toggle manual mode
//          printRAM(alei);
//          while (SDL_PollEvent(&ev)) {
//             if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_m) {
//                manualMode = !manualMode;
//                std::cout << (manualMode ? "Manual mode ON" : "Manual mode OFF")
//                          << std::endl;
//             }
//          }
         
//          // When we loose a live, we need to press FIRE to start again
//          if (alei.lives() < lives) {
//             lives = alei.lives();
//             alei.act(PLAYER_A_FIRE);
//          }

//          if (manualMode) {
//             totalReward += manualStep(alei);
//          } else {
//             totalReward += 1;
//          }

//          SDL_Delay(1000 / 60); // ~60 FPS
//          ++step;
//       }

//       std::cout << "Steps: " << step << std::endl;
//       std::cout << "Reward: " << totalReward << std::endl;
//    }

//    return 0;
// }