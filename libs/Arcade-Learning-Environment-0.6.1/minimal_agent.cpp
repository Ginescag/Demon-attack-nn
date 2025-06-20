#include <iostream>
#include <cmath>
#include <cstdint>
#include "src/ale_interface.hpp"
#include <SDL.h>
#include "perceptron.hpp"
#include <vector>

// Constants
constexpr uint32_t maxSteps = 7500;

///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////
int32_t getPlayerX(ALEInterface& alei) {
   return alei.getRAM().get(72) + ((rand() % 3) - 1);
}

int32_t getBallX(ALEInterface& alei) {
   return alei.getRAM().get(99) + ((rand() % 3) - 1);
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////
reward_t agentStep(ALEInterface& alei, Perceptron& model) {
   static int32_t lives { alei.lives() };
   reward_t reward{0};

   // When we loose a live, we need to press FIRE to start again
   if (alei.lives() < lives) {
      lives = alei.lives();
      alei.act(PLAYER_A_FIRE);
   }

   // Obtener observaciones y predecir la acción con el perceptrón
   auto playerX { getPlayerX(alei) };
   auto ballX   { getBallX(alei)   };
   std::vector<float> inputs {
      static_cast<float>(playerX) / 255.0f,
      static_cast<float>(ballX) / 255.0f
   };
   int action = model.predict(inputs);

   if (action == 0)      { reward = alei.act(PLAYER_A_LEFT);  }
   else if (action == 1) { reward = alei.act(PLAYER_A_RIGHT); }

   return reward + alei.act(PLAYER_A_NOOP);
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>\n";
   exit(-1);
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   reward_t totalReward{};
   ALEInterface alei{};
   Perceptron model(2);


   // Check input parameter
   if (argc != 2)
      usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", true);
   alei.loadROM (argv[1]);

   // Init
   std::srand(static_cast<uint32_t>(std::time(0)));

   // Main loop
   {
      alei.act(PLAYER_A_FIRE);
      uint32_t step{};
      bool manualMode{false};
      SDL_Event ev;
      while ( !alei.game_over() && step < maxSteps ) {
         // Check for keypresses to toggle manual mode
         while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_m) {
               manualMode = !manualMode;
               std::cout << (manualMode ? "Manual mode ON" : "Manual mode OFF")
                         << std::endl;
            }
         }

         if (manualMode) {
            SDL_PumpEvents();
            Uint8* state = SDL_GetKeyState(NULL);
            if (state[SDLK_LEFT]) {
               totalReward += alei.act(PLAYER_A_LEFT);
            } else if (state[SDLK_RIGHT]) {
               totalReward += alei.act(PLAYER_A_RIGHT);
            } else {
               totalReward += alei.act(PLAYER_A_NOOP);
            }
         } else {
            totalReward += agentStep(alei, model);
         }
         ++step;
      }

      std::cout << "Steps: " << step << std::endl;
      std::cout << "Reward: " << totalReward << std::endl;
   }

   return 0;
}
