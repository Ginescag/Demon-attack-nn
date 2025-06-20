#include <iostream>
#include <cmath>
#include <cstdint>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h>
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
   reward_t reward{0};

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
   else { reward = alei.act(PLAYER_A_NOOP); } // Add a NOOP case

   return reward;
}


void clearScreen() {
   // Clear the console screen
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
/// Do Next Manual Step
///////////////////////////////////////////////////////////////////////////////
reward_t manualStep(ALEInterface& alei) {
   Action action = PLAYER_A_NOOP;
   Uint8* keystates = SDL_GetKeyState(NULL);

   if (keystates[SDLK_LEFT]) {
      action = PLAYER_A_LEFT;
   }
   else if (keystates[SDLK_RIGHT]) {
      action = PLAYER_A_RIGHT;
   }
   else if (keystates[SDLK_SPACE]) {
      action = PLAYER_A_FIRE;
   }
   else if (keystates[SDLK_ESCAPE]) {
      std::cout << "Exiting..." << std::endl;
      exit(0);
   }

   return alei.act(action);
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
      uint32_t step{};
      bool manualMode{false};
      SDL_Event ev;
      int32_t lives { alei.lives() };
      std::cout << "Juego iniciado. Pulsa 'M' para cambiar a modo manual." << std::endl;
      clearScreen();
      while ( !alei.game_over() && step < maxSteps ) {
         // Check for keypresses to toggle manual mode
         printRAM(alei);
         while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_m) {
               manualMode = !manualMode;
               std::cout << (manualMode ? "Manual mode ON" : "Manual mode OFF")
                         << std::endl;
            }
         }
         
         // When we loose a live, we need to press FIRE to start again
         if (alei.lives() < lives) {
            lives = alei.lives();
            alei.act(PLAYER_A_FIRE);
         }

         if (manualMode) {
            totalReward += manualStep(alei);
         } else {
            totalReward += agentStep(alei, model);
         }

         SDL_Delay(1000 / 60); // ~60 FPS
         ++step;
      }

      std::cout << "Steps: " << step << std::endl;
      std::cout << "Reward: " << totalReward << std::endl;
   }

   return 0;
}
