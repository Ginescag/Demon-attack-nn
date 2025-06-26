#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h>

struct GameplayFrame {
    std::vector<uint8_t> ram_state;
    Action action;
};

void printRAM(ALEInterface& alei) {
   auto* RAM = alei.getRAM().array();
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

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_a_la_rom>" << std::endl;
        return 1;
    }

    ALEInterface alei{};
    alei.setBool("display_screen", true);
    alei.setBool("sound", true);
    alei.setInt("frame_skip", 1);  // Sin saltar frames para capturar todo
    alei.loadROM(argv[1]);

    std::vector<GameplayFrame> recorded_gameplay;
    std::cout << "Grabando partida. Controles: Flechas para mover, Espacio para disparar, ESC para terminar." << std::endl;

    while (true) {
        printRAM(alei);
        std::cout << "\nVidas: " << alei.lives() << " | Frames grabados: " << recorded_gameplay.size() << std::endl;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                goto end_recording;  // Salir de los bucles anidados
            }
        }

        Uint8* keystates = SDL_GetKeyState(NULL);
        Action action = PLAYER_A_NOOP;

        if (keystates[SDLK_LEFT] && keystates[SDLK_SPACE]) action = PLAYER_A_LEFTFIRE;
        else if (keystates[SDLK_RIGHT] && keystates[SDLK_SPACE]) action = PLAYER_A_RIGHTFIRE;
        else if (keystates[SDLK_LEFT]) action = PLAYER_A_LEFT;
        else if (keystates[SDLK_RIGHT]) action = PLAYER_A_RIGHT;
        else if (keystates[SDLK_SPACE]) action = PLAYER_A_FIRE;

        // Guardar el estado actual y la acción
        GameplayFrame frame;
        frame.ram_state = std::vector<uint8_t>(alei.getRAM().array(), alei.getRAM().array() + 128);
        frame.action = action;
        recorded_gameplay.push_back(frame);

        // Ejecutar acción
        alei.act(action);

        if (alei.game_over()) {
            std::cout << "\n¡Juego terminado! Reiniciando..." << std::endl;
            alei.reset_game();
            SDL_Delay(1000);  // Pausa de 1 segundo entre partidas
        }

        SDL_Delay(1000/60);  // ~60 FPS
    }

end_recording:
    // Guardar los datos en un archivo
    std::ofstream outfile("demon_gameplay_data.bin", std::ios::binary);
    size_t num_frames = recorded_gameplay.size();
    outfile.write(reinterpret_cast<const char*>(&num_frames), sizeof(num_frames));
    
    for (const auto& frame : recorded_gameplay) {
        size_t ram_size = frame.ram_state.size();
        outfile.write(reinterpret_cast<const char*>(&ram_size), sizeof(ram_size));
        outfile.write(reinterpret_cast<const char*>(frame.ram_state.data()), ram_size);
        outfile.write(reinterpret_cast<const char*>(&frame.action), sizeof(frame.action));
    }
    outfile.close();

    std::cout << "Grabación finalizada. Se guardaron " << recorded_gameplay.size() << " frames." << std::endl;
    return 0;
}