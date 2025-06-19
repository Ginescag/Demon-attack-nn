#include <ale_interface.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "perceptron.hpp"

using namespace std;

int main() {
    // Inicializar ALE
    ALEInterface ale;

    // Configurar el entorno
    ale.setBool("display_screen", true);
    ale.setInt("random_seed", 123);
    ale.setFloat("repeat_action_probability", 0.25);

    // Cargar la ROM
    string romPath = "/home/cpc/Escritorio/atascaburrasStudios-P2/roms/demon_attack.bin";
    try {
        ale.loadROM(romPath);
        cout << "ROM cargada con éxito: " << romPath << endl;
    } catch (const exception &e) {
        cerr << "Error al cargar la ROM: " << e.what() << endl;
        return 1;
    }

    // Obtener las acciones legales
    vector<Action> legalActions = ale.getLegalActionSet();

    // Mostrar el tamaño de la RAM
    auto ram = ale.getRAM();
    cout << "Tamaño de la RAM: " << ram.size() << endl;

    // Crear el archivo de datos
    ofstream dataFile("data/game_data.csv");
    if (!dataFile.is_open()) {
        cerr << "Error al abrir el archivo de datos." << endl;
        return 1;
    }
    dataFile << "frame,player_x,enemy_x,enemy_y,action\n";

    int score = 0;

    // Bucle principal del juego
    while (!ale.game_over()) {
        // Obtener el estado del juego desde la RAM
        int player_x = ram.get(20);  // Propuesta: posición X del jugador
        int enemy_x = ram.get(30);  // Propuesta: posición X del enemigo
        int enemy_y = ram.get(31);  // Propuesta: posición Y del enemigo

        // Seleccionar una acción aleatoria
        Action action = legalActions[rand() % legalActions.size()];
        score += ale.act(action);

        // Guardar los datos en el archivo
        dataFile << ale.getEpisodeFrameNumber() << ","
                 << player_x << ","
                 << enemy_x << ","
                 << enemy_y << ","
                 << action << "\n";

        // Mostrar la puntuación actual
        cout << "Frame: " << ale.getEpisodeFrameNumber()
             << " | Puntuación: " << score << endl;
    }

    dataFile.close();

    // Mensaje final
    cout << "Fin del juego. Puntuación final: " << score << endl;

    return 0;
}
