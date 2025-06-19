#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "perceptron.hpp"

using namespace std;

int main() {
    ifstream dataFile("data/game_data.csv");
    if (!dataFile.is_open()) {
        cerr << "Error al abrir el archivo de datos." << endl;
        return 1;
    }

    string line;
    getline(dataFile, line);  // Ignorar la cabecera

    vector<vector<float>> inputs;
    vector<int> targets;

    // Leer los datos del archivo
    while (getline(dataFile, line)) {
        stringstream ss(line);
        vector<float> input(3);  // Definimos el vector de entrada aquí
        int action;

        ss >> input[0];  // frame (no lo usamos para el modelo)
        ss.ignore(1);    // Ignorar la coma
        ss >> input[0];  // player_x
        ss.ignore(1);
        ss >> input[1];  // enemy_x
        ss.ignore(1);
        ss >> input[2];  // enemy_y
        ss.ignore(1);
        ss >> action;    // action

        // Normalizar las entradas
        float max_value = 255.0;  // Ajusta según el rango esperado
        input[0] /= max_value;
        input[1] /= max_value;
        input[2] /= max_value;

        inputs.push_back(input);
        targets.push_back(action);
    }

    dataFile.close();

    // Crear el perceptrón
    Perceptron perceptron(3);  // 3 entradas: player_x, enemy_x, enemy_y
    size_t maxEpochs = 1000;
    float previousError = 0.0f;

    // Entrenar el perceptrón
    for (size_t epoch = 1; epoch <= maxEpochs; ++epoch) {
        float totalError = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            int prediction = perceptron.predict(inputs[i]);
            int error = targets[i] - prediction;
            totalError += abs(error);  // Suma del error absoluto
            perceptron.train(inputs[i], targets[i]);
        }

        // Mostrar el progreso
        cout << "Época: " << epoch << " | Error Total: " << totalError << endl;

        // Condición de convergencia
        if (abs(totalError - previousError) < 1e-5) {
            cout << "Convergencia alcanzada después de " << epoch << " épocas." << endl;
            break;
        }

        previousError = totalError;
    }

    cout << "Entrenamiento completado." << endl;

    // Guardar los pesos entrenados
    ofstream weightsFile("data/perceptron_weights.txt");
    if (weightsFile.is_open()) {
        for (float weight : perceptron.getWeights()) {
            weightsFile << weight << " ";
        }
        weightsFile.close();
        cout << "Pesos guardados correctamente en data/perceptron_weights.txt" << endl;
    } else {
        cerr << "Error al guardar los pesos entrenados." << endl;
    }

    return 0;
}
