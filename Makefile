CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17 -Ilibs/Arcade-Learning-Environment-0.6.1 -Isrc
LDFLAGS = -Llibs/Arcade-Learning-Environment-0.6.1 -Wl,-rpath,'$$ORIGIN/../libs/Arcade-Learning-Environment-0.6.1' -lale -lz -lSDL

SRC_DIR = src
BUILD_DIR = build

# --- Archivos para el Agente DQN (Original) ---
SRC_DQN = libs/Arcade-Learning-Environment-0.6.1/minimal_agent.cpp \
          $(SRC_DIR)/neural_network.cpp
OBJ_DQN = $(BUILD_DIR)/minimal_agent.o $(BUILD_DIR)/neural_network.o
EXEC_DQN = $(BUILD_DIR)/demon_bot

# --- Archivos para el Agente Genético ---
SRC_GENETIC = libs/Arcade-Learning-Environment-0.6.1/minimal_agent_genetic.cpp \
              $(SRC_DIR)/neural_network.cpp
OBJ_GENETIC = $(BUILD_DIR)/minimal_agent_genetic.o $(BUILD_DIR)/neural_network.o
EXEC_GENETIC = $(BUILD_DIR)/demon_bot_genetic

# --- Reglas de Compilación ---

# El objetivo 'all' compila ambos ejecutables
all: $(EXEC_DQN) $(EXEC_GENETIC)

# Regla para el ejecutable DQN
$(EXEC_DQN): $(OBJ_DQN)
	@echo "Compilando demon_bot (DQN)..."
	$(CXX) $^ -o $@ $(LDFLAGS)

# Regla para el ejecutable Genético
$(EXEC_GENETIC): $(OBJ_GENETIC)
	@echo "Compilando demon_bot_genetic..."
	$(CXX) $^ -o $@ $(LDFLAGS)

# --- Reglas para los Archivos Objeto ---

# Regla para minimal_agent.o
$(BUILD_DIR)/minimal_agent.o: libs/Arcade-Learning-Environment-0.6.1/minimal_agent.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para minimal_agent_genetic.o
$(BUILD_DIR)/minimal_agent_genetic.o: libs/Arcade-Learning-Environment-0.6.1/minimal_agent_genetic.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla genérica para los archivos en src/ (como neural_network.cpp)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Regla de Limpieza ---
clean:
	rm -rf $(BUILD_DIR)/*.o $(EXEC_DQN) $(EXEC_GENETIC)
