
CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17 -Ilibs/Arcade-Learning-Environment-0.6.1 -Isrc
LDFLAGS = -Llibs/Arcade-Learning-Environment-0.6.1 -Wl,-rpath,'$$ORIGIN/../libs/Arcade-Learning-Environment-0.6.1' -lale -lz -lSDL

SRC_DIR = src
BUILD_DIR = build

# Archivos fuente
SRC_MAIN = libs/Arcade-Learning-Environment-0.6.1/minimal_agent.cpp \
           $(SRC_DIR)/perceptron.cpp
SRC_TRAIN = $(SRC_DIR)/train_model.cpp $(SRC_DIR)/perceptron.cpp

# Archivos objeto
OBJ_MAIN = $(BUILD_DIR)/minimal_agent.o $(BUILD_DIR)/perceptron.o
OBJ_TRAIN = $(SRC_TRAIN:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Ejecutables
MAIN_EXEC = $(BUILD_DIR)/demon_bot
TRAIN_EXEC = $(BUILD_DIR)/train_model

all: $(MAIN_EXEC) $(TRAIN_EXEC)

$(MAIN_EXEC): $(OBJ_MAIN)
	@echo "Compilando demon_bot..."
	$(CXX) $^ -o $@ $(LDFLAGS)

$(TRAIN_EXEC): $(OBJ_TRAIN)
	@echo "Compilando train_model..."
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/minimal_agent.o: libs/Arcade-Learning-Environment-0.6.1/minimal_agent.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(MAIN_EXEC) $(TRAIN_EXEC)
