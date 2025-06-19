# Variables
CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17 -Ilibs/Arcade-Learning-Environment-0.6.1/src
LDFLAGS = -Llibs/Arcade-Learning-Environment-0.6.1 -lale_interface -lz -lSDL
SRC_DIR = src
BUILD_DIR = build

# Archivos fuente
SRC_MAIN = $(SRC_DIR)/main.cpp $(SRC_DIR)/perceptron.cpp
SRC_TRAIN = $(SRC_DIR)/train_model.cpp $(SRC_DIR)/perceptron.cpp

# Archivos objeto
OBJ_MAIN = $(SRC_MAIN:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
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

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(MAIN_EXEC) $(TRAIN_EXEC)
