# Compiler settings
CC := nvcc

# Directories
SRC_DIR := algo
BUILD_DIR := build
BIN_DIR := .

# Make sure build directories exist
$(shell mkdir -p $(BUILD_DIR)/$(SRC_DIR))

# Files
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/$(SRC_DIR)/%.o,$(CU_SRCS))
DEPS := $(wildcard $(SRC_DIR)/*.h) $(wildcard $(SRC_DIR)/*.cuh)

# Target executable
TARGET := $(BIN_DIR)/main

# Flags
NVCC_FLAGS := -arch=sm_70 -lineinfo

# Rules
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(CU_OBJS)
	$(CC) $^ -o $@

# Pattern rule for .cu files
$(BUILD_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/$(SRC_DIR)/*.o $(TARGET)
