# Compiler and linker
CC = /usr/local/cuda-12.0/bin/nvcc
CFLAGS =
LDFLAGS =

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
EXECUTABLE = $(BIN_DIR)/gds_sandbox

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*.o $(EXECUTABLE)

.PHONY: all clean

