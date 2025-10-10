# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -std=c++14
INCLUDES = -Iinclude
LIBS = -lcufft

# Directories
SRCDIR = src
BINDIR = bin
TARGET = $(BINDIR)/vib_analysis

# Source files
SOURCES = $(SRCDIR)/main.cu $(SRCDIR)/kernels.cu

# Default target
all: $(TARGET)

# Create binary directory if not exists
$(BINDIR):
	mkdir -p $(BINDIR)

# Build target
$(TARGET): $(BINDIR) $(SOURCES)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LIBS)

# Clean rule
clean:
	rm -rf $(BINDIR) output

# Build rule (clean + compile)
build: clean all

.PHONY: all clean build
