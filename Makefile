NVCC = nvcc
CXXFLAGS = -O3 -std=c++14
INCLUDES = -Iinclude
LIBS = -lcufft
SRCDIR = src
BINDIR = bin
TARGET = $(BINDIR)/vib_analysis

SOURCES = $(SRCDIR)/main.cu $(SRCDIR)/kernels.cu

all: $(TARGET)

$(BINDIR):
	mkdir -p $(BINDIR)

$(TARGET): $(BINDIR) $(SOURCES)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -rf $(BINDIR) output

.PHONY: all clean