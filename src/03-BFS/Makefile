.PHONY: all clean run asm

INC = -I./include
NVCC = nvcc
CC = gcc
CFLAGS = $(INC) -O3 -g -std=c++11
NVCCFLAGS = $(NVCCLIBS) $(CFLAGS) --resource-usage
NVCCFLAGS += -Xcompiler -fopenmp

CUSRC = spmbfs.cu 
OBJ_SRC = $(sort $(wildcard *.cu))
OBJ = $(patsubst %.cu,%.o,$(OBJ_SRC)) 
EXE = $(patsubst %.cu,%.exe,$(CUSRC))

all: $(EXE) asm
	@echo "built:" $(EXE)

asm: spmbfs.ptx

run: $(EXE)
	./spmbfs.exe ./test_matrix.mtx

clean:
	rm -rf $(EXE) *.o *.exe.dSYM *.optrpt *.asm *.ptx *.so

$(EXE): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.ptx : %.cu
	$(NVCC) $(NVCCFLAGS) --ptx -o $@ $<

%.o : %.cu ./include/*
	$(NVCC) $(NVCCFLAGS) -c $<
