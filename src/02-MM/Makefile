.PHONY: all clean run asm

INC = -I./include
NVCC = nvcc
CC = gcc
CFLAGS = $(INC) -O3 -g
NVCCFLAGS = $(NVCCLIBS) $(CFLAGS) --resource-usage
NVCCFLAGS += -Xcompiler -fopenmp

CUSRC = spmv.cu 
OBJ_SRC = $(sort $(wildcard *.cu))
OBJ = $(patsubst %.cu,%.o,$(OBJ_SRC)) 
EXE = $(patsubst %.cu,%.exe,$(CUSRC))

all: $(EXE) asm
	@echo "built:" $(EXE)

asm: spmv.ptx

run: $(EXE)
	spmv.exe ./test_matrix.mtx

clean:
	rm -rf $(EXE) *.o *.exe.dSYM *.optrpt *.asm *.ptx *.so

$(EXE): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.ptx : %.cu
	$(NVCC) $(NVCCFLAGS) --ptx -o $@ $<

%.o : %.cu ./include/*
	$(NVCC) $(NVCCFLAGS) -c $<
