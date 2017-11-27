# 
#Makefile
#

MF= Makefile

EXE=cuda

SRC=example7.cu

ARCH=sm_60
CC=/usr/local/cuda-8.0/bin/nvcc

ROWS=3000
COLS=3000
CFLAGS= -O3 -arch=$(ARCH) -DROWS=$(ROWS) -DCOLS=$(COLS) 



.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)


.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

default: $(EXE)

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) core
