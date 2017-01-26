SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
# -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops 

all: recol compute-accuracy

recol : recol.c
	$(CC) recol.c -o ${BIN_DIR}/recol $(CFLAGS) -g
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS) -g

clean:
	pushd ${BIN_DIR} && rm -rf recol; popd
