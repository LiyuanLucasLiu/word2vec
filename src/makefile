SCRIPTS_DIR=../scripts
BIN_DIR=../bin
SRC_DIR=../src

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: ReHession ReHessionN ReHessionN_CS410 printVal

ReHession: ReHession.c
	$(CC) ReHession.c -o ${BIN_DIR}/ReHession $(CFLAGS) -DDROPOUT -DACTIVE

ReHessionN: ReHessionN.c
	$(CC) ReHessionN.c -o ${BIN_DIR}/ReHessionN $(CFLAGS) -DDROPOUT -DACTIVE

ReHessionN_CS410: ReHessionN_CS410.c
	$(CC) ReHessionN_CS410.c -o ${BIN_DIR}/ReHessionN_CS410 $(CFLAGS) -DDROPOUT -DACTIVE

printVal: printVal.c
	$(CC) printVal.c -o ${BIN_DIR}/printVal $(CFLAGS) -DDROPOUT -DACTIVE

clean:
	pushd ${BIN_DIR}