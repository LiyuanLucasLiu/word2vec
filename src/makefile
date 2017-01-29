SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
# -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops 

all: recol compute-accuracy old_recol gcrecol

recol : recol.c
	$(CC) recol.c -o ${BIN_DIR}/recol $(CFLAGS)
gcrecol : recol.c
	$(CC) recol.c -o ${BIN_DIR}/gcrecol $(CFLAGS) -DGRADCLIP
old_recol : old_recol.c
	$(CC) old_recol.c -o ${BIN_DIR}/old_recol $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)

clean:
	pushd ${BIN_DIR}