SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
# -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: val hsre mhsre rhsre mrhsre modify mmodify dmodify rmodify

hsre : hsre.c
	$(CC) hsre.c -o ${BIN_DIR}/hsre $(CFLAGS)
rhsre : hsre.c
	$(CC) hsre.c -o ${BIN_DIR}/rhsre $(CFLAGS) -DDROPOUT -g
mhsre : hsre.c
	$(CC) hsre.c -o ${BIN_DIR}/mhsre $(CFLAGS) -DMARGIN
mrhsre : hsre.c
	$(CC) hsre.c -o ${BIN_DIR}/mhsre $(CFLAGS) -DMARGIN	-DDROPOUT
modify : modify.c
	$(CC) modify.c -o ${BIN_DIR}/modify $(CFLAGS)
rmodify : modify.c
	$(CC) modify.c -o ${BIN_DIR}/rmodify $(CFLAGS) -DDROPOUT
dmodify : modify.c
	$(CC) modify.c -o ${BIN_DIR}/dmodify $(CFLAGS) -DDEBUG
mmodify : modify.c
	$(CC) modify.c -o ${BIN_DIR}/mmodify $(CFLAGS) -DMARGIN
val : val.c
	$(CC) val.c -o ${BIN_DIR}/val $(CFLAGS)

clean:
	pushd ${BIN_DIR}