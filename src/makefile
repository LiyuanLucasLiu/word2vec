SCRIPTS_DIR=../scripts
BIN_DIR=../bin
SRC_DIR=../src

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
# -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: nfinalExp finalExp repExp#val hsre mhsre rhsre mrhsre modify mmodify rmodify armodify mrmodify

# hsre : hsre.c
# 	$(CC) hsre.c -o ${BIN_DIR}/hsre $(CFLAGS)
# rhsre : hsre.c
# 	$(CC) hsre.c -o ${BIN_DIR}/rhsre $(CFLAGS) -DDROPOUT -g
# mhsre : hsre.c
# 	$(CC) hsre.c -o ${BIN_DIR}/mhsre $(CFLAGS) -DMARGIN
# mrhsre : hsre.c
# 	$(CC) hsre.c -o ${BIN_DIR}/mhsre $(CFLAGS) -DMARGIN	-DDROPOUT
# modify : modify.c
# 	$(CC) modify.c -o ${BIN_DIR}/modify $(CFLAGS)
# rmodify : modify.c
# 	$(CC) modify.c -o ${BIN_DIR}/rmodify $(CFLAGS) -DDROPOUT
# armodify : modify.c
# 	$(CC) modify.c -o ${BIN_DIR}/armodify $(CFLAGS) -DDROPOUT -DACTIVE
# mmodify : modify.c
# 	$(CC) modify.c -o ${BIN_DIR}/dmodify $(CFLAGS) -DMARGIN
# mrmodify : modify.c
# 	$(CC) modify.c -o ${BIN_DIR}/mmodify $(CFLAGS) -DMARGIN -DDROPOUT
finalExp : finalExp.c
	$(CC) finalExp.c -o ${BIN_DIR}/finalExp $(CFLAGS) -DDROPOUT -DACTIVE	
nfinalExp : finalExp.c
	$(CC) nfinalExp.c -o ${BIN_DIR}/nfinalExp $(CFLAGS) -DDROPOUT -DACTIVE	
repExp : entityRepresentation.c
	$(CC) entityRepresentation.c -o ${BIN_DIR}/repExp $(CFLAGS) -DDROPOUT -DACTIVE	
val : val.c
	$(CC) val.c -o ${BIN_DIR}/val $(CFLAGS)

clean:
	pushd ${BIN_DIR}