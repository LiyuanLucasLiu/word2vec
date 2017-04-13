SCRIPTS_DIR=../scripts
BIN_DIR=../bin
SRC_DIR=../src

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
# -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: wfinalExp_ht wfinalExp TD
	#distance Rep
#finalExp repExp 
#val hsre mhsre rhsre mrhsre modify mmodify rmodify armodify mrmodify

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
# finalExp : finalExp.c
	# $(CC) finalExp.c -o ${BIN_DIR}/finalExp $(CFLAGS) -DDROPOUT -DACTIVE	
wfinalExp : wfinalExp.c
	$(CC) wfinalExp.c -o ${BIN_DIR}/wfinalExp $(CFLAGS) -DDROPOUT -DACTIVE	
wfinalExp_ht : wfinalExp_ht.c
	$(CC) wfinalExp_ht.c -o ${BIN_DIR}/wfinalExp_ht $(CFLAGS) -DDROPOUT -DACTIVE	
Rep : RepresentCS.c
	$(CC) RepresentCS.c -o ${BIN_DIR}/repExp $(CFLAGS) -DDROPOUT -DACTIVE	
TD : TruthCS.c
	$(CC) TruthCS.c -o ${BIN_DIR}/tdExp $(CFLAGS) -DDROPOUT -DACTIVE	
# distance: distance.c
	# $(CC) distance.c -o ${BIN_DIR}/distance $(CFLAGS)
# val : val.c
	# $(CC) val.c -o ${BIN_DIR}/val $(CFLAGS)

clean:
	pushd ${BIN_DIR}