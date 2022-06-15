CC=gcc
CFLAGS=-c -Wall
LFLAGS=-lm
BIN=sort

all: $(BIN)

$(BIN): SeqBitonicSort.o
	$(CC) $^ $(LFLAGS) -o $(BIN)

SeqBitonicSort.o: SeqBitonicSort.c
	$(CC) $(CFLAGS) $< $(LFLAGS) -o $@

clean:
	rm *.o
	rm $(BIN)
