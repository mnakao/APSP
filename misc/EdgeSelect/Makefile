CC=gcc
CFLAGS=-O2 -std=gnu99 -Wall
LDFLAGS=-lm
#################################################
a.out: main.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

main.o: main.c
	$(CC) $(CFLAGS) $< -c -o $@
#################################################
all: a.out
clean:
	rm -rf *.o *~

clean_all: clean
	rm -f a.out
