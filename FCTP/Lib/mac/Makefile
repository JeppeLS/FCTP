# Subject: makefile for the library libFCTPy.so (to be used from Python)
# Date   : Feb 14, 2020

# Compilers:
CC = x86_64-apple-darwin15-clang

# Flags
CFLAGS = -c -O2 -I../src --target=x86_64-apple-darwin

# Object files
obj = rnd.o FCTPheur.o

# Library to be created
lib = libFCTPy.so

# make rules follow:

$(lib):	$(obj)
	$(CC) -shared $(obj) -undefined dynamic_lookup -o $@ 

%.o:	../src/%.c
	$(CC) $(CFLAGS) $< -o $@

clean-o:
	rm -f $(obj)
clean:
	rm -f $(obj) $(lib)

