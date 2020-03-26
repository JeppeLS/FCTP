# Subject: makefile for the library libFCTPy.so (to be used from Python)
# Date   : Dec 26, 2019

# Compilers:
CC = x86_64-w64-mingw32-gcc
CFLAGS = -c -O2 -I../src

# Object files
obj = rnd.o FCTPheur.o

# Library to be created
lib = FCTPy.dll

$(lib):	$(obj)
	$(CC) -shared $(obj) -o $@ -Wl,--out-implib,FCTPy.lib

%.o:	../src/%.c
	$(CC) $(CFLAGS) $< -o $@

clean-o:
	rm -f $(obj)

clean:
	rm -f $(obj) $(lib)

