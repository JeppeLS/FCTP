# Subject: makefile for the library libFCTPy.so (to be used from Python)
# Date   : Dec 20, 2019

# Compilers:
CC = gcc

# Librarires
LIBS = -lm

# Source files
src = ../src

# Flags
CFLAGS = -shared -Ofast -Wall -I$(src) -fPIC

# Object files
obj = rnd.o FCTPheur.o

# Library to be created
lib = libFCTPy.so

$(lib):	$(obj)
	$(LINK.c) -o $@ $(obj) $(LIBS)
	execstack -c $(lib)
			
%.o:	../src/%.c		
	$(COMPILE.c) $<
	
clean-o:
	rm -f $(obj)
	
clean:
	rm -f $(obj) $(lib)
