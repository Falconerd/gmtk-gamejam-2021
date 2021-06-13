flags = -Wall -pedantic -std=c99
libs = -lX11 -lglfw -ldl
inc = -I./deps/include

build: main.c glad.o
	gcc -g3 $(flags) $(libs) $(inc) $^

glad.o: deps/src/glad.c
	gcc -c $(inc) $^

run:
	./a.out

clean:
	@rm -f ./a.out
	@rm -f ./*.o
	@rm -f ./*.obj
