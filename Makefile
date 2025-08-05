.PHONY: all build test clean

all: build test

clean:
	rm -rf build/

build: clean
	mkdir build/ && cd build && cmake .. && make -j4

test:
	cd build && make test
