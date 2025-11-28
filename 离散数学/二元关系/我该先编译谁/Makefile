SRC  := $(wildcard src/*.c)
OBJ  := $(SRC:src/%.c=build/%.o)

all: app
app: $(OBJ)
	gcc $^ -o $@

build/%.o: src/%.c | build
	gcc -MMD -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build app

show:
	python order.py