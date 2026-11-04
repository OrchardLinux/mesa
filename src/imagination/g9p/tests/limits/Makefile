CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
all: build/limits
build/limits: limits.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< -lvulkan
clean:
	rm -rf build
.PHONY: all clean
