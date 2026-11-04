# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -g -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/depth-vk

define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

DEPTH_H := $(B)/depth/color.vert.h $(B)/depth/color.frag.h
$(eval $(call glsl,$(B)/depth/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/depth/color.frag.h,spv_color_frag,$(C)/color.frag))
$(B)/depth-vk: depth/depth-vk.c $(DEPTH_H)
	$(CC) $(CFLAGS) -I$(B)/depth -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
