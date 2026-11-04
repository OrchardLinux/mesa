# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/msaa-vk

define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

MSAA_H := $(B)/msaa/color.vert.h $(B)/msaa/color.frag.h $(B)/msaa/smask.frag.h
$(eval $(call glsl,$(B)/msaa/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/msaa/color.frag.h,spv_color_frag,$(C)/color.frag))
$(eval $(call glsl,$(B)/msaa/smask.frag.h,spv_smask_frag,msaa/shaders/msaa-smask.frag))

$(B)/msaa-vk: msaa/msaa-vk.c $(MSAA_H)
	$(CC) $(CFLAGS) -I$(B)/msaa -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
