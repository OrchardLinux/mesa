# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -g -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/mrt-vk

define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

MRT_H := $(B)/mrt/pass.vert.h $(B)/mrt/one.frag.h $(B)/mrt/mrt2.frag.h $(B)/mrt/mrt4.frag.h \
         $(B)/mrt/mrt-vary.vert.h $(B)/mrt/mrt4-vary.frag.h
$(eval $(call glsl,$(B)/mrt/pass.vert.h,spv_pass_vert,$(C)/pass.vert))
$(eval $(call glsl,$(B)/mrt/one.frag.h,spv_one_frag,$(C)/solid.frag))
$(eval $(call glsl,$(B)/mrt/mrt2.frag.h,spv_mrt2_frag,mrt/shaders/mrt2.frag))
$(eval $(call glsl,$(B)/mrt/mrt4.frag.h,spv_mrt4_frag,mrt/shaders/mrt4.frag))
$(eval $(call glsl,$(B)/mrt/mrt-vary.vert.h,spv_mrt_vary_vert,mrt/shaders/mrt-vary.vert))
$(eval $(call glsl,$(B)/mrt/mrt4-vary.frag.h,spv_mrt4_vary_frag,mrt/shaders/mrt4-vary.frag))

$(B)/mrt-vk: mrt/mrt-vk.c $(MRT_H)
	$(CC) $(CFLAGS) -I$(B)/mrt -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
