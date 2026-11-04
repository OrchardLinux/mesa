# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/depth-vk $(B)/ds-sample-vk

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
DSAMP_H := $(B)/dsamp/color.vert.h $(B)/dsamp/color.frag.h $(B)/dsamp/uv.vert.h $(B)/dsamp/ds-raw.frag.h \
           $(B)/dsamp/ds-cmp.frag.h $(B)/dsamp/ds-stencil.frag.h
$(eval $(call glsl,$(B)/dsamp/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/dsamp/color.frag.h,spv_color_frag,$(C)/color.frag))
$(eval $(call glsl,$(B)/dsamp/uv.vert.h,spv_uv_vert,$(C)/uv.vert))
$(eval $(call glsl,$(B)/dsamp/ds-raw.frag.h,spv_ds_raw_frag,depth/shaders/ds-raw.frag))
$(eval $(call glsl,$(B)/dsamp/ds-cmp.frag.h,spv_ds_cmp_frag,depth/shaders/ds-cmp.frag))
$(eval $(call glsl,$(B)/dsamp/ds-stencil.frag.h,spv_ds_stencil_frag,depth/shaders/ds-stencil.frag))

$(B)/ds-sample-vk: depth/ds-sample-vk.c $(DSAMP_H)
	$(CC) $(CFLAGS) -I$(B)/dsamp -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
