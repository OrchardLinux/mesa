# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -g -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/tt-vk

define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

TT_SH := tt-arr tt-arr-lod tt-arr-lod1 tt-cube tt-lod
TT_H := $(B)/tt/pass_vert.h $(B)/tt/solid_frag.h $(B)/tt/uv_vert.h $(B)/tt/tex_frag.h \
        $(foreach s,$(TT_SH),$(B)/tt/$(subst -,_,$(s))_frag.h)
$(eval $(call glsl,$(B)/tt/pass_vert.h,pass_vert,$(C)/pass.vert))
$(eval $(call glsl,$(B)/tt/solid_frag.h,solid_frag,$(C)/solid.frag))
$(eval $(call glsl,$(B)/tt/uv_vert.h,uv_vert,$(C)/uv.vert))
$(eval $(call glsl,$(B)/tt/tex_frag.h,tex_frag,$(C)/tex.frag))
$(foreach s,$(TT_SH),$(eval $(call glsl,$(B)/tt/$(subst -,_,$(s))_frag.h,$(subst -,_,$(s))_frag,textures/shaders/$(s).frag)))

$(B)/tt-vk: textures/tt-vk.c $(TT_H)
	$(CC) $(CFLAGS) -I$(B)/tt -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
