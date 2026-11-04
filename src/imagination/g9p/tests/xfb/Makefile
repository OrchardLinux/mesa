# SPDX-License-Identifier: MIT
CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

all: $(B)/xfb-test

define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

XFB_SH := xfb-rec.vert xfb-attr.vert xfb-attr16.vert xfb-inst.vert xfb-dup.vert xfb-cover.vert xfb.frag
XFB_H := $(XFB_SH:%=$(B)/xfb/%.h)
$(foreach s,$(XFB_SH),$(eval $(call glsl,$(B)/xfb/$(s).h,$(subst -,_,$(subst .,_,$(s))),xfb/shaders/$(s))))

$(B)/xfb-test: xfb/xfb-test.c $(XFB_H)
	$(CC) $(CFLAGS) -I$(B)/xfb -o $@ $< $(VK_LIBS)
clean:
	rm -rf $(B)
.PHONY: all clean
