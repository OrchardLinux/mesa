# SPDX-License-Identifier: MIT
# Builds the G9P ES 3.0 rungs. Each program's SPIR-V headers are generated from the GLSL here by
# glslangValidator into build/<program>/ (no SPIR-V is shipped). Native aarch64 Linux:
#   make            all eight programs into build/
#   make vk         the Vulkan ones only (mrt-vk xfb-test depth-vk ds-sample-vk msaa-vk tt-vk)
#   make gles       the EGL/GLES ones only (es3-smoke ds-ladder)
#   make headers    only the generated headers (for a syntax check elsewhere)
#   make clean
# Overrides: CC, CFLAGS, GLSLANG, VK_LIBS, GLES_LIBS (e.g. CFLAGS+=-I/path/to/headers).

CC      ?= cc
GLSLANG ?= glslangValidator
CFLAGS  ?= -std=gnu11 -O1 -Wall -Wextra -Wno-missing-field-initializers
VK_LIBS ?= -lvulkan -lm
GLES_LIBS ?= -lEGL -lGLESv2 -lm
B := build

VK_PROGS   := $(B)/mrt-vk $(B)/xfb-test $(B)/depth-vk $(B)/ds-sample-vk $(B)/msaa-vk $(B)/tt-vk
GLES_PROGS := $(B)/es3-smoke $(B)/ds-ladder

all: vk gles
vk: $(VK_PROGS)
gles: $(GLES_PROGS)

# $(call glsl,OUT.h,VARNAME,SRC): one SPIR-V module as a uint32_t array named VARNAME
define glsl
$(1): $(3)
	@mkdir -p $$(@D)
	$$(GLSLANG) -V --vn $(2) $$< -o $$@ >/dev/null
endef

C := shaders/common

# mrt-vk: spv_<file name with . and - as _>; one.frag is the common solid red fragment program
MRT_H := $(B)/mrt/pass.vert.h $(B)/mrt/one.frag.h $(B)/mrt/mrt2.frag.h $(B)/mrt/mrt4.frag.h \
         $(B)/mrt/mrt-vary.vert.h $(B)/mrt/mrt4-vary.frag.h
$(eval $(call glsl,$(B)/mrt/pass.vert.h,spv_pass_vert,$(C)/pass.vert))
$(eval $(call glsl,$(B)/mrt/one.frag.h,spv_one_frag,$(C)/solid.frag))
$(eval $(call glsl,$(B)/mrt/mrt2.frag.h,spv_mrt2_frag,mrt/shaders/mrt2.frag))
$(eval $(call glsl,$(B)/mrt/mrt4.frag.h,spv_mrt4_frag,mrt/shaders/mrt4.frag))
$(eval $(call glsl,$(B)/mrt/mrt-vary.vert.h,spv_mrt_vary_vert,mrt/shaders/mrt-vary.vert))
$(eval $(call glsl,$(B)/mrt/mrt4-vary.frag.h,spv_mrt4_vary_frag,mrt/shaders/mrt4-vary.frag))

# depth-vk, msaa-vk: spv_color_vert / spv_color_frag (+ spv_smask_frag)
DEPTH_H := $(B)/depth/color.vert.h $(B)/depth/color.frag.h
$(eval $(call glsl,$(B)/depth/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/depth/color.frag.h,spv_color_frag,$(C)/color.frag))
# ds-sample-vk: pass 1 colour shaders, pass 2 uv.vert + the three sampling programs
DSAMP_H := $(B)/dsamp/color.vert.h $(B)/dsamp/color.frag.h $(B)/dsamp/uv.vert.h $(B)/dsamp/ds-raw.frag.h \
           $(B)/dsamp/ds-cmp.frag.h $(B)/dsamp/ds-stencil.frag.h
$(eval $(call glsl,$(B)/dsamp/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/dsamp/color.frag.h,spv_color_frag,$(C)/color.frag))
$(eval $(call glsl,$(B)/dsamp/uv.vert.h,spv_uv_vert,$(C)/uv.vert))
$(eval $(call glsl,$(B)/dsamp/ds-raw.frag.h,spv_ds_raw_frag,depth/shaders/ds-raw.frag))
$(eval $(call glsl,$(B)/dsamp/ds-cmp.frag.h,spv_ds_cmp_frag,depth/shaders/ds-cmp.frag))
$(eval $(call glsl,$(B)/dsamp/ds-stencil.frag.h,spv_ds_stencil_frag,depth/shaders/ds-stencil.frag))
MSAA_H := $(B)/msaa/color.vert.h $(B)/msaa/color.frag.h $(B)/msaa/smask.frag.h
$(eval $(call glsl,$(B)/msaa/color.vert.h,spv_color_vert,$(C)/color.vert))
$(eval $(call glsl,$(B)/msaa/color.frag.h,spv_color_frag,$(C)/color.frag))
$(eval $(call glsl,$(B)/msaa/smask.frag.h,spv_smask_frag,msaa/shaders/msaa-smask.frag))

# xfb-test: <name>.h with the array <name with . and - as _>
XFB_SH := xfb-rec.vert xfb-attr.vert xfb-attr16.vert xfb-inst.vert xfb-dup.vert xfb-cover.vert xfb.frag
XFB_H := $(XFB_SH:%=$(B)/xfb/%.h)
$(foreach s,$(XFB_SH),$(eval $(call glsl,$(B)/xfb/$(s).h,$(subst -,_,$(subst .,_,$(s))),xfb/shaders/$(s))))

# tt-vk: <name>_<stage>.h with the array <name>_<stage>
TT_SH := tt-arr tt-arr-lod tt-arr-lod1 tt-cube tt-lod
TT_H := $(B)/tt/pass_vert.h $(B)/tt/solid_frag.h $(B)/tt/uv_vert.h $(B)/tt/tex_frag.h \
        $(foreach s,$(TT_SH),$(B)/tt/$(subst -,_,$(s))_frag.h)
$(eval $(call glsl,$(B)/tt/pass_vert.h,pass_vert,$(C)/pass.vert))
$(eval $(call glsl,$(B)/tt/solid_frag.h,solid_frag,$(C)/solid.frag))
$(eval $(call glsl,$(B)/tt/uv_vert.h,uv_vert,$(C)/uv.vert))
$(eval $(call glsl,$(B)/tt/tex_frag.h,tex_frag,$(C)/tex.frag))
$(foreach s,$(TT_SH),$(eval $(call glsl,$(B)/tt/$(subst -,_,$(s))_frag.h,$(subst -,_,$(s))_frag,textures/shaders/$(s).frag)))

headers: $(MRT_H) $(DEPTH_H) $(DSAMP_H) $(MSAA_H) $(XFB_H) $(TT_H)

$(B)/mrt-vk: mrt/mrt-vk.c $(MRT_H)
	$(CC) $(CFLAGS) -I$(B)/mrt -o $@ $< $(VK_LIBS)
$(B)/xfb-test: xfb/xfb-test.c $(XFB_H)
	$(CC) $(CFLAGS) -I$(B)/xfb -o $@ $< $(VK_LIBS)
$(B)/depth-vk: depth/depth-vk.c $(DEPTH_H)
	$(CC) $(CFLAGS) -I$(B)/depth -o $@ $< $(VK_LIBS)
$(B)/ds-sample-vk: depth/ds-sample-vk.c $(DSAMP_H)
	$(CC) $(CFLAGS) -I$(B)/dsamp -o $@ $< $(VK_LIBS)
$(B)/msaa-vk: msaa/msaa-vk.c $(MSAA_H)
	$(CC) $(CFLAGS) -I$(B)/msaa -o $@ $< $(VK_LIBS)
$(B)/tt-vk: textures/tt-vk.c $(TT_H)
	$(CC) $(CFLAGS) -I$(B)/tt -o $@ $< $(VK_LIBS)
$(B)/es3-smoke: gles/es3-smoke.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< $(GLES_LIBS)
$(B)/ds-ladder: gles/ds-ladder.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< $(GLES_LIBS)

clean:
	rm -rf $(B)

.PHONY: all vk gles headers clean
