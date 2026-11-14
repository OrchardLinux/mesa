#version 300 es
precision highp float;precision highp int;
flat in vec4 v[8];out vec4 o;void main(){o=(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7])/8.;}