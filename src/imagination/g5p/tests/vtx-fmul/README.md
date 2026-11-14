# Vertex FMUL uniform modifiers

Compile these six shaders with the real G5P standalone SPIR-V compiler.
On R32/4f8fa623860 the uniform/register abs shaders must refuse the missing modified-FMUL
form. After register staging all six compile. The literal abs reproduction also refuses before literal staging.
The three previously accepted
controls must retain identical USC. All instructions must pass native tuple
and register-bank auditing. No encoder is extended.

The linux-a9 projection's exact42 remaining fmul-modifier cases (14 GLES2,
28 GLES3) are the numerical CTS packet candidates. Their host typed-node
pipeline admission is not a pixel verdict.
