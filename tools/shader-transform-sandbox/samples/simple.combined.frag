#version 450

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

vec4 t(sampler2D s, vec2 i) {
	return texture(s, i);
}

void main() {
  outColor = t(tex, inUV);
}
