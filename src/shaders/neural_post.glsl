#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_output;

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    vec4 pixel_color = imageLoad(img_output, pixel_coords);

    // PRISM-GL Neural Bloom/Tone-mapping Simulation
    pixel_color.rgb = pixel_color.rgb / (pixel_color.rgb + vec3(1.0)); // Simple Tone-mapping
    pixel_color.rgb = pow(pixel_color.rgb, vec3(1.0/2.2)); // Gamma Correction

    imageStore(img_output, pixel_coords, pixel_color);
}