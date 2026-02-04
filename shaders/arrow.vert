#version 330 core

layout(location = 0) in vec2 aPos;       // Arrow template vertex (line segment)
layout(location = 1) in vec2 aArrowPos;  // Grid position (instanced)
layout(location = 2) in vec2 aVelocity;  // Velocity vector (instanced)

uniform mat4 projection;
uniform float arrowScale;

out vec3 fragColor;

void main()
{
    // Calculate velocity magnitude for coloring
    float magnitude = length(aVelocity);

    // Normalize velocity direction
    vec2 direction = magnitude > 0.0001 ? normalize(aVelocity) : vec2(1.0, 0.0);

    // Rotate and scale the arrow template based on velocity
    float angle = atan(direction.y, direction.x);
    float cosA = cos(angle);
    float sinA = sin(angle);
    mat2 rotation = mat2(cosA, sinA, -sinA, cosA);

    // Scale arrow by fixed size (arrowScale controls overall size, not velocity-dependent)
    vec2 rotatedPos = rotation * (aPos * arrowScale);

    // Transform to grid position
    vec2 finalPos = aArrowPos + rotatedPos;

    gl_Position = projection * vec4(finalPos, 0.0, 1.0);

    // Color by magnitude - bright white to bright yellow/orange
    float normalizedMag = min(magnitude * 50.0, 1.0);  // Amplify for better visibility
    fragColor = vec3(1.0, 1.0 - normalizedMag * 0.5, 0.2);  // Bright white to orange
}
