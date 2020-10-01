#version 400 core

uniform float inCol;

out vec4 FragColor;

void main()
{
    vec3 n;

    n.xy = gl_PointCoord.xy*vec2(2.0,-2.0) + vec2(-1.0,1.0);
    float m = dot(n.xy, n.xy);
    if (m > 1.0)
    {
        discard;
    }
    vec4 col = vec4(exp(-m*m)*vec3(0.53,0.80,0.98), 1.0);
    FragColor = col;
};