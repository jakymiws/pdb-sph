#version 400 core

uniform float inCol;

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, inCol, 1.0) ;
    //FragColor = texture(mainTexture, TexCoord);
};