#version 330

smooth in vec4 theColor;

out vec4 outputColor;

void main()
{
	outputColor = vec4(theColor.r, theColor.r, theColor.r, 1.0);
}
