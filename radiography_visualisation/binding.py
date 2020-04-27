from OpenGL.GL import glGenVertexArrays, glGenBuffers, glGenTextures, glGenFramebuffers
from OpenGL.GL import glBindVertexArray, glBindBuffer, glBindTexture, glBindFramebuffer, glUseProgram
from OpenGL.GL import glGetUniformLocation
from OpenGL.GL import GL_ARRAY_BUFFER, GL_TEXTURE_2D, GL_FRAMEBUFFER

class VertexArrayObject:
    gl_identifier = None
 
    def __enter__(self):
        if self.gl_identifier is None:
            self.gl_identifier = glGenVertexArrays(1)
        glBindVertexArray(self.gl_identifier)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        glBindVertexArray(0)
        
class BufferObject:
    gl_identifier = None
    gl_bind_type = None
    
    def __init__(self, buf_type=GL_ARRAY_BUFFER):
        self.gl_bind_type = buf_type
        
    def __enter__(self):
        if self.gl_identifier is None:
            self.gl_identifier = glGenBuffers(1)
        glBindBuffer(self.gl_bind_type, self.gl_identifier)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        glBindBuffer(self.gl_bind_type, 0)
        
class Program:
    gl_identifier = None
    
    def __init__(self, prog_id):
        self.gl_identifier = prog_id
        
    def __enter__(self):
        glUseProgram(self.gl_identifier)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        glUseProgram(0)
        
    def getUniformLocation(self, unif_name):
        return glGetUniformLocation(self.gl_identifier, unif_name)
    
class Texture2D:
    gl_identifier = None
    
    def __enter__(self):
        if self.gl_identifier is None:
            self.gl_identifier = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gl_identifier)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        glBindTexture(GL_TEXTURE_2D, 0)

class FrameBufferObject:
    gl_identifier = None
        
    def __enter__(self):
        if self.gl_identifier is None:
            self.gl_identifier = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.gl_identifier)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
