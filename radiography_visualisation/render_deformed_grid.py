import numpy as np
import matplotlib.pyplot as plt
import h5py

from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.GL import *
from shaders import loadShader, createProgram
from binding import VertexArrayObject, BufferObject, FrameBufferObject, Texture2D

class GridRender:
    width = 1000
    height = 800
    
    vertexData = None
    vertexDim = 2
    texDim = 2
    colorDim = 1
    nVertices = 0
    
    # Global variable to represent the compiled shader program, written in GLSL
    theProgram = None
    toneMapProgram = None
    
    # Field to represent the buffer that will hold the position vectors
    vertexBufferObject = BufferObject()
    # Field to hold vertex array objects
    vao = VertexArrayObject()
    
    # Framebuffer object
    hdrFBO = FrameBufferObject()
    quadVAO = VertexArrayObject()
    quadVBO = BufferObject()
    colorBuffer = Texture2D()
    
    # Global variables to store the location of the shader's uniform variables
    modelToCameraMatrixUnif = None
    cameraToClipMatrixUnif = None
    
    # Global display variables
    cameraToClipMatrix = np.identity(4, dtype='float32')
    
    # View offsets (user controlled)
    hoffset = 0
    woffset = 0
    scale_mult = 1
    
    # Shader names
    vert_shader = "screen_transform.vert"
    frag_shader = "red2gray.frag"
    
    # Tonemap shader names
    tonemap_vert_shader = "noop.vert"
    tonemap_frag_shader = "tonemap.frag"
    
    n_refresh = 0
    
    X0 = None
    Y0 = None
    A0 = None
    A = None
    A_pixel = 1
    
    # Probe energy in MeV
    E0 = 40
    energies = np.array([E0])
    runEnergyScan = False
    runAngleScan = False
    runTimeScan = False
    runProductScan = False
    # Proton mass
    Mp_MeV = 938.2720813
    mass_ratio = 1836.15267343
    c = 2.9979e8
    # Geometry factors
    magnification = 1
    f_g = 1
    # Travel time of light from source to plasma
    t_c = 0
    # Time offset of light hitting plasma
    t_0 = 0
    T = np.inf
    
    field_source = None
    
    quadVertices = np.array([
        # XY coords
        -1.0,  1.0,
        -1.0, -1.0,
         1.0,  1.0,
         1.0, -1.0,
         # UV coords
         0.0,  1.0,
         0.0,  0.0,
         1.0,  1.0,
         1.0,  0.0,
    ], dtype='float32')
    
    def set_xy0(self, x0, y0):
        '''
        Set un-deviated, un-magnified plasma coordinates.
        These can be specified in any length unit,
        but it must be consistent with z_s and z_i to be meaningful
        '''
        if x0.ndim == 1:
            self.X0, self.Y0 = np.meshgrid(x0, y0, indexing='ij')
            dx0 = np.diff(x0).mean()
            dy0 = np.diff(y0).mean()
            self.A0 = (dx0 * dy0 / 4) * np.ones(4 * (x0.size - 1) * (y0.size - 1))
        else:
            self.X0 = x0
            self.Y0 = y0
            _, _, self.A0 = self.calc_xy_rectgrid(self.X0, self.Y0)
        self.base_scale =  1 / np.maximum( np.abs(self.X0).max() / self.width, np.abs(self.Y0).max() / self.height )


        
    def set_geom(self, z_s, z_i):
        '''
        Set proton source and image distances.
        Calculates magnification and geometric focal length
        z_s: distance from proton source to interaction region
        z_i: distance from interaction regin to proton detector
        These should be specified in the same units of length as X0 and Y0
        '''
        self.magnification = 1 + z_i / z_s
        self.f_g = z_i / self.magnification
        # Travel time of light
        # Assuming microns and ps are our units of interest
        # Or equivalent e.g. mm, ns; m, us; 
        self.t_c = z_s / self.c * 1e6
        
    
    def set_dP(self, dPx, dPy):
        '''
        These should be 'dimensionless momenta', i.e. P = p / (m_e c) = (M_p / m_e) γβ
        (This is compatible with PIC units; dP = n x int(B, dl))
        '''
        self.dPx = dPx
        self.dPy = dPy
    
    def recalc_displacements(self):
        gamma = 1 + self.E0 / self.Mp_MeV
        P0 = self.mass_ratio * np.sqrt( gamma**2 - 1 )
        X = self.magnification * ( self.X0 + self.f_g * self.dPx / P0 )
        Y = self.magnification * ( self.Y0 + self.f_g * self.dPy / P0 )

        xx, yy, self.A = self.calc_xy_rectgrid(X, Y)
        self.A_pixel = 1 / (self.base_scale * self.scale_mult)**2
        fluence = self.A0 / np.maximum(self.A / self.magnification**2, self.A_pixel)
        self.set_vertex_data(xx, yy, fluence)
        self.init_displacements = True
        
    def update_pixel_size(self):
        self.A_pixel = 1 / (self.base_scale * self.scale_mult)**2
        fluence = self.A0 / np.maximum(self.A / self.magnification**2, self.A_pixel)
        fluence_verts = np.repeat(fluence, 3)
        self.vertexData[-(fluence_verts.size):] = fluence_verts
        self._updateVertexBuffers()
        
    def calc_xy_rectgrid(self, X, Y):
        ''' 
        Give me a pair of 2D arrays representing X and Y positions.
        I will return the x and y components of all vertices of a simple 
        symmetric triangulation of the grid and the area of each triangle
        '''
        # Corners of quads
        X_ul = X[:-1, :-1].flatten()
        X_ur = X[ 1:, :-1].flatten()
        X_dl = X[:-1,  1:].flatten()
        X_dr = X[ 1:,  1:].flatten()
        Y_ul = Y[:-1, :-1].flatten()
        Y_ur = Y[ 1:, :-1].flatten()
        Y_dl = Y[:-1,  1:].flatten()
        Y_dr = Y[ 1:,  1:].flatten()
        X_c = (X_ul + X_ur + X_dl + X_dr).flatten() / 4
        Y_c = (Y_ul + Y_ur + Y_dl + Y_dr).flatten() / 4
        xx = np.stack((X_c, X_ul, X_ur, X_c, X_ur, X_dr, X_c, X_dr, X_dl, X_c, X_dl, X_ul), axis=1).ravel()
        yy = np.stack((Y_c, Y_ul, Y_ur, Y_c, Y_ur, Y_dr, Y_c, Y_dr, Y_dl, Y_c, Y_dl, Y_ul), axis=1).ravel()

        Bx_ul = X_ul - X_c
        Bx_ur = X_ur - X_c
        Bx_dl = X_dl - X_c
        Bx_dr = X_dr - X_c
        By_ul = Y_ul - Y_c
        By_ur = Y_ur - Y_c
        By_dl = Y_dl - Y_c
        By_dr = Y_dr - Y_c
        
        A_u = np.abs(Bx_ul * By_ur - Bx_ur * By_ul) / 2
        A_r = np.abs(Bx_ur * By_dr - Bx_dr * By_ur) / 2
        A_d = np.abs(Bx_dr * By_dl - Bx_dl * By_dr) / 2
        A_l = np.abs(Bx_dl * By_ul - Bx_ul * By_dl) / 2
        
        A_tri = np.stack((A_u, A_r, A_d, A_l), axis=1).ravel()
        return xx, yy, A_tri
        
    def set_vertex_data(self, xx, yy, fluence):

        rr = np.stack((xx, yy), axis=1).ravel()
        
        fluence_verts = np.repeat(fluence, 3)
        
        self.nVertices = xx.size
        self.vertexData = np.concatenate((rr, fluence_verts)).astype('float32')
        
    def initialise_random_displacements(self, N_squares, sigma, N_zoom=1):
        from scipy.interpolate import interp2d
        x0 = np.linspace(-1, 1, N_squares + 1)
        self.set_xy0(x0, x0)
        x0_zoom = np.linspace(-1, 1, N_squares*N_zoom + 1)
        dx = 2 / N_squares
        dX = np.random.randn(*self.X0.shape) * sigma * dx
        dY = np.random.randn(*self.Y0.shape) * sigma * dx
        dX = interp2d(x0, x0, dX, kind='cubic')(x0_zoom, x0_zoom)
        dY = interp2d(x0, x0, dY, kind='cubic')(x0_zoom, x0_zoom)
        self.set_xy0(x0_zoom, x0_zoom)
        self.set_dP(dX, dY)
        self.recalc_displacements()

    # Initialize the OpenGL environment
    def _init_opengl(self):
        sizeOfFloat = 4 # all our arrays are dtype='float32'
        colorDataOffset = c_void_p(self.vertexDim * self.nVertices * sizeOfFloat)
        uvDataOffset = c_void_p(self.vertexDim * 4 * sizeOfFloat)

        self._initializeFramebuffer()
        self._initializePrograms()
        self._updateVertexBuffers()
        

        with self.quadVAO, self.quadVBO:
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, self.vertexDim, GL_FLOAT, False, 0, None);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, self.texDim, GL_FLOAT, False, 0, uvDataOffset);

        with self.vao, self.vertexBufferObject:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, self.vertexDim, GL_FLOAT, False, 0, None)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, self.colorDim, GL_FLOAT, False, 0, colorDataOffset)
        
        # We want both sides of faces to render        
        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        
        # No depth tests
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        glDepthFunc(GL_LEQUAL)
        
        # Unconventional blending: Add up all contributions to a pixel
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_ONE, GL_ONE)
       
    def run(self):
        if not self.init_displacements:
            self.recalc_displacements()
        self.n_refresh = 1
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA;
        glutInitDisplayMode(displayMode)
        
        glutInitContextVersion(3,3)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        
        glutInitWindowSize(self.width, self.height)
        
        glutInitWindowPosition(100, 100)
        
        window = glutCreateWindow("Proton Radiograph")
        
        self._init_opengl()
        
        glutDisplayFunc (lambda: self.display())
        glutReshapeFunc (lambda w, h: self.reshape(w, h))
        glutKeyboardFunc(lambda key, x, y: self.keyboard(key, x, y))
        glutSpecialFunc (lambda key, x, y: self.special(key, x, y))

        glutMainLoop();

    def _initializeFramebuffer(self):
        with self.colorBuffer:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, self.width, self.height, 0, GL_RED, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with self.hdrFBO:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.colorBuffer.gl_identifier, 0)

    # Set up the list of shaders, and call functions to compile them
    def _initializePrograms(self):
        # Load main shader
        shaderList = [
            loadShader(GL_VERTEX_SHADER, self.vert_shader),
            loadShader(GL_FRAGMENT_SHADER, self.frag_shader)
        ]
        
        self.theProgram = createProgram(shaderList)

        for shader in shaderList:
            glDeleteShader(shader)

        self.modelToCameraMatrixUnif = self.theProgram.getUniformLocation("modelToCameraMatrix")
        self.cameraToClipMatrixUnif = self.theProgram.getUniformLocation("cameraToClipMatrix")

        with self.theProgram:
            glUniformMatrix4fv(self.cameraToClipMatrixUnif, 1, False, self.cameraToClipMatrix.transpose())
            
        # Load tonemap shader
        shaderList = [
            loadShader(GL_VERTEX_SHADER, self.tonemap_vert_shader),
            loadShader(GL_FRAGMENT_SHADER, self.tonemap_frag_shader)
        ]

        self.toneMapProgram = createProgram(shaderList)

        for shader in shaderList:
            glDeleteShader(shader)            

    # Set up the vertex buffer that will store our vertex coordinates for OpenGL's access
    def _updateVertexBuffers(self):
        with self.vertexBufferObject:
            glBufferData(
                self.vertexBufferObject.gl_bind_type,
                self.vertexData,
                GL_STATIC_DRAW
            )
            
        with self.quadVBO:
            glBufferData(
                self.quadVBO.gl_bind_type,
                self.quadVertices,
                GL_STATIC_DRAW
            )
          
    def pull_image(self):
        with self.colorBuffer:
            fluence = glGetTexImagef(GL_TEXTURE_2D, 0, GL_RED)
        
        fluence.shape = fluence.T.shape
        
        return fluence

    def save_data(self, fname, fields):

        with h5py.File(fname, "w") as f:
            for key, val in fields.items():
                dset = f.create_dataset(key, val.shape, dtype=val.dtype)
                dset[...] = val
                
    def plot_data(self, fluence):
        plt.pcolormesh(fluence)
        plt.axis('image')
                    
    ## Callbacks
    def display(self):
        print('disp')
        if self.runProductScan:
            fluences = np.zeros((self.functions.shape[0], int(self.height), int(self.width)))
            for index, E in enumerate(self.energies):
                self.E0 = E
                self.set_time(self.get_time_for_energy(E))
                self.recalc_displacements()
                self._updateVertexBuffers()
                self.doDisplay()
                f = self.pull_image() * np.exp(-E/self.T)
                fluences += np.einsum('i,jk->ijk', self.functions[:,index], f)
            self.save_data(f'fluence_T_{self.T}MeV.h5', {'fluence': fluences})
            self.runProductScan = False

        if self.runEnergyScan:
            fluences = np.zeros((self.energies.size, int(self.height), int(self.width)))
            for index, E in enumerate(self.energies):
                self.E0 = E
                self.recalc_displacements()
                self._updateVertexBuffers()
                self.doDisplay()
                fluences[index, :, :] = self.pull_image()
                
            self.save_data('fluence.h5', {'energies': self.energies, 'fluence': fluences})
            self.runEnergyScan = False            
        if self.runAngleScan:
            angles = np.zeros((self.n_angles,))
            fluences = np.zeros((self.n_angles, int(self.height), int(self.width)))
            for index in np.arange(self.n_angles):
                self.get_displacements_right()
                self.recalc_displacements()
                self._updateVertexBuffers()
                self.doDisplay()
                fluences[index, :, :] = self.pull_image()
                angles[index] = self.field_source.theta
            self.save_data('fluence_rotate.h5', {'angles': angles, 'fluence': fluences})
            self.runAngleScan = False      
        if self.runTimeScan:
            fluences = np.zeros((self.field_source.times.size, int(self.height), int(self.width)))
            for index in np.arange(fluences.shape[0]):
                fluences[index, :, :] = self.pull_image()
                self.get_displacements_up()
                self.recalc_displacements()
                self._updateVertexBuffers()
                self.doDisplay()
                self.doDisplay()
                
            self.save_data('fluence_timeresolved.h5', {'times': self.field_source.times, 'fluence': fluences})
            self.runTimeScan = False            
        else:
            self.doDisplay()
        if self.n_refresh > 0:
            self.n_refresh -= 1
            glutPostRedisplay()

    def doDisplay(self):
        with self.hdrFBO:
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)
    
            with self.theProgram, self.vao:
                shiftMatrix = np.eye(4)
                shiftMatrix[0:3,3] = [self.woffset, self.hoffset, 0]
                scaleMatrix = np.eye(4)
                scaleMatrix[0,0] = self.base_scale * self.scale_mult / self.magnification
                scaleMatrix[1,1] = self.base_scale * self.scale_mult / self.magnification
                transformMatrix = scaleMatrix @ shiftMatrix
            
                glUniformMatrix4fv(self.modelToCameraMatrixUnif, 1, False, transformMatrix.transpose())
                glDrawArrays(GL_TRIANGLES, 0, self.nVertices)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
      
        with self.toneMapProgram, self.quadVAO, self.colorBuffer:
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glutSwapBuffers()

    
    def keyboard(self, key, x, y):
        if key == b'0':
            self.woffset = 0
            self.hoffset = 0
            self.scale_mult = 1
        elif key == b'e':
            self.scale_mult *= 2
        elif key == b'q':
            self.scale_mult /= 2
        elif key == b'w':
            self.hoffset -= 100 * self.magnification / (self.base_scale * self.scale_mult)
        elif key == b's':
            self.hoffset += 100 * self.magnification / (self.base_scale * self.scale_mult)
        elif key == b'a':
            self.woffset += 100 * self.magnification / (self.base_scale * self.scale_mult)
        elif key == b'd':
            self.woffset -= 100 * self.magnification / (self.base_scale * self.scale_mult)
        elif key == b'+':
            self.E0 *= 2
            print(f'Energy: {self.E0} MeV')
            self.recalc_displacements()
            self._updateVertexBuffers()
        elif key == b'-':
            self.E0 /= 2
            print(f'Energy: {self.E0} MeV')            
            self.recalc_displacements()
            self._updateVertexBuffers()
        elif key == b'g':
            self.runEnergyScan = True
        elif key == b'r':
            self.runAngleScan = True
        elif key == b't':
            self.runTimeScan = True
        elif key == b'p':
            fluence = self.pull_image()
            self.plot_data(fluence)
        elif key == b'f':
            fluence = self.pull_image()
            self.save_data(f'fluence_{self.E0}MeV.h5', {'fluence': fluence})
        elif key == b'h':
            self.runProductScan = True
        elif key == b'\x1B': # esc
            if self.field_source is not None:
                self.field_source.save()
            glutLeaveMainLoop()
            return
        else:
            return
        self.n_refresh = 1
        glutPostRedisplay()
        print('Posted redisplay')
    
    def special(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.get_displacements_left()
            self.recalc_displacements()
            self._updateVertexBuffers()
        elif key == GLUT_KEY_RIGHT:
            self.get_displacements_right()
            self.recalc_displacements()
            self._updateVertexBuffers()
        elif key == GLUT_KEY_UP:
            self.get_displacements_up()
            self.recalc_displacements()
            self._updateVertexBuffers()
        elif key == GLUT_KEY_DOWN:
            self.get_displacements_down()
            self.recalc_displacements()
            self._updateVertexBuffers()
        else:
            return
        self.n_refresh = 1

        glutPostRedisplay()
        print('Posted redisplay')
            
    def set_field_source(self, source):
        self.field_source = source
        self.n_angles = self.field_source.n_angles
        self.set_xy0(*source.get_axes())
        self.set_dP(*source.get_dP())
        self.recalc_displacements()
        
    def get_displacements_right(self):
        if self.field_source is not None:
            if self.field_source.right():
                self.set_dP(*self.field_source.get_dP())
                
    def get_displacements_left(self):
        if self.field_source is not None:
            if self.field_source.left():
                self.set_dP(*self.field_source.get_dP())
                
    def get_displacements_up(self):
        if self.field_source is not None:
            if self.field_source.up():
                self.set_dP(*self.field_source.get_dP())
                
    def get_displacements_down(self):
        if self.field_source is not None:
            if self.field_source.down():
                self.set_dP(*self.field_source.get_dP())
                
    def get_time_for_energy(self, E):
        gamma = 1 + E / self.Mp_MeV
        beta = np.sqrt( gamma**2 - 1 ) / gamma
        # How much slower than light are we getting there?
        dt = self.t_c * (1 / beta - 1)
        return self.t_0 + dt
        
    def set_time(self, t):
        if self.field_source is not None:
            print(f'Time {t} ps')
            self.set_dP(*self.field_source.dP_at_time(t))
                
    # Called whenever the window's size changes (including once when the program starts)
    def reshape(self, w, h):
        global cameraToClipMatrix
        global hoffset
        self.width = float(w)
        self.height = float(h)
        self.cameraToClipMatrix[0][0] = 1 / self.width
        self.cameraToClipMatrix[1][1] = 1 / self.height
        
        with self.theProgram:
            glUniformMatrix4fv(self.cameraToClipMatrixUnif, 1, False, self.cameraToClipMatrix.transpose())
            
        with self.colorBuffer:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, self.width, self.height, 0, GL_RED, GL_FLOAT, None)

        self.base_scale =  1 / np.maximum( np.abs(self.X0).max() / self.width, np.abs(self.Y0).max() / self.height )
        glViewport(0, 0, w, h)
        

if __name__ == '__main__':
    main()
    
def main():
    GR = GridRender()
    GR.set_geom(1E4, 1E5)
    GR.initialise_random_displacements(8,0.1,10)
    GR.run()
