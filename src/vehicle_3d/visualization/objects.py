''' classes to help contain objects to add to draw in 3D '''
import ctypes
from abc import abstractmethod
from PIL import Image


try:
    import OpenGL.GL as gl
    import OpenGL.GL.shaders
    GL_AVAILABLE = True
except ImportError:
    # workaround for Mac OS OpenGL import
    try:
        from ctypes import util
        orig_util_find_library = util.find_library
        def _new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res:
                return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = _new_util_find_library
        import OpenGL.GL as gl
        import OpenGL.GL.shaders
        GL_AVAILABLE = True
    except ImportError:
        # at this point OpenGL is probably not available
        GL_AVAILABLE = False


import numpy as np

from vehicle_3d.pytypes import Position, OrientationQuaternion, Renderable
from vehicle_3d.utils.load_utils import get_assets_file
from vehicle_3d.visualization import shaders
from vehicle_3d.visualization import glm

class UBOObject:
    '''
    class wrapper for a uniform buffer object
    used to avoid duplicate view/perspective transformation matrices across programs
    to avoid excessive update calls when either changes.
    '''
    _UBO: int
    binding_point: int

    def __init__(self):
        self.binding_point = 0
        self._UBO = gl.glGenBuffers(1)

        # populate the buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._UBO)
        I = np.array([np.eye(4), np.eye(4)], dtype = np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        I.nbytes,
                        I,
                        gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.binding_point, self._UBO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def update_camera_pose(self, mat: np.ndarray):
        '''
        update camera transform for the object
        expects a 4x4 numpy array of np.float32
        '''
        self._update_u_view(mat)

    def update_projection(self, mat: np.ndarray):
        '''
        update camera projection for the object
        expects a 4x4 numpy array of np.float32
        '''
        self._update_u_projection(mat)

    def _update_u_projection(self, mat: np.ndarray):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._UBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, mat.nbytes, mat.T)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _update_u_view(self, mat: np.ndarray):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._UBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, mat.nbytes, mat.nbytes, mat.T)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


class OpenGLObject(Renderable):
    '''
    class for creating and manipulating an object to draw with OpenGL,
    such as a car, the road surface, etc..
    '''
    _program: int = None
    _static_draw: bool

    def __init__(self, static_draw = True):
        self._static_draw = static_draw
        self._get_program()

        self._update_u_model(np.eye(4).astype(np.float32))

    @abstractmethod
    def _get_vertex(self):
        ''' get glsl vertex code'''

    @abstractmethod
    def _get_fragment(self):
        ''' get glsl fragment code'''

    def _get_program(self):
        compiled_vertex   = OpenGL.GL.shaders.compileShader(
            self._get_vertex(),
            gl.GL_VERTEX_SHADER)
        compiled_fragment = OpenGL.GL.shaders.compileShader(
            self._get_fragment(),
            gl.GL_FRAGMENT_SHADER)
        self._program = OpenGL.GL.shaders.compileProgram(
            compiled_vertex,
            compiled_fragment)

    def _link_ubo(self, ubo: UBOObject):
        mat_index = gl.glGetUniformBlockIndex(self._program, "Matrices")
        if mat_index >= 0:
            gl.glUniformBlockBinding(self._program, mat_index, ubo.binding_point)

    def update_pose(self, x: Position = None, q: OrientationQuaternion = None,
            mat: np.ndarray = None):
        ''' update the pose transform for the object'''
        if mat is None:
            u_model = np.eye(4, dtype=np.float32)
            if x is not None:
                u_model[:3,-1] = x.to_vec()
            if q is not None:
                u_model[:3,:3] = q.R()
            self._update_u_model(u_model.astype(np.float32))
        else:
            self._update_u_model(mat)

    def _update_u_model(self, u_model: np.ndarray):
        if not self._program:
            return
        gl.glUseProgram(self._program)
        transformLoc = gl.glGetUniformLocation(self._program, "u_model")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_model.T.tobytes())
        gl.glUseProgram(0)


class PrimitiveOpenGLObject(OpenGLObject):
    ''' a single rigid opengl object, additions for several helpers '''

    _VAO: int
    _VBO: int
    _no_indices: int = None

    def generate_standard_arrays(self):
        ''' helper to generate vbo and vao'''
        VAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(VAO)
        VBO = gl.glGenBuffers(1)

        self._VAO = VAO
        self._VBO = VBO


class PrimitiveUBOOpenGLObject(PrimitiveOpenGLObject):
    ''' object that uses a uniform buffer object '''


class DefaultPrimitiveOpenGLObject(PrimitiveUBOOpenGLObject):
    ''' rigid opengl object with standard vertex / shader pipeline '''
    _draw_mode: int = gl.GL_TRIANGLES

    def _get_vertex(self):
        return shaders.DEFAULT_VERTEX

    def _get_fragment(self):
        return shaders.DEFAULT_FRAGMENT

    def setup(self, vertices, indices = None):
        '''
        set up the object in terms of vertices and indices to draw
        this can be called any time after initialization to change the data for the object
        ie. updating an object for a planned trajectory
        '''
        if not self._program:
            return

        if indices is not None:
            vertices = vertices[indices]

        self._no_indices = vertices.shape[0]


        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._VBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        vertices.nbytes,
                        vertices,
                        gl.GL_STATIC_DRAW if self._static_draw else gl.GL_DYNAMIC_DRAW)

        position = gl.glGetAttribLocation(self._program, 'a_position')
        gl.glVertexAttribPointer(position, 3, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(position)

        normal = gl.glGetAttribLocation(self._program, 'a_normal')
        if normal > 0: # this attribute is not active for the simple fragment
            gl.glVertexAttribPointer(normal, 3, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(12))
            gl.glEnableVertexAttribArray(normal)

        color = gl.glGetAttribLocation(self._program, 'a_color')
        gl.glVertexAttribPointer(color, 4, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(24))
        gl.glEnableVertexAttribArray(color)

        gl.glUseProgram(0)

    def draw(self):
        ''' draw the object'''
        if not self._program:
            return

        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)
        gl.glDrawArrays(self._draw_mode, 0, self._no_indices)
        gl.glUseProgram(0)


class VertexObject(DefaultPrimitiveOpenGLObject):
    ''' opengl object with vertex data, potentially that changes '''
    simple_fragment: bool = False

    def __init__(self, ubo: UBOObject, V, I = None,
                 simple = False,
                 static_draw = True,
                 lines = False):
        self.simple_fragment = simple
        if lines:
            self._draw_mode = gl.GL_LINES
        else:
            self._draw_mode = gl.GL_TRIANGLES

        self.generate_standard_arrays()
        super().__init__(static_draw)
        self._link_ubo(ubo)
        self.setup(V, I)

    def _get_fragment(self):
        return shaders.DEFAULT_FRAGMENT if not self.simple_fragment else shaders.SIMPLE_FRAGMENT


class InstancedVertexObject(VertexObject):
    '''
    vertex object that is instanced
    for instance for drawing many copies
    '''
    _instance_buffer: int = None
    _no_instances: int

    def __init__(self, ubo: UBOObject, V, I = None,
                 simple = False,
                 static_draw = True,
                 lines = False):

        super().__init__(ubo, V, I, simple, static_draw, lines)
        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)

        self._instance_buffer = gl.glGenBuffers(1)
        gl.glUseProgram(0)

    def apply_instancing(self, R: np.ndarray):
        '''
        instance the object with positions and scales
        R: Nx4x4 numpy array
        '''
        R = R.astype(np.float32).transpose([0,2,1])
        self._no_instances = R.shape[0]

        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        R.nbytes,
                        R,
                        gl.GL_STATIC_DRAW if self._static_draw else gl.GL_DYNAMIC_DRAW)
        instance = gl.glGetAttribLocation(self._program, 'a_instance_matrix')
        for k in range(4):
            ptr = ctypes.c_void_p(16*k)
            gl.glVertexAttribPointer(instance+k, 4, gl.GL_FLOAT, gl.GL_FALSE, 64, ptr)
            gl.glEnableVertexAttribArray(instance+k)
            gl.glVertexAttribDivisor(instance+k, 1)

        gl.glUseProgram(0)

    def draw(self):
        if self._instance_buffer is None:
            return
        if not self._program:
            return

        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)
        gl.glDrawArraysInstanced(self._draw_mode, 0, self._no_indices, self._no_instances)
        gl.glUseProgram(0)

    def _get_vertex(self):
        return shaders.INSTANCED_VERTEX


class Skybox(PrimitiveOpenGLObject):
    ''' a skybox object to replicate surrounding scenery '''
    texture_id: int
    def __init__(self):
        self.generate_standard_arrays()
        super().__init__(static_draw=True)
        self.texture_id = self._load_cubemap()
        self.setup()

    def _get_vertex(self):
        return shaders.SKYBOX_VERTEX

    def _get_fragment(self):
        return shaders.SKYBOX_FRAGMENT

    def setup(self):
        '''
        set up the skybox
        '''
        if not self._program:
            return

        vertices = np.array([
            -1.0,  1.0, 1.0,
            -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0,
            1.0, -1.0, 1.0,
            1.0,  1.0, 1.0,
            -1.0,  1.0, 1.0,], dtype = np.float32)

        self.no_indices = 6

        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._VBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        vertices.nbytes,
                        vertices,
                        gl.GL_STATIC_DRAW)

        position = gl.glGetAttribLocation(self._program, 'a_position')
        gl.glVertexAttribPointer(position, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(position)

        gl.glUseProgram(0)

    def _update_u_model(self, u_model):
        u_model = glm.rotate(u_model, 90, 1, 0, 0)
        return super()._update_u_model(u_model.T)

    def _update_u_view(self, u_view: np.ndarray):
        if not self._program:
            return
        gl.glUseProgram(self._program)
        transformLoc = gl.glGetUniformLocation(self._program, "u_view")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_view.tobytes())

    def _update_u_projection(self, u_projection: np.ndarray):
        u_projection = np.linalg.inv(u_projection)
        if not self._program:
            return
        gl.glUseProgram(self._program)
        transformLoc = gl.glGetUniformLocation(self._program, "u_projection")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_projection.T.tobytes())
        gl.glUseProgram(0)

    def update_pose(self, x: Position = None, q: OrientationQuaternion = None,
            mat: np.ndarray = None):
        return

    def update_camera_pose(self, mat: np.ndarray):
        '''
        update camera transform for the object
        expects a 4x4 numpy array of np.float32
        '''
        self._update_u_view(mat)

    def update_projection(self, mat: np.ndarray):
        '''
        update camera projection for the object
        expects a 4x4 numpy array of np.float32
        '''
        self._update_u_projection(mat)

    def draw(self):
        ''' draw the object'''
        if not self._program:
            return
        gl.glUseProgram(self._program)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glBindVertexArray(self._VAO)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texture_id)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.no_indices)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glUseProgram(0)

    def _load_cubemap(self):
        files = [
            'right.jpg',
            'left.jpg',
            'top.jpg',
            'bottom.jpg',
            'front.jpg',
            'back.jpg'
        ]
        texture = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture)
        for k in range(6):
            filename = get_assets_file(files[k])
            img = Image.open(filename)
            img.load()
            data = np.asarray( img, dtype="int32" )
            width = data.shape[0]
            height = data.shape[1]

            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + k,
                0, gl.GL_RGB, width, height, 0, gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE, data)

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        return texture


class GroundPlane(DefaultPrimitiveOpenGLObject):
    ''' a ground plane to draw, currently by drawing a big square '''

    def __init__(self, ubo: UBOObject):
        self.generate_standard_arrays()
        super().__init__(static_draw=True)
        self._link_ubo(ubo)
        self._setup()

    def _get_vertex(self):
        return shaders.DEFAULT_VERTEX

    def _get_fragment(self):
        return shaders.DEFAULT_FRAGMENT

    def _setup(self):
        ''' set up the ground plane '''
        if not self._program:
            return

        vertices = np.zeros(6, dtype = shaders.vtype)
        vertices['a_position'] = np.array([
            [-1.0,  1.0, 0.0,],
            [-1.0, -1.0, 0.0,],
            [1.0, -1.0, 0.0,],
            [1.0, -1.0, 0.0,],
            [1.0,  1.0, 0.0,],
            [-1.0,  1.0, 0.0,]], dtype = np.float32) * 1000
        vertices['a_normal'] = np.array([0,0,1])
        vertices['a_color'] = np.array([0.4,0.4,0.4,1.0])
        super().setup(vertices)

    def draw(self):
        if not self._program:
            return

        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._no_indices)
        gl.glUseProgram(0)
        gl.glDepthFunc(gl.GL_LESS)
