'''
loading and rendering gltf assets
'''
import ctypes
import io
import gc
from dataclasses import dataclass, field
from typing import List, Union, Callable, Dict
from PIL import Image

import numpy as np
from pygltflib import GLTF2, Accessor, Primitive, Material, Mesh, Node, BufferView

from vehicle_3d.pytypes import OrientationQuaternion, PythonMsg
from vehicle_3d.utils.load_utils import get_assets_file
from vehicle_3d.visualization import glm
from vehicle_3d.visualization.objects import gl, OpenGLObject, UBOObject

# primitive.mode table
PRIMITIVE_MODES = {
    0: gl.GL_POINTS,
    1: gl.GL_LINES,
    2: gl.GL_LINE_LOOP,
    3: gl.GL_LINE_STRIP,
    4: gl.GL_TRIANGLES,
    5: gl.GL_TRIANGLE_STRIP,
    6: gl.GL_TRIANGLE_FAN
}

# accessor.componentType table
COMPONENT_TYPES = {
    5120: np.byte,
    5121: np.ubyte,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32
}
COMPONENT_SIZES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4

}
GL_COMPONENT_TYPES = {
    5120: gl.GL_BYTE,
    5121: gl.GL_UNSIGNED_BYTE,
    5122: gl.GL_INT,
    5123: gl.GL_UNSIGNED_SHORT,
    5125: gl.GL_UNSIGNED_INT,
    5126: gl.GL_FLOAT
}

# accessor.type table
ACCESSOR_SHAPES = {
    'SCALAR': (-1),
    'VEC2': (-1,2),
    'VEC3': (-1,3),
    'VEC4': (-1,4),
    'MAT2': (-1,2,2),
    'MAT3': (-1,3,3),
    'MAT4': (-1,4,4)
}
ACCESSOR_SIZES = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16

}

def accessor_element_size(accessor: Accessor):
    ''' get accessor element size in bytes '''
    return int(COMPONENT_SIZES[accessor.componentType] * ACCESSOR_SIZES[accessor.type])

vtype = [('a_position', np.float32, 3),
         ('a_normal',   np.float32, 3),
         ('a_texcoord', np.float32, 2)]

GLTF_VERTEX = """
    #version 330 core

    layout (location = 0) in vec3 a_position;      // Vertex position
    layout (location = 1) in vec3 a_normal;        // Vertex normal
    layout (location = 2) in vec2 a_texcoord;      // Vertex texture coords

    out vec3 v_pos;
    out vec3 v_normal;
    out vec2 v_texcoord;
    out vec4 v_color;

    uniform vec4   base_color;
    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model;         // Model matrix

    void main()
    {
        v_pos    = vec3(u_model * vec4(a_position, 1.0));
        v_normal = vec3(u_model * vec4(a_normal, 0.0));
        v_color  = base_color;
        v_texcoord = a_texcoord;

        gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
    }
    """

INSTANCED_GLTF_VERTEX = """
    #version 330 core

    layout (location = 0) in vec3 a_position;      // Vertex position
    layout (location = 1) in vec3 a_normal;        // Vertex normal
    layout (location = 2) in vec2 a_texcoord;      // Vertex texture coords
    layout (location = 3) in mat4 a_instance_matrix;

    out vec3 v_pos;
    out vec3 v_normal;
    out vec2 v_texcoord;
    out vec4 v_color;

    uniform vec4   base_color;
    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model;         // Model matrix

    void main()
    {
        v_pos    = vec3(u_model * a_instance_matrix * vec4(a_position, 1.0));
        v_normal = vec3(u_model * a_instance_matrix * vec4(a_normal, 0.0));
        v_color  = base_color;
        v_texcoord = a_texcoord;

        gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
    }
    """

GLTF_FRAGMENT = """
    #version 330 core

    out vec4 FragColor;

    in vec3 v_pos;
    in vec3 v_normal;
    in vec2 v_texcoord;
    in vec4 v_color;

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };

    const vec3 light_dir = vec3(0,0,1);
    const vec3 light_color = vec3(1,1,1);

    uniform sampler2D color_texture;
    uniform float metallic_factor;
    uniform float roughness_factor;

    void main()
    {
        // ambient
        float ambient = 0.3;

        // diffuse
        vec3 normal = normalize(v_normal);
        float diffuse = roughness_factor * max(dot(normal, light_dir), 0.0);
        
        vec3 view_pos = - transpose(mat3(u_view)) * vec3(u_view[3][0], u_view[3][1], u_view[3][2]);

        // specular
        vec3 view_dir = normalize(view_pos - v_pos);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float specular = metallic_factor * pow(max(dot(view_dir, reflect_dir), 0.0), 32);

        vec3 result = specular * light_color + (ambient + diffuse) * vec3(v_color) * vec3(texture(color_texture, v_texcoord));
        FragColor = vec4(result, v_color.w);
    }
    """

GLTF_FRAGMENT_NOTEX = """
    #version 330 core

    out vec4 FragColor;

    in vec3 v_pos;
    in vec3 v_normal;
    in vec2 v_texcoord;
    in vec4 v_color;

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };

    const vec3 light_dir = vec3(0,0,1);
    const vec3 light_color = vec3(1,1,1);

    uniform sampler2D color_texture;
    uniform float metallic_factor;
    uniform float roughness_factor;
    uniform vec3 emissive_factor;

    void main()
    {
        // ambient
        float ambient = 0.3;

        // diffuse
        vec3 normal = normalize(v_normal);
        float diffuse = roughness_factor * max(dot(normal, light_dir), 0.0);
        
        vec3 view_pos = - transpose(mat3(u_view)) * vec3(u_view[3][0], u_view[3][1], u_view[3][2]);

        // specular
        vec3 view_dir = normalize(view_pos - v_pos);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float specular = metallic_factor * pow(max(dot(view_dir, reflect_dir), 0.0), 32);

        vec3 result = specular * light_color + (ambient + diffuse) * vec3(v_color);
        FragColor = vec4(result + emissive_factor, v_color.w);
    }
    """


def get_accessor_info(accessor_no: int, gltf: GLTF2):
    '''
    return some basic info for the accessor
    '''
    accessor : Accessor = gltf.accessors[accessor_no]
    buffer_view: BufferView = gltf.bufferViews[accessor.bufferView]

    buffer_no = buffer_view.buffer
    np_dtype = COMPONENT_TYPES[accessor.componentType]
    np_shape = ACCESSOR_SHAPES[accessor.type]

    return buffer_no, np_dtype, np_shape

def get_accessor_slice(accessor_no: int, gltf: GLTF2):
    '''
    get buffer data slice for an accessor
    '''
    accessor : Accessor = gltf.accessors[accessor_no]
    buffer_view: BufferView = gltf.bufferViews[accessor.bufferView]

    element_size = accessor_element_size(accessor)

    if buffer_view.byteStride is None or \
        buffer_view.byteStride == element_size:
        num_bytes = element_size * accessor.count
        data_slice = range(
            accessor.byteOffset + buffer_view.byteOffset,
            accessor.byteOffset + buffer_view.byteOffset + num_bytes
        )
    else:
        data_slice = (
            np.arange(0, accessor.count, buffer_view.byteStride)[:, np.newaxis] + \
                np.arange(element_size)
        ).reshape(-1)

    assert len(data_slice) == accessor.count * element_size

    return data_slice

def get_node_tf(node: Node):
    ''' get transform matrix for a node '''
    if node.matrix is not None:
        return np.array(node.matrix, dtype = np.float32).reshape(4,4).T

    T = np.eye(4, dtype = np.float32)
    if node.scale is not None:
        T = glm.scale(T, *node.scale)
    if node.rotation is not None:
        q = OrientationQuaternion()
        q.from_vec(node.rotation)
        T[:3,:3] = T[:3,:3] @ q.R()
    if node.translation is not None:
        M = np.eye(4)
        M[:3,-1] = np.array(node.translation)
        T = M @ T
    return T.astype(np.float32)


@dataclass
class TargetObjectSize(PythonMsg):
    ''' target dimesnions for a GLTF object'''
    root_tf: np.ndarray = field(default = None)
    fixed_aspect: bool = field(default = True)
    squish_z: bool = field(default = True)
    max_dims: Union[List[float], np.ndarray] = field(default = None)
    min_dims: Union[List[float], np.ndarray] = field(default = None)

    def __post_init__(self):
        if self.max_dims is None:
            self.max_dims = [2]*3
        if self.min_dims is None:
            self.min_dims = [-2, -2, 0]


class GLTFPrimitive(OpenGLObject):
    ''' renderer for a single primitive'''
    _VAO: int
    _VBO: int
    _EBO: int
    _texture: int
    _texture_available = False
    _use_texture = False

    _draw_mode: int
    _no_indices: int = None
    _index_type: int

    _base_color_factor: List[float]
    def __init__(self,
            ubo: UBOObject,
            primitive: Primitive,
            _get_accessor: Callable[[int], np.ndarray],
            _get_texture: Callable[[int], np.ndarray],
            _get_material: Callable[[int], Material],
            T = None):
        self.primitive = primitive
        self._get_accessor = _get_accessor
        self._get_texture = _get_texture
        self._get_material = _get_material
        self._create_vbo()
        super().__init__(static_draw=True)
        self._link_ubo(ubo)


        self._draw_mode = primitive.mode
        self.setup(primitive, T = T)

    def _get_vertex(self):
        return GLTF_VERTEX

    def _get_fragment(self):
        material = self._get_material(self.primitive.material)
        if material.pbrMetallicRoughness:
            if material.pbrMetallicRoughness.baseColorTexture:
                return GLTF_FRAGMENT
        return GLTF_FRAGMENT_NOTEX

    def _create_vbo(self):

        self._VAO = gl.glGenVertexArrays(1)
        self._VBO = gl.glGenBuffers(1)
        self._EBO = gl.glGenBuffers(1)
        self._texture = gl.glGenTextures(1)

    def _buffer_mode(self):
        return gl.GL_STATIC_DRAW if self._static_draw else gl.GL_DYNAMIC_DRAW

    def setup(self, primitive: Primitive, T = None):
        ''' set up buffers to draw the primitive '''
        # unpack vertex and index data
        verts = self._get_accessor(primitive.attributes.POSITION  )
        if primitive.attributes.NORMAL is not None:
            norms = self._get_accessor(primitive.attributes.NORMAL    )
        else:
            norms = np.zeros(verts.shape)
            norms[:,2] = 1
        inds  = self._get_accessor(primitive.indices              )

        V = np.zeros(verts.shape[0], dtype = vtype)
        V['a_position'] = verts
        V['a_normal'] = norms
        if primitive.attributes.TEXCOORD_0 is not None:
            texs  = self._get_accessor(primitive.attributes.TEXCOORD_0)
            V['a_texcoord'] = texs
            self._texture_available = True
        else:
            self._texture_available = False

        if T is not None:
            # transpose rotation due to shape of numpy array
            # R @ v --> (R @ V.T).T = V @ R.T
            V['a_position'] = V['a_position'] @ T[:3,:3].T + T[:3, -1]
            V['a_normal']   = V['a_normal'] @ T[:3,:3].T

        I = inds
        self._no_indices = len(I)
        self._index_type = [k for k, i in COMPONENT_TYPES.items() if i == inds.dtype][0]

        # unpack mateiral and texture data
        material: Material = self._get_material(primitive.material)

        self.opaque = material.alphaMode not in ('BLEND')

        if material.pbrMetallicRoughness is not None:
            base_color = material.pbrMetallicRoughness.baseColorFactor
        elif material.emissiveFactor is not None:
            base_color = material.emissiveFactor + [1.]
        else:
            base_color = [1., 1., 1., 1.]

        self._base_color_factor = [float(b) for b in base_color]

        # create buffers
        buffer_mode = self._buffer_mode()
        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._VBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, V.nbytes, V, buffer_mode)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._EBO)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, I.nbytes, I, buffer_mode)

        # bind buffers to attributes
        attrib_loc = gl.glGetAttribLocation(self._program, 'a_position')
        if attrib_loc >= 0:
            gl.glVertexAttribPointer(
                attrib_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(attrib_loc)

        attrib_loc = gl.glGetAttribLocation(self._program, 'a_normal')
        if attrib_loc >= 0:
            gl.glVertexAttribPointer(
                attrib_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(12))
            gl.glEnableVertexAttribArray(attrib_loc)

        attrib_loc = gl.glGetAttribLocation(self._program, 'a_texcoord')
        if attrib_loc >= 0:
            gl.glVertexAttribPointer(
                attrib_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(24))
            gl.glEnableVertexAttribArray(attrib_loc)

        self._use_texture = False
        if self._texture_available:
            if material.pbrMetallicRoughness is not None:
                if material.pbrMetallicRoughness.baseColorTexture:
                    texture_idx = material.pbrMetallicRoughness.baseColorTexture.index
                    if texture_idx is not None:
                        texture = \
                            self._get_texture(material.pbrMetallicRoughness.baseColorTexture.index)
                        # create texture
                        if len(texture.shape) == 2:
                            tex_format = gl.GL_RED
                        else:
                            if texture.shape[2] == 2:
                                tex_format = gl.GL_RG
                            elif texture.shape[2] == 3:
                                tex_format = gl.GL_RGB
                            else:
                                tex_format = gl.GL_RGBA

                        gl.glBindTexture(
                            gl.GL_TEXTURE_2D,
                            self._texture)
                        gl.glTexParameteri(
                            gl.GL_TEXTURE_2D,
                            gl.GL_TEXTURE_WRAP_S,
                            gl.GL_REPEAT)
                        gl.glTexParameteri(
                            gl.GL_TEXTURE_2D,
                            gl.GL_TEXTURE_WRAP_T,
                            gl.GL_REPEAT)
                        gl.glTexParameteri(
                            gl.GL_TEXTURE_2D,
                            gl.GL_TEXTURE_MIN_FILTER,
                            gl.GL_LINEAR_MIPMAP_LINEAR)
                        gl.glTexParameteri(
                            gl.GL_TEXTURE_2D,
                            gl.GL_TEXTURE_MAG_FILTER,
                            gl.GL_LINEAR)
                        gl.glTexImage2D(
                            gl.GL_TEXTURE_2D,
                            0,
                            tex_format,
                            texture.shape[1],
                            texture.shape[0],
                            0,
                            tex_format,
                            gl.GL_UNSIGNED_BYTE,
                            texture)
                        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                        self._use_texture = True

        # set uniforms
        uniform_loc = gl.glGetUniformLocation(self._program, "base_color")
        gl.glUniform4f(uniform_loc, *self._base_color_factor)

        uniform_loc = gl.glGetUniformLocation(self._program, "metallic_factor")
        if material.pbrMetallicRoughness is None \
                or material.pbrMetallicRoughness.metallicFactor is None:
            metallic_factor = 1.0
        else:
            metallic_factor = material.pbrMetallicRoughness.metallicFactor
        gl.glUniform1f(uniform_loc, float(metallic_factor))

        uniform_loc = gl.glGetUniformLocation(self._program, "roughness_factor")
        if material.pbrMetallicRoughness is None \
                or material.pbrMetallicRoughness.roughnessFactor is None:
            roughness_factor = 1.0
        else:
            roughness_factor = material.pbrMetallicRoughness.roughnessFactor
        gl.glUniform1f(uniform_loc, float(roughness_factor))

        uniform_loc = gl.glGetUniformLocation(self._program, "emissive_factor")
        emmisive_factor = material.emissiveFactor
        if emmisive_factor is None:
            emmisive_factor = [0.0, 0.0, 0.0]
        gl.glUniform3f(uniform_loc, *emmisive_factor)

        gl.glUseProgram(0)

    def draw(self):
        ''' draw the primitive '''
        if self._no_indices is None:
            raise RuntimeError('Primitive Renderer has not been initialized')

        gl.glUseProgram(self._program)
        if self._use_texture:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glBindVertexArray(self._VAO)

        gl.glDrawElements(self._draw_mode, self._no_indices, self._index_type, None)
        gl.glUseProgram(0)


class InstancedGLTFPrimitive(GLTFPrimitive):
    ''' primitive that supports instanced rendering '''
    _no_instances: int = None

    def _get_vertex(self):
        return INSTANCED_GLTF_VERTEX

    def apply_instancing(self, no_instances: int):
        '''
        apply instancing,
        assumes that current active buffer has been set to active buffer of instance data
        '''
        self._no_instances = no_instances
        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._VAO)
        instance = gl.glGetAttribLocation(self._program, 'a_instance_matrix')
        for k in range(4):
            ptr = ctypes.c_void_p(16*k)
            gl.glVertexAttribPointer(instance+k, 4, gl.GL_FLOAT, gl.GL_FALSE, 64, ptr)
            gl.glEnableVertexAttribArray(instance+k)
            gl.glVertexAttribDivisor(instance+k, 1)

    def draw(self):
        ''' draw instanced primitive '''
        if self._no_indices is None:
            raise RuntimeError('Primitive Renderer has not been initialized')
        if self._no_instances is None:
            raise RuntimeError('Instances have not been initialized')

        gl.glUseProgram(self._program)
        if self._use_texture:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glBindVertexArray(self._VAO)

        gl.glDrawElementsInstanced(self._draw_mode, self._no_indices, self._index_type, None,
                                   self._no_instances)
        gl.glUseProgram(0)


class GLTFObject:
    ''' gltf object for rendering '''
    gltf: GLTF2

    # raw data buffers, cleared after loading
    buffers: List[bytes]
    #transformed buffers with initial offsets / scales added

    # bounds of the object (after loading, before vehicle shape scaling)
    original_max_pos : np.ndarray
    original_min_pos : np.ndarray

    # target size of the object after scaling
    size: TargetObjectSize

    instanced: bool
    _instance_buffer: int = -1
    primitive_class: Union[GLTFPrimitive, InstancedGLTFPrimitive]
    primitives: List[Union[GLTFPrimitive, InstancedGLTFPrimitive]]

    _scene: int = 0

    def __init__(self,
            filename: str,
            ubo: UBOObject,
            size: TargetObjectSize = None,
            instanced: bool = False,
            cull_faces: bool = True,
            ignored_nodes: List[str] = None,
            matl_color_swaps: Dict[str, List[float]] = None):
        self.size = size
        self.cull_faces = cull_faces
        if ignored_nodes is None:
            ignored_nodes = []
        self.ignored_nodes = ignored_nodes
        self.instanced = instanced
        self.primitive_class = InstancedGLTFPrimitive if instanced else GLTFPrimitive

        self.gltf = GLTF2().load(filename)
        if matl_color_swaps is not None:
            for material in self.gltf.materials:
                if material.name in matl_color_swaps:
                    material.pbrMetallicRoughness.baseColorFactor = matl_color_swaps[material.name]
        self.ubo = ubo

        self._load_buffers()
        self._create_primitives()
        self._cleanup()

    def _load_buffers(self):
        self.buffers = []
        for buffer in self.gltf.buffers:
            self.buffers.append(
                self.gltf.get_data_from_buffer_uri(buffer.uri)
            )

    def _create_primitives(self):
        self.primitives = []

        self.original_max_pos = np.array([-np.inf, -np.inf, -np.inf], dtype = np.float32)
        self.original_min_pos = np.array([ np.inf,  np.inf,  np.inf], dtype = np.float32)

        def process_node(node: Node, root_tf = None, get_bbox = False):
            # carry through transforms
            tf = get_node_tf(node)
            if root_tf is not None:
                tf = root_tf @ tf

            if not get_bbox and node.name in self.ignored_nodes:
                return

            # if it has a mesh - translate vertices, rotate normals, and change min/max bounds
            if node.mesh is not None:
                mesh: Mesh = self.gltf.meshes[node.mesh]
                for primitive in mesh.primitives:
                    if get_bbox:
                        verts = self._get_accessor(primitive.attributes.POSITION)
                        verts = verts @ tf[:3,:3].T + tf[:3, -1]
                        self.original_max_pos = np.maximum(self.original_max_pos, verts.max(axis=0))
                        self.original_min_pos = np.minimum(self.original_min_pos, verts.min(axis=0))

                    else:
                        gltf_primitive = self.primitive_class(
                            self.ubo,
                            primitive,
                            self._get_accessor,
                            self._get_texture,
                            self._get_material,
                            tf
                        )
                        self.primitives.append(gltf_primitive)

            # loop over children
            for child in node.children:
                child_node = self.gltf.nodes[child]
                process_node(child_node, root_tf = tf, get_bbox = get_bbox)

        root_tf = np.eye(4)
        # if a target size has been specified, scale for this
        if self.size is not None:
            if self.size.root_tf is not None:
                root_tf = self.size.root_tf
            else:
                # first get bounds of the entire object
                for node_num in self.gltf.scenes[0].nodes:
                    node: Node = self.gltf.nodes[node_num]
                    process_node(node, root_tf, True)

                max_dims = np.array(self.size.max_dims)
                min_dims = np.array(self.size.min_dims)

                # scale the object, preserving the largest dimension if aspect ratio is fixed
                scale = (max_dims - min_dims)/(self.original_max_pos - self.original_min_pos)
                if self.size.fixed_aspect:
                    scale = scale.min()
                    scale = [scale] * 3
                root_tf = glm.scale(root_tf, *scale)

                # translate the object
                offset = (max_dims + min_dims) / 2 \
                    - (self.original_max_pos + self.original_min_pos) / 2 * scale
                if self.size.fixed_aspect and np.isfinite(self.size.min_dims[2]) \
                        and self.size.squish_z:
                    offset[2] = - self.original_min_pos[2] * scale[0] + self.size.min_dims[2]

                M = np.eye(4)
                M[:3,-1] = np.array(offset)
                root_tf = M @ root_tf

                self.size.root_tf = root_tf

        # now add all the nodes
        for node_num in self.gltf.scenes[0].nodes:
            node: Node = self.gltf.nodes[node_num]
            process_node(node, root_tf)

    def _cleanup(self):
        self.buffers = None
        self.tf_buffers = None
        self.textures = None
        self.gltf = None
        gc.collect()

    def _get_texture(self, texture_no):
        tex_info = self.gltf.textures[texture_no]
        image = self.gltf.images[tex_info.source]
        buffer_view = self.gltf.bufferViews[image.bufferView]
        buffer = self.buffers[buffer_view.buffer]
        tex_data = buffer[buffer_view.byteOffset: buffer_view.byteOffset + buffer_view.byteLength]
        im = Image.open(io.BytesIO(tex_data))
        data = np.array(im)
        if len(data.shape) == 2:
            data = np.array([data, data, data]).T
        return data

    def _get_accessor(self, accessor_no: int):
        ''' load accessor data from self.buffers '''
        if self.buffers is None:
            raise RuntimeWarning('Buffers have been uploaded already')

        buffer_no, np_dtype, np_shape = get_accessor_info(accessor_no, self.gltf)
        data_slice = get_accessor_slice(accessor_no, self.gltf)
        if isinstance(data_slice, range):
            byte_data = self.buffers[buffer_no][data_slice.start: data_slice.stop]
        else:
            byte_data = bytes(self.buffers[buffer_no][k] for k in data_slice)
        data = np.frombuffer(byte_data, np_dtype).reshape(np_shape)
        return data

    def _get_material(self, material_no: int):
        return self.gltf.materials[material_no]

    def draw(self):
        ''' draw the thing '''
        if not self.cull_faces:
            gl.glDisable(gl.GL_CULL_FACE)

        for primitive in self.primitives:
            if primitive.opaque:
                primitive.draw()

        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        for primitive in self.primitives:
            if not  primitive.opaque:
                primitive.draw()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        if not self.cull_faces:
            gl.glEnable(gl.GL_CULL_FACE)
        gl.glUseProgram(0)

    def update_pose(self, x = None, q = None, mat = None):
        ''' update the pose of the object '''
        for primitive in self.primitives:
            primitive.update_pose(x, q, mat)

    def apply_instancing(self, R: np.ndarray):
        ''' apply instancing to instanced primitives '''
        assert self.instanced is True

        R = R.astype(np.float32).transpose([0,2,1])
        no_instances = R.shape[0]

        if self._instance_buffer < 0:
            self._instance_buffer = gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        R.nbytes,
                        R,
                        gl.GL_DYNAMIC_DRAW)

        for primitive in self.primitives:
            primitive.apply_instancing(no_instances=no_instances)

def load_car(ubo: UBOObject,
             color: List = None,
             size: TargetObjectSize = None,
             instanced: bool = False):
    ''' load car model '''
    if size is None:
        alpha = 2.45 / (1.025 + 1.28)
        d =2.00* alpha
        size = TargetObjectSize(
            fixed_aspect = True,
            max_dims=[d,d,d],
            min_dims=[-d,-d,0])

    if color is not None:
        mat_swaps = {'Material.012':color}
    else:
        mat_swaps = {}

    return GLTFObject(
        get_assets_file('lambo.glb'),
        ubo,
        size,
        cull_faces = False,
        instanced = instanced,
        matl_color_swaps = mat_swaps
    )

def load_motorcycle(ubo: UBOObject,
             color: List = None,
             size: TargetObjectSize = None,
             instanced: bool = False):
    ''' load motorcycle model '''
    if size is None:
        d =1.2
        size = TargetObjectSize(
            fixed_aspect = True,
            max_dims=[d,d,d],
            min_dims=[-d,-d,-0.1])

    if color is not None:
        mat_swaps = {
            'Car_Paint_-_Red.001': color,
            'Car_Paint_-_Black': color
        }
    else:
        mat_swaps = {}
    return GLTFObject(
        get_assets_file('motorcycle.glb'),
        ubo,
        size,
        cull_faces = True,
        instanced = instanced,
        ignored_nodes= ['Object_24'],
        matl_color_swaps = mat_swaps,
    )

def load_split_motorcycle(ubo: UBOObject,
             color: List = None,
             size: TargetObjectSize = None):
    ''' load motorcycle model with front steering assembly separated'''
    if size is None:
        d =1.2
        size = TargetObjectSize(
            fixed_aspect = True,
            max_dims=[d,d,d],
            min_dims=[-d,-d,-0.1])

    if color is not None:
        mat_swaps = {
            'Car_Paint_-_Red.001': color,
            'Car_Paint_-_Black': color
        }
    else:
        mat_swaps = {}

    body = GLTFObject(
        get_assets_file('motorcycle.glb'),
        ubo,
        size,
        cull_faces = True,
        instanced = False,
        ignored_nodes= ['Object_24', 'Srad750.001_3', 'Srad750.006_6'],
        matl_color_swaps = mat_swaps,
    )
    fork = GLTFObject(
        get_assets_file('motorcycle.glb'),
        ubo,
        size,
        cull_faces = True,
        instanced = True,
        ignored_nodes= ['Srad750_1', 'Srad750.002_4', 'Srad750.004_5'],
        matl_color_swaps = mat_swaps,
    )
    return body, fork
