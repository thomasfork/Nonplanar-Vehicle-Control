import numpy as np

vertex_default = """
        #version 330
        uniform mat4   u_model;         // Model matrix
        uniform mat4   u_view;          // View matrix
        uniform mat4   u_projection;    // Projection matrix
        attribute vec4 a_color;         // Vertex color
        attribute vec3 a_position;      // Vertex position
        attribute vec3 a_normal;        // Vertex normal
        varying vec4   v_color;         // Interpolated fragment color (out)
        varying vec3   v_normal;        // Interpolated normal (out)
        void main()
        {
            // Assign varying variables
            v_color    = a_color;    
            v_normal   = a_normal;
            
            // Final position
            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        }
        """
fragment_default = """
        #version 330
        uniform mat4      u_model;           // Model matrix
        uniform mat4      u_view;            // View matrix
        uniform mat4      u_projection;      // Projection matrix
        varying vec4      v_color;           // Interpolated fragment color (in)
        varying vec3      v_normal;          // Interpolated normal (in)
        const   vec3      light_normal = vec3(0,0,1); 
        
        void main()
        {
            float ambient = max(dot(v_normal, light_normal) * 0.4, 0.0) + 0.1;
            
            vec3 normal = vec3(u_model * vec4(v_normal, 0.0));
            float diffuse = dot(normal, light_normal);
            diffuse = max(min(diffuse,1.0), ambient);
            
            gl_FragColor = v_color * diffuse;
            gl_FragColor.w = v_color.w;  
        }
        """   

fragment_simple = """
        #version 330
        uniform mat4      u_model;           // Model matrix
        uniform mat4      u_view;            // View matrix
        uniform mat4      u_projection;      // Projection matrix
        varying vec4      v_color;           // Interpolated fragment color (in)
        varying vec3      v_normal;          // Interpolated normal (in)
        
        void main()
        {
            gl_FragColor = v_color;
                                  
        }
        """
 
vtype = [('a_position', np.float32, 3),
             ('a_normal',   np.float32, 3), ('a_color',    np.float32, 4)] 
