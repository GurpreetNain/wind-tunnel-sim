import numpy as np

def quat2rot(quat_att):
    qw = quat_att[0]
    qx = quat_att[1]
    qy = quat_att[2]
    qz = quat_att[3]
    return np.array([[1 - 2*(qy**2 + qz**2) ,   2*(qx*qy - qw*qz)     ,   2*(qx*qz + qw*qy)  ],
                     [  2*(qx*qy + qw*qz)   , 1 - 2*(qx**2 + qz**2)   ,   2*(qy*qz - qw*qx)  ],
                     [  2*(qx*qz - qw*qy)   ,   2*(qy*qz + qw*qx)     , 1 - 2*(qx**2 + qy**2)]])

def rot2quat(R):
    # Ensure input is a numpy array
    R = np.asarray(R, dtype=np.float64)
    
    # Extract matrix elements
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    
    # Calculate matrix trace
    tr = m00 + m11 + m22
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
        
    return q/np.linalg.norm(q)

def quat_conjugate(q):
        """Returns the complex conjugate of a quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q, r):
    """Performs full Hamilton product quaternion multiplication (q x r)."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,  # FIX: Corrected y0*w1
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])