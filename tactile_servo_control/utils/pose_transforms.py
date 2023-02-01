import numpy as np
import pybullet as pb

POSE_UNITS = np.array([1e-3, 1e-3, 1e-3, np.pi/180, np.pi/180, np.pi/180])


def euler2quat(pose_e):
    """Converts an euler rotation pose to a quaternion rotation pose.
    """
    assert len(pose_e) == 6, "Invalid euler pose"
    pose_e *= POSE_UNITS
    rot_q = pb.getQuaternionFromEuler(pose_e[3:])
    pos_q = pose_e[:3]
    pose_q = [*pos_q, *rot_q]
    return pose_q


def quat2euler(pose_q):
    """Converts an euler rotation pose to a quaternion rotation pose.
    """
    assert len(pose_q) == 7, "Invalid quaternion pose"
    rot_e_rad = pb.getEulerFromQuaternion(pose_q[3:])
    pos_e = pose_q[:3]
    pose_e = [*pos_e, *rot_e_rad] / POSE_UNITS
    return pose_e


def transform(pose_a, frame_b_a):
    """Transforms a quaternion pose between reference frames.

    Transforms a pose in reference frame A to a pose in reference frame
    B (B is expressed relative to reference frame A).
    """
    inv_frame_b_a_pos, inv_frame_b_a_rot = pb.invertTransform(
        frame_b_a[:3], frame_b_a[3:],
    )
    pos_b, rot_b = pb.multiplyTransforms(
        inv_frame_b_a_pos, inv_frame_b_a_rot,
        pose_a[:3], pose_a[3:]
    )
    pose_b = [*pos_b, *rot_b]
    
    return pose_b


def transform_pose(pose_a, frame_b_a):
    """Transforms an Euler pose between reference frames.

    Transforms a pose in reference frame A to a pose in reference frame
    B (B is expressed relative to reference frame A).
    """
    pose_a_q = euler2quat(pose_a)
    frame_b_a_q = euler2quat(frame_b_a)
    pose_b_q = transform(pose_a_q, frame_b_a_q)
    pose_b = quat2euler(pose_b_q)

    return pose_b


def inv_transform(pose_b, frame_b_a):
    """Inverse transforms a quaternion pose between reference frames.

    Transforms a pose in reference frame B to a pose in reference frame
    A (B is expressed relative to A).
    """
    pos_a, rot_a = pb.multiplyTransforms(
        frame_b_a[:3], frame_b_a[3:],
        pose_b[:3], pose_b[3:]
    )

    return [*pos_a, *rot_a]


def inv_transform_pose(pose_b, frame_b_a):
    """Inverse transforms an Euler pose between reference frames.

    Transforms a pose in reference frame B to a pose in reference frame
    A (B is expressed relative to A).
    """
    pose_b_q = euler2quat(pose_b)
    frame_b_a_q = euler2quat(frame_b_a)
    pose_a_q = inv_transform(pose_b_q, frame_b_a_q)
    pose_a = quat2euler(pose_a_q)

    return pose_a
