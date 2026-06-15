"""Mirror motion data across the x-z plane (left-right symmetry).

Joint/body indices follow PhysX BFS (level-order) simulator ordering.
"""

import argparse
import numpy as np

# PhysX BFS (level-order) joint indices for G1 23-DOF
# URDF axes: pitch=y[0,1,0], roll=x[1,0,0], yaw=z[0,0,1]
# x-z mirror: M·R_axis(θ)·M^T = R_axis(θ) for y (pitch); = R_axis(-θ) for x,z (roll,yaw)
# Rule: preserve pitch, negate roll and yaw
JOINT_LEFT_RIGHT_PAIRS = [
    # (left_idx, right_idx, negate)
    (0, 1, False),   # hip_pitch      — pitch(y) → preserve
    (3, 4, True),    # hip_roll       — roll(x)  → negate
    (7, 8, True),    # hip_yaw        — yaw(z)   → negate
    (11, 12, False), # knee           — pitch(y) → preserve
    (15, 16, False), # ankle_pitch    — pitch(y) → preserve
    (19, 20, True),  # ankle_roll     — roll(x)  → negate
    (5, 6, False),   # shoulder_pitch — pitch(y) → preserve
    (9, 10, True),   # shoulder_roll  — roll(x)  → negate
    (13, 14, True),  # shoulder_yaw   — yaw(z)   → negate
    (17, 18, False), # elbow          — pitch(y) → preserve
    (21, 22, True),  # wrist_roll     — roll(x)  → negate
]

BODY_LEFT_RIGHT_PAIRS = [
    # Legs
    (1, 2),   # hip_pitch_link
    (4, 5),   # hip_roll_link
    (8, 9),   # hip_yaw_link
    (12, 13), # knee_link
    (16, 17), # ankle_pitch_link
    (20, 21), # ankle_roll_link
    # Arms
    (6, 7),   # shoulder_pitch_link
    (10, 11), # shoulder_roll_link
    (14, 15), # shoulder_yaw_link
    (18, 19), # elbow_link
    (22, 23), # wrist_roll_link
]

MIDLINE_BODIES = [0, 3]  # pelvis, torso_link


def mirror_npz(input_path: str, output_path: str) -> None:
    data = np.load(input_path)
    mirrored = {}

    # --- Joint positions & velocities ---
    joint_pos = data["joint_pos"].copy()
    joint_vel = data["joint_vel"].copy()

    for li, ri, negate in JOINT_LEFT_RIGHT_PAIRS:
        sign = -1.0 if negate else 1.0
        joint_pos[:, li], joint_pos[:, ri] = sign * data["joint_pos"][:, ri], sign * data["joint_pos"][:, li]
        joint_vel[:, li], joint_vel[:, ri] = sign * data["joint_vel"][:, ri], sign * data["joint_vel"][:, li]

    # waist_yaw (index 2 in BFS): negate (yaw around z-axis negated under x-z mirror)
    joint_pos[:, 2] = -data["joint_pos"][:, 2]
    joint_vel[:, 2] = -data["joint_vel"][:, 2]

    mirrored["joint_pos"] = joint_pos
    mirrored["joint_vel"] = joint_vel

    # --- Body positions & velocities (world frame) ---
    body_pos = data["body_pos_w"].copy()
    body_lin_vel = data["body_lin_vel_w"].copy()
    body_ang_vel = data["body_ang_vel_w"].copy()

    # Mirror: negate y for all bodies
    body_pos[:, :, 1] = -data["body_pos_w"][:, :, 1]
    body_lin_vel[:, :, 1] = -data["body_lin_vel_w"][:, :, 1]

    # Angular velocity: pseudovector, ω' = det(M)·M·ω = -M·ω = (-ωx, ωy, -ωz)
    body_ang_vel[:, :, 0] = -data["body_ang_vel_w"][:, :, 0]  # negate x
    body_ang_vel[:, :, 2] = -data["body_ang_vel_w"][:, :, 2]  # negate z

    # Swap left/right bodies
    for li, ri in BODY_LEFT_RIGHT_PAIRS:
        body_pos[:, li], body_pos[:, ri] = body_pos[:, ri].copy(), body_pos[:, li].copy()
        body_lin_vel[:, li], body_lin_vel[:, ri] = body_lin_vel[:, ri].copy(), body_lin_vel[:, li].copy()
        body_ang_vel[:, li], body_ang_vel[:, ri] = body_ang_vel[:, ri].copy(), body_ang_vel[:, li].copy()

    mirrored["body_pos_w"] = body_pos
    mirrored["body_lin_vel_w"] = body_lin_vel
    mirrored["body_ang_vel_w"] = body_ang_vel

    # --- Body quaternions (wxyz) ---
    body_quat = data["body_quat_w"].copy()
    # Mirror quaternion (wxyz): negate x and z, preserve w and y
    body_quat[:, :, 1] = -data["body_quat_w"][:, :, 1]  # negate x
    body_quat[:, :, 3] = -data["body_quat_w"][:, :, 3]  # negate z

    # Swap left/right bodies
    for li, ri in BODY_LEFT_RIGHT_PAIRS:
        body_quat[:, li], body_quat[:, ri] = body_quat[:, ri].copy(), body_quat[:, li].copy()

    mirrored["body_quat_w"] = body_quat

    # --- Metadata ---
    mirrored["fps"] = data["fps"]

    np.savez(output_path, **mirrored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror motion data (left-right symmetry)")
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("output", help="Output .npz file")
    args = parser.parse_args()
    mirror_npz(args.input, args.output)
