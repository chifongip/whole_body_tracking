"""Insert default pose frames at the beginning and end of a motion CSV file.

Usage:
    python scripts/insert_default_pose.py \
        --input motions/run_back_stop.csv \
        --output motions/run_back_stop_with_default.csv \
        --num_frames 15
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

# Default joint positions for 23-DOF robot
DEFAULT_JOINT_POS_23DOF = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],  # left leg
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],  # right leg
    *[0],  # waist
    *[0.35, 0.18, 0.0, 0.87, 0.0],  # left arm
    *[0.35, -0.18, 0.0, 0.87, 0.0],  # right arm
]


def quat_xyzw_to_yaw_only(quat_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from a quaternion in xyzw format.

    Args:
        quat_xyzw: Quaternion in xyzw format (4,).

    Returns:
        Quaternion with only yaw component in xyzw format (4,).
    """
    # Convert to scipy Rotation (expects xyzw)
    r = Rotation.from_quat(quat_xyzw)
    # Convert to Euler angles (roll, pitch, yaw) in radians
    _, _, yaw = r.as_euler("xyz")
    # Create yaw-only rotation
    r_yaw = Rotation.from_euler("z", yaw)  # pyright: ignore[reportCallIssue]
    # Return as xyzw quaternion
    return r_yaw.as_quat()


def main():
    parser = argparse.ArgumentParser(description="Insert default pose frames at the beginning and end of a motion CSV file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input motion CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output motion CSV file.")
    parser.add_argument("--num_frames", type=int, default=15, help="Number of default pose frames to insert at each end.")
    args = parser.parse_args()

    # Load motion data
    motion = np.loadtxt(args.input, delimiter=",")
    print(f"Loaded motion: {motion.shape[0]} frames, {motion.shape[1]} columns")

    # Extract base position from first and last frames
    first_frame_pos = motion[0, :3].copy()
    last_frame_pos = motion[-1, :3].copy()

    # Extract yaw-only rotation from first and last frames
    first_frame_quat_xyzw = motion[0, 3:7]
    last_frame_quat_xyzw = motion[-1, 3:7]
    first_frame_yaw_quat = quat_xyzw_to_yaw_only(first_frame_quat_xyzw)
    last_frame_yaw_quat = quat_xyzw_to_yaw_only(last_frame_quat_xyzw)

    # Create default pose rows
    default_joint_pos = np.array(DEFAULT_JOINT_POS_23DOF, dtype=motion.dtype)

    start_row = np.concatenate([first_frame_pos, first_frame_yaw_quat, default_joint_pos])
    end_row = np.concatenate([last_frame_pos, last_frame_yaw_quat, default_joint_pos])

    # Duplicate rows
    start_frames = np.tile(start_row, (args.num_frames, 1))
    end_frames = np.tile(end_row, (args.num_frames, 1))

    # Concatenate: start + original + end
    output = np.concatenate([start_frames, motion, end_frames], axis=0)

    # Save output
    np.savetxt(args.output, output, delimiter=",")
    print(f"Saved output: {output.shape[0]} frames ({args.num_frames} start + {motion.shape[0]} original + {args.num_frames} end)")


if __name__ == "__main__":
    main()
