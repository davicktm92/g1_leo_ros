import threading

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


PACKAGE_SHARE_DIR = Path(
    get_package_share_directory("g1_leo_ros_description")
)

DEFAULT_MODEL_PATH = str(
    PACKAGE_SHARE_DIR
    / "robots"
    / "g1"
    / "g1_29dof.xml"
)

LEFT_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_JOINTS = {"left": LEFT_JOINTS, "right": RIGHT_JOINTS}
WRIST_BODIES = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
MOTOR_JOINTS = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint",
    3: "left_knee_joint", 4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint",
    9: "right_knee_joint", 10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_yaw_joint", 13: "waist_roll_joint", 14: "waist_pitch_joint",
    15: LEFT_JOINTS[0], 16: LEFT_JOINTS[1], 17: LEFT_JOINTS[2], 18: LEFT_JOINTS[3],
    19: LEFT_JOINTS[4], 20: LEFT_JOINTS[5], 21: LEFT_JOINTS[6],
    22: RIGHT_JOINTS[0], 23: RIGHT_JOINTS[1], 24: RIGHT_JOINTS[2], 25: RIGHT_JOINTS[3],
    26: RIGHT_JOINTS[4], 27: RIGHT_JOINTS[5], 28: RIGHT_JOINTS[6],
}


def euler_xyz_to_quat(euler):
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def matrix_to_euler_xyz(matrix):
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


class G1IKSolver:
    """MuJoCo FK/IK for the 14 arm joints on a 29-DOF G1."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, damping=1e-3, max_step_degrees=5.0):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.lock = threading.RLock()
        self.data = mujoco.MjData(self.model)
        self.qpos_home = self.data.qpos.copy()
        self.damping = damping * np.eye(6)
        self.max_step = np.deg2rad(max_step_degrees)
        self.jac_pos = np.zeros((3, self.model.nv))
        self.jac_rot = np.zeros((3, self.model.nv))
        self.jacobian = np.zeros((6, self.model.nv))
        self.qpos_addresses = {}
        self.velocity_addresses = {}
        self.joint_limits = {}
        self.body_ids = {}

        for arm, names in ARM_JOINTS.items():
            joint_ids = np.array([self.model.joint(name).id for name in names])
            self.qpos_addresses[arm] = np.array(
                [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids]
            )
            self.velocity_addresses[arm] = np.array(
                [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
            )
            self.joint_limits[arm] = self.model.jnt_range[joint_ids].copy()
            self.body_ids[arm] = self.model.body(WRIST_BODIES[arm]).id
        self.motor_qpos_addresses = {
            index: self.model.jnt_qposadr[self.model.joint(name).id]
            for index, name in MOTOR_JOINTS.items()
        }

    def qpos_from_motor_state(self, motor_state, base_qpos=None):
        if len(motor_state) < 29:
            raise ValueError(f"LowState has {len(motor_state)} motors; expected at least 29")
        with self.lock:
            qpos = self.data.qpos.copy() if base_qpos is None else np.array(base_qpos, copy=True)
            for motor_index, qpos_address in self.motor_qpos_addresses.items():
                qpos[qpos_address] = float(motor_state[motor_index].q)
        return qpos

    def forward_kinematics(self, qpos, arm):
        with self.lock:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            body_id = self.body_ids[arm]
            position = self.data.xpos[body_id].copy()
            rotation = self.data.xmat[body_id].reshape(3, 3)
            return position, matrix_to_euler_xyz(rotation)

    def solve(self, target_position, target_euler, initial_qpos, arm, max_iterations=10):
        if arm not in ARM_JOINTS:
            raise ValueError(f"Unknown arm: {arm}")

        with self.lock:
            qpos_addresses = self.qpos_addresses[arm]
            velocity_addresses = self.velocity_addresses[arm]
            joint_limits = self.joint_limits[arm]
            target_xyzw = euler_xyz_to_quat(target_euler)
            target_wxyz = target_xyzw[[3, 0, 1, 2]]
            self.data.qpos[:] = initial_qpos

            for _ in range(max_iterations):
                mujoco.mj_forward(self.model, self.data)
                body_id = self.body_ids[arm]
                error = np.zeros(6)
                error[:3] = target_position - self.data.xpos[body_id]

                body_quat = np.empty(4)
                body_quat_inverse = np.empty(4)
                error_quat = np.empty(4)
                mujoco.mju_mat2Quat(body_quat, self.data.xmat[body_id])
                mujoco.mju_negQuat(body_quat_inverse, body_quat)
                mujoco.mju_mulQuat(error_quat, target_wxyz, body_quat_inverse)
                mujoco.mju_quat2Vel(error[3:], error_quat, 1.0)
                if np.linalg.norm(error) < 1e-3:
                    break

                self.jac_pos.fill(0.0)
                self.jac_rot.fill(0.0)
                mujoco.mj_jacBody(
                    self.model, self.data, self.jac_pos, self.jac_rot, body_id
                )
                self.jacobian[:3, :] = self.jac_pos
                self.jacobian[3:, :] = self.jac_rot
                jacobian_arm = self.jacobian[:, velocity_addresses]
                normal_matrix = jacobian_arm @ jacobian_arm.T + self.damping
                delta = jacobian_arm.T @ np.linalg.solve(normal_matrix, error)
                largest_step = float(np.max(np.abs(delta)))
                if largest_step > self.max_step:
                    delta *= self.max_step / largest_step

                full_delta = np.zeros(self.model.nv)
                full_delta[velocity_addresses] = delta
                mujoco.mj_integratePos(self.model, self.data.qpos, full_delta, 1.0)
                self.data.qpos[qpos_addresses] = np.clip(
                    self.data.qpos[qpos_addresses],
                    joint_limits[:, 0],
                    joint_limits[:, 1],
                )

            return self.data.qpos.copy()

    def solve_cartesian_velocity(
        self, qpos, targets, linear_velocity, angular_velocity, selection, dt,
        max_cartesian_offset=0.25,
    ):
        if selection not in ("left", "right", "dual"):
            raise ValueError(f"Unknown arm selection: {selection}")

        active_arms = ("left", "right") if selection == "dual" else (selection,)
        qpos_solution = np.array(qpos, copy=True)
        for arm in active_arms:
            pose = targets[arm]
            linear = np.array(linear_velocity, dtype=float)
            angular = np.zeros(3) if selection == "dual" else np.array(angular_velocity, dtype=float)
            if selection == "dual":
                linear[1] *= -1.0 if arm == "left" else 1.0

            pose["position"] += linear * dt
            pose["position"] = np.clip(
                pose["position"],
                pose["origin"] - max_cartesian_offset,
                pose["origin"] + max_cartesian_offset,
            )
            pose["euler"] += angular * dt
            qpos_solution = self.solve(
                pose["position"], pose["euler"], qpos_solution, arm
            )
        return qpos_solution