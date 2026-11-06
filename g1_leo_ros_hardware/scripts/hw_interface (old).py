#!/usr/bin/env python3
"""Mode-driven Cartesian waypoint controller for Unitree G1 (29 DoF).

Requires the full-arm G1IKSolver.py supplied in this project. Mode topic:
std_msgs/Int32: 1=locomotion/release arm_sdk, 2=arm teleoperation.
Waypoints: trajectory_msgs/MultiDOFJointTrajectory on /arm_cartesian_waypoints.
"""
import math
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Int8, String
from trajectory_msgs.msg import MultiDOFJointTrajectory
import tf2_ros

from G1IKSolver import ARM_JOINTS, DEFAULT_MODEL_PATH, MOTOR_JOINTS, G1IKSolver

#esto para teleoperacion de locomocion
from geometry_msgs.msg import Twist


MODE_LOCOMOTION = 1
MODE_ARMS = 2
WEIGHT_INDEX = 29
CONTROLLED_INDICES = tuple(range(12, 29))  # waist + 7 joints per arm
LINK_TO_ARM = {"left_wrist_yaw_link": "left", "right_wrist_yaw_link": "right"}


def quat_norm(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if not np.all(np.isfinite(q)) or n < 1e-9:
        raise ValueError("Quaternion inválido o de norma cero")
    return q / n


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    x1, y1, z1, w1 = quat_norm(a)
    x2, y2, z2, w2 = quat_norm(b)
    return quat_norm([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    q = quat_norm(q)
    qv = q[:3]
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[3] * v)


def quat_to_euler_xyz(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat_norm(q)
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=float)


def euler_xyz_to_quat(e: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = map(float, e)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return quat_norm([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ])


def quat_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = float(np.clip(abs(np.dot(quat_norm(a), quat_norm(b))), 0.0, 1.0))
    return 2.0 * math.acos(dot)


class G1CartesianModeNode(Node):
    def __init__(self):
        super().__init__("g1_cartesian_mode_controller")
        defaults = {
            "mode_topic": "/mode",
            "waypoints_topic": "/arm_cartesian_waypoints",
            "status_topic": "/arm_controller/status",
            "joint_targets_topic": "/arm_joint_targets",
            "ik_frame_id": "pelvis",
            "enable_hardware": True,
            "domain_id": 0,
            "network_interface": "enp12s0",
            "model_path": DEFAULT_MODEL_PATH,
            "control_rate": 50.0,
            "transition_duration": 2.0,
            "mode_timeout": 0.75,
            "lowstate_timeout": 1.0,
            "max_waypoints": 50,
            "max_ik_iterations": 10,
            "position_tolerance": 0.03,
            "orientation_tolerance": 0.25,
            "max_joint_velocity": 1.0,
            "kp_active": 60.0,
            "kd_active": 1.5,
            "kp_hold": 40.0,
            "kd_hold": 2.0,
            "cmd_vel_topic": "/cmd_vel",
            "cmd_vel_timeout": 0.5,
            "cmd_vel_rate": 10.0,
            "max_vx": 0.4,
            "max_vy": 0.2,
            "max_wz": 0.5,
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        self.mode_topic = str(p("mode_topic"))
        self.waypoints_topic = str(p("waypoints_topic"))
        self.status_topic = str(p("status_topic"))
        self.targets_topic = str(p("joint_targets_topic"))
        self.ik_frame_id = str(p("ik_frame_id")).strip()
        self.enable_hardware = bool(p("enable_hardware"))
        self.domain_id = int(p("domain_id"))
        self.network_interface = str(p("network_interface")).strip()
        self.model_path = str(p("model_path"))
        self.rate = float(p("control_rate"))
        self.dt = 1.0 / self.rate
        self.transition_duration = float(p("transition_duration"))
        self.mode_timeout = float(p("mode_timeout"))
        self.lowstate_timeout = float(p("lowstate_timeout"))
        self.max_waypoints = int(p("max_waypoints"))
        self.max_ik_iterations = int(p("max_ik_iterations"))
        self.pos_tol = float(p("position_tolerance"))
        self.rot_tol = float(p("orientation_tolerance"))
        self.max_joint_velocity = float(p("max_joint_velocity"))
        self.kp_active, self.kd_active = float(p("kp_active")), float(p("kd_active"))
        self.kp_hold, self.kd_hold = float(p("kp_hold")), float(p("kd_hold"))
        self.cmd_vel_topic = str(p("cmd_vel_topic"))
        self.cmd_vel_timeout = float(p("cmd_vel_timeout"))
        self.cmd_vel_rate = float(p("cmd_vel_rate"))

        self.max_vx = float(p("max_vx"))
        self.max_vy = float(p("max_vy"))
        self.max_wz = float(p("max_wz"))


        if (self.rate <= 0 or self.transition_duration <= 0 or self.mode_timeout <= 0
                or self.lowstate_timeout <= 0):
            raise ValueError("control_rate, transition_duration y timeouts deben ser positivos")
        if set(ARM_JOINTS) != {"left", "right"} or any(len(ARM_JOINTS[a]) != 7 for a in ("left", "right")):
            raise RuntimeError("Se requiere G1IKSolver con ambos brazos de 7 DoF")
        if not set(range(29)).issubset(MOTOR_JOINTS):
            raise RuntimeError("MOTOR_JOINTS debe mapear los índices 0..28")

        self.solver = G1IKSolver(model_path=self.model_path)
        self.qadr = self.solver.motor_qpos_addresses
        if len(self.qadr) < 29:
            raise RuntimeError("G1IKSolver no contiene el mapeo de los 29 motores")
        self.joint_to_motor = {name: idx for idx, name in MOTOR_JOINTS.items()}

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.control_thread = None
        self.planning_thread = None
        self.plan_generation = 0
        self.requested_mode = None
        self.last_mode = None
        self.last_mode_time = None
        self.pending_activation = False
        self.state = "IDLE"
        self.status_text = "Esperando modo 2."
        self.weight = 0.0
        self.transition_from = 0.0
        self.transition_to = 0.0
        self.transition_start = 0.0
        self.measured_qpos = None
        self.measured_positions = None
        self.last_lowstate_time = None
        self.command_targets: Dict[int, float] = {}
        self.active_arms = {"left", "right"}
        self.trajectory = None
        self.last_write_error = 0.0
        self.low_cmd = self.crc = self.sdk_pub = self.sdk_sub = None
        self.cartesian_reference = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.loco_client = None

        self.last_cmd_vel = None
        self.last_cmd_vel_time = None

        self.loco_stop_sent = True
        self.stop_move_pending = False
        self.arm_activation_after_stop = False

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.targets_pub = self.create_publisher(JointState, self.targets_topic, 10)
        self.mode_sub = self.create_subscription(Int8, self.mode_topic, self.mode_callback, 10)
        self.waypoint_sub = self.create_subscription(
            MultiDOFJointTrajectory, self.waypoints_topic, self.waypoint_callback, 10
        )
        self.maintenance_timer = self.create_timer(0.05, self.maintenance_callback)

        self.cmd_vel_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10,)

        self.locomotion_timer = self.create_timer(1.0 / self.cmd_vel_rate, self.locomotion_timer_callback)

        if self.enable_hardware:
            self.init_unitree()
        else:
            qpos = self.solver.qpos_home.copy()
            self.measured_qpos = qpos
            self.measured_positions = {i: float(qpos[self.qadr[i]]) for i in range(29)}
            self.last_lowstate_time = time.monotonic()
            self.get_logger().warning("DRY RUN: DDS Unitree deshabilitado; no se enviarán comandos físicos.")
        self.control_thread = threading.Thread(target=self.control_loop, name="g1_arm_sdk_control", daemon=True)
        self.control_thread.start()
        self.get_logger().info(
            f"Nodo listo: {self.mode_topic} (1=locomoción, 2=brazos), "
            f"waypoints={self.waypoints_topic}, hardware={self.enable_hardware}, {self.rate:.0f} Hz"
        )

    def init_unitree(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        if self.network_interface:
            ChannelFactoryInitialize(self.domain_id, self.network_interface)
        else:
            ChannelFactoryInitialize(self.domain_id)
        #inicializador del Loco Client para teleoperacion y manipulacion
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(5.0)
        self.loco_client.Init()

        #esto es del brazo
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.sdk_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.sdk_pub.Init()
        self.sdk_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sdk_sub.Init(self.lowstate_callback, 10)
        self.set_status("DDS inicializado; no se publica hasta recibir modo 2.")

    def lowstate_callback(self, msg):
        if len(msg.motor_state) < 29:
            return
        try:
            qpos = self.solver.qpos_from_motor_state(msg.motor_state, base_qpos=self.solver.qpos_home)
            positions = {i: float(msg.motor_state[i].q) for i in range(29)}
        except Exception as exc:
            self.get_logger().error(f"Error leyendo LowState: {exc}")
            return
        with self.lock:
            self.measured_qpos = qpos
            self.measured_positions = positions
            self.last_lowstate_time = time.monotonic()

    def cmd_vel_callback(self, msg: Twist):
        now = time.monotonic()

        vx = float(np.clip(msg.linear.x, -self.max_vx, self.max_vx))
        vy = float(np.clip(msg.linear.y, -self.max_vy, self.max_vy))
        wz = float(np.clip(msg.angular.z, -self.max_wz, self.max_wz))

        with self.lock:
            # Solo aceptar velocidades en modo locomoción.
            if self.requested_mode != MODE_LOCOMOTION:
                return

            # No aceptar velocidades durante la liberación de los brazos.
            if self.state != "IDLE":
                return

            # Comprobar que el tópico de modo sigue actualizado.
            if (
                self.last_mode_time is None
                or now - self.last_mode_time > self.mode_timeout
            ):
                return

            self.last_cmd_vel = (vx, vy, wz)
            self.last_cmd_vel_time = now

    def locomotion_timer_callback(self):
        """Envía velocidades al G1 o solicita una parada cuando corresponde."""

        # En modo de prueba no se envían comandos físicos.
        if not self.enable_hardware or self.loco_client is None:
            return

        now = time.monotonic()

        with self.lock:
            # Verificar que el tópico de modo sigue publicando.
            mode_fresh = (
                self.last_mode_time is not None
                and now - self.last_mode_time <= self.mode_timeout
            )

            # Verificar que cmd_vel no ha caducado.
            cmd_fresh = (
                self.last_cmd_vel is not None
                and self.last_cmd_vel_time is not None
                and now - self.last_cmd_vel_time <= self.cmd_vel_timeout
            )

            # Una parada pendiente tiene prioridad sobre cualquier velocidad.
            force_stop = self.stop_move_pending

            can_move = (
                self.requested_mode == MODE_LOCOMOTION
                and self.state == "IDLE"
                and mode_fresh
                and cmd_fresh
                and not force_stop
            )

            if can_move:
                velocity = self.last_cmd_vel
                need_stop = False
            else:
                velocity = None

                # Detener si existe una solicitud pendiente o si previamente
                # habíamos enviado órdenes de movimiento.
                need_stop = force_stop or not self.loco_stop_sent

        # ---------------------------------------------------------
        # Caso 1: detener la locomoción
        # ---------------------------------------------------------
        if need_stop:
            try:
                self.loco_client.StopMove()

            except Exception as exc:
                # No limpiar la solicitud pendiente: se volverá a intentar
                # en la siguiente ejecución del temporizador.
                self.get_logger().error(
                    f"Error enviando StopMove: {exc}"
                )
                return

            with self.lock:
                self.loco_stop_sent = True
                self.stop_move_pending = False

                # Si entramos en modo brazos, activar la solicitud de
                # activación únicamente después de enviar StopMove.
                if (
                    self.arm_activation_after_stop
                    and self.requested_mode == MODE_ARMS
                ):
                    if self.measured_qpos is not None:
                        self.arm_activation_after_stop = False
                        self.pending_activation = True

                        self.status_text = (
                            "StopMove enviado; preparada la activación de brazos."
                        )
                    else:
                        self.status_text = (
                            "StopMove enviado; esperando LowState para activar brazos."
                        )

            return

        # ---------------------------------------------------------
        # Caso 2: enviar una orden de velocidad
        # ---------------------------------------------------------
        if velocity is not None:
            vx, vy, wz = velocity

            try:
                self.loco_client.Move(
                    vx,
                    vy,
                    wz,
                    
                )

            except Exception as exc:
                self.get_logger().error(
                    f"Error enviando Move: {exc}"
                )

                # Ante un error de comunicación, solicitar una parada
                # en el siguiente ciclo del temporizador.
                with self.lock:
                    self.stop_move_pending = True

                return

            with self.lock:
                self.loco_stop_sent = False
        

    def mode_callback(self, msg: Int8):
        mode = int(msg.data)
        now = time.monotonic()

        can_activate = False

        with self.lock:
            # Actualizar siempre la hora del último mensaje, incluso
            # cuando se repite el mismo modo.
            self.last_mode_time = now

            # Los mensajes repetidos a 10 Hz no reinician las transiciones.
            if mode == self.last_mode:
                return

            old_mode = self.last_mode
            self.last_mode = mode
            self.requested_mode = mode

            # Descartar las velocidades anteriores.
            # Al volver a modo 1 será necesario recibir un cmd_vel reciente.
            self.last_cmd_vel = None
            self.last_cmd_vel_time = None

            # -----------------------------------------------------
            # MODO 2: detener locomoción antes de activar los brazos
            # -----------------------------------------------------
            if mode == MODE_ARMS:

                if self.enable_hardware and self.loco_client is not None:
                    # La parada se ejecutará desde locomotion_timer_callback().
                    self.stop_move_pending = True
                    self.arm_activation_after_stop = True

                    # Evitar que maintenance_callback active los brazos antes
                    # de que se haya enviado StopMove.
                    self.pending_activation = False

                    self.status_text = (
                        f"Modo {old_mode} -> 2: deteniendo locomoción "
                        "antes de activar los brazos."
                    )

                    can_activate = False

                else:
                    # En modo de prueba no hay locomoción física que detener.
                    self.arm_activation_after_stop = False
                    self.pending_activation = True

                    self.status_text = (
                        f"Modo {old_mode} -> 2: solicitando activación de brazos."
                    )

                    can_activate = self.measured_qpos is not None

            # -----------------------------------------------------
            # MODO 1: locomoción y liberación de los brazos
            # -----------------------------------------------------
            else:
                self.arm_activation_after_stop = False
                self.pending_activation = False

                # Si se recibe un modo desconocido, detener la locomoción
                # como medida conservadora.
                if (
                    mode != MODE_LOCOMOTION
                    and self.enable_hardware
                    and self.loco_client is not None
                ):
                    self.stop_move_pending = True

                if self.state == "WAITING_FOR_STATE":
                    self.cancel_trajectory_locked()
                    self.state = "IDLE"
                    self.weight = 0.0

                    self.status_text = (
                        f"Modo {mode}: esperando locomoción; "
                        "no se activa arm_sdk."
                    )

                elif self.state in (
                    "ACTIVATING",
                    "ACTIVE",
                    "RELEASING",
                ):
                    self.start_release_locked(
                        now,
                        f"modo cambiado a {mode}",
                    )

                else:
                    self.cancel_trajectory_locked()
                    self.state = "IDLE"
                    self.weight = 0.0

                    self.status_text = (
                        f"Modo {mode}: IDLE; "
                        "no se publican comandos de brazos."
                    )

                can_activate = False

        # En modo de prueba, activar directamente sin detener hardware.
        if can_activate:
            self.begin_activation()

    def maintenance_callback(self):
        with self.lock:
            # Si StopMove ya se ha enviado, pero todavía no había LowState,
            # permitir la activación cuando llegue un estado articular válido.
            if (
                self.arm_activation_after_stop
                and not self.stop_move_pending
                and self.requested_mode == MODE_ARMS
                and self.measured_qpos is not None
            ):
                self.arm_activation_after_stop = False
                self.pending_activation = True

            pending = (
                self.pending_activation
                and self.requested_mode == MODE_ARMS
                and self.measured_qpos is not None
            )

            status_text = self.status_text
            targets = self.command_targets.copy()
        if pending:
            self.begin_activation()
        status = String()
        status.data = status_text
        self.status_pub.publish(status)
        if targets:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [MOTOR_JOINTS[i] for i in CONTROLLED_INDICES]
            msg.position = [float(targets[i]) for i in CONTROLLED_INDICES]
            self.targets_pub.publish(msg)

    def begin_activation(self):
        with self.lock:
            if self.requested_mode != MODE_ARMS or self.measured_qpos is None:
                return
            if (self.enable_hardware and
                    (self.last_lowstate_time is None or
                     time.monotonic() - self.last_lowstate_time > self.lowstate_timeout)):
                self.state = "WAITING_FOR_STATE"
                self.pending_activation = True
                self.status_text = "Modo 2 recibido; esperando un LowState reciente antes de activar arm_sdk."
                return
            qpos = self.measured_qpos.copy()
            measured = self.measured_positions.copy()
        try:
            refs = {}
            for arm in ("left", "right"):
                pos, euler = self.solver.forward_kinematics(qpos, arm)
                refs[arm] = {"position": np.asarray(pos).copy(), "euler": np.asarray(euler).copy()}
        except Exception as exc:
            self.set_status(f"Error inicializando FK: {exc}")
            return
        now = time.monotonic()
        with self.lock:
            if self.requested_mode != MODE_ARMS:
                return
            self.plan_generation += 1
            self.trajectory = None
            self.cartesian_reference = refs
            self.command_targets = {i: float(measured[i]) for i in CONTROLLED_INDICES}
            self.active_arms = {"left", "right"}
            self.transition_from = self.weight
            self.transition_to = 1.0
            self.transition_start = now
            self.state = "ACTIVATING"
            self.pending_activation = False
            self.status_text = "ACTIVATING: postura capturada; peso de arm_sdk subiendo a 1."

    def cancel_trajectory_locked(self):
        self.plan_generation += 1
        self.trajectory = None

    def start_release_locked(self, now, reason):
        self.advance_trajectory_locked(now)
        self.cancel_trajectory_locked()
        if self.weight <= 1e-4:
            self.weight = 0.0
            self.state = "IDLE"
            self.status_text = f"IDLE: peso arm_sdk=0 ({reason})."
            return
        self.transition_from = self.weight
        self.transition_to = 0.0
        self.transition_start = now
        self.state = "RELEASING"
        self.status_text = f"RELEASING: liberando arm_sdk gradualmente ({reason})."

    def set_status(self, text):
        with self.lock:
            self.status_text = text
        self.get_logger().info(text)

    def waypoint_callback(self, msg: MultiDOFJointTrajectory):
        arms = []
        for name in msg.joint_names:
            if name not in LINK_TO_ARM or LINK_TO_ARM[name] in arms:
                self.get_logger().error(f"Trayectoria rechazada: joint_names no válido: {name}")
                return
            arms.append(LINK_TO_ARM[name])
        if not arms or not msg.points or len(msg.points) > int(self.get_parameter("max_waypoints").value):
            self.get_logger().error("Trayectoria rechazada: faltan brazos/puntos o se supera max_waypoints.")
            return
        if not msg.header.frame_id:
            self.get_logger().error("Trayectoria rechazada: header.frame_id es obligatorio.")
            return
        parsed, previous_time = [], 0.0
        for pi, point in enumerate(msg.points):
            if len(point.transforms) != len(msg.joint_names):
                self.get_logger().error(f"Punto {pi}: transforms y joint_names tienen tamaños distintos.")
                return
            t = float(point.time_from_start.sec) + 1e-9 * float(point.time_from_start.nanosec)
            if not math.isfinite(t) or t <= previous_time:
                self.get_logger().error("time_from_start debe ser positivo y estrictamente creciente.")
                return
            poses = []
            try:
                for tr in point.transforms:
                    pos = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=float)
                    quat = quat_norm([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])
                    if not np.all(np.isfinite(pos)):
                        raise ValueError("posición no finita")
                    poses.append((pos, quat))
            except ValueError as exc:
                self.get_logger().error(f"Trayectoria rechazada: {exc}")
                return
            parsed.append((t, poses))
            previous_time = t

        now = time.monotonic()
        with self.lock:
            if self.requested_mode != MODE_ARMS or self.state not in ("ACTIVATING", "ACTIVE"):
                self.get_logger().warning("Waypoint ignorado: el modo 2 no está activo ni activándose.")
                return
            if self.planning_thread is not None and self.planning_thread.is_alive():
                self.get_logger().warning("Waypoint ignorado: ya hay una planificación IK en curso.")
                return
            self.advance_trajectory_locked(now)
            self.trajectory = None
            self.plan_generation += 1
            generation = self.plan_generation
            if self.measured_qpos is None or not self.command_targets:
                self.get_logger().warning("Waypoint ignorado: no hay estado articular inicial.")
                return
            initial_qpos = self.measured_qpos.copy()
            start_targets = self.command_targets.copy()
            for i in CONTROLLED_INDICES:
                initial_qpos[self.qadr[i]] = start_targets[i]

        self.set_status(f"Planificando {len(parsed)} waypoint(s) para {', '.join(arms)}.")
        self.planning_thread = threading.Thread(
            target=self.plan_worker,
            args=(generation, msg.header.frame_id, arms, parsed, initial_qpos, start_targets),
            name="g1_cartesian_ik_planner", daemon=True,
        )
        self.planning_thread.start()

    def plan_worker(self, generation, source_frame, arms, parsed, qpos, start_targets):
        try:
            planned, prev_t, prev_targets = [], 0.0, start_targets.copy()
            for point_time, point_poses in parsed:
                with self.lock:
                    if generation != self.plan_generation or self.requested_mode != MODE_ARMS:
                        return
                transformed = {}
                for arm, (pos, quat) in zip(arms, point_poses):
                    transformed[arm] = self.transform_pose(source_frame, pos, quat)
                for arm in arms:
                    pos, quat = transformed[arm]
                    euler = quat_to_euler_xyz(quat)
                    #comentado para que no haga el checkeo de la solucion de IK, ya que a veces no encuentra solucion y da error
                    # qpos = self.solver.solve(pos, euler, qpos, arm, max_iterations=self.max_ik_iterations)
                    # actual_pos, actual_euler = self.solver.forward_kinematics(qpos, arm)

                    #añadido para debugeo
                    # Posición inicial calculada por MuJoCo
                    initial_pos, initial_euler = self.solver.forward_kinematics(
                        qpos, arm
                    )

                    qpos_before = qpos.copy()

                    # Resolver IK
                    qpos = self.solver.solve(
                        pos,
                        euler,
                        qpos,
                        arm,
                        max_iterations=self.max_ik_iterations
                    )

                    # Posición obtenida después de resolver IK
                    actual_pos, actual_euler = self.solver.forward_kinematics(
                        qpos, arm
                    )

                    self.get_logger().info(
                        f"\nIK {arm}"
                        f"\n  Objetivo: {np.round(pos, 3)}"
                        f"\n  Inicial:  {np.round(initial_pos, 3)}"
                        f"\n  Final:    {np.round(actual_pos, 3)}"
                        f"\n  Cambio articular máximo: "
                        f"{np.max(np.abs(qpos - qpos_before)):.4f} rad"
                    )



                    ep = float(np.linalg.norm(pos - actual_pos))
                    er = quat_distance(quat, euler_xyz_to_quat(actual_euler))
                    if ep > self.pos_tol:
                        raise ValueError(f"IK {arm}: error posición {ep:.3f} m > {self.pos_tol:.3f} m")
                    if er > self.rot_tol:
                        raise ValueError(f"IK {arm}: error orientación {er:.3f} rad > {self.rot_tol:.3f} rad")
                targets = {i: float(qpos[self.qadr[i]]) for i in CONTROLLED_INDICES}
                segment_dt = point_time - prev_t
                vmax = max(abs(targets[i] - prev_targets[i]) / segment_dt for i in CONTROLLED_INDICES)
                if vmax > self.max_joint_velocity:
                    raise ValueError(
                        f"Se requieren {vmax:.2f} rad/s, por encima del límite "
                        f"{self.max_joint_velocity:.2f}; aumenta el tiempo del waypoint."
                    )
                planned.append((point_time, targets))
                prev_t, prev_targets = point_time, targets
            with self.lock:
                if generation != self.plan_generation or self.requested_mode != MODE_ARMS:
                    return
                self.trajectory = {"start_time": None, "start_targets": start_targets,
                                   "points": planned, "arms": set(arms)}
                self.active_arms = set(arms)
                self.status_text = f"Trayectoria validada: {len(planned)} waypoint(s), brazos={arms}."
            self.get_logger().info("Trayectoria IK validada; se ejecutará al completar la activación.")
        except Exception as exc:
            with self.lock:
                if generation == self.plan_generation:
                    self.trajectory = None
                    self.status_text = f"Trayectoria rechazada: {exc}"
            self.get_logger().error(f"Trayectoria cartesiana rechazada: {exc}")

    def transform_pose(self, source_frame, position, quaternion):
        if source_frame == self.ik_frame_id:
            return position.copy(), quaternion.copy()
        tf = self.tf_buffer.lookup_transform(
            self.ik_frame_id, source_frame, Time(), timeout=Duration(seconds=0.25)
        )
        tr, rot = tf.transform.translation, tf.transform.rotation
        tq = quat_norm([rot.x, rot.y, rot.z, rot.w])
        t = np.array([tr.x, tr.y, tr.z], dtype=float)
        return quat_rotate(tq, position) + t, quat_mul(tq, quaternion)

    def advance_trajectory_locked(self, now):
        tr = self.trajectory
        if tr is None or self.state != "ACTIVE":
            return
        if tr["start_time"] is None:
            tr["start_time"] = now
        elapsed = max(0.0, now - tr["start_time"])
        prev_t, prev = 0.0, tr["start_targets"]
        for t, target in tr["points"]:
            if elapsed <= t:
                alpha = float(np.clip((elapsed - prev_t) / (t - prev_t), 0.0, 1.0))
                self.command_targets = {i: prev[i] + alpha*(target[i]-prev[i]) for i in CONTROLLED_INDICES}
                return
            prev_t, prev = t, target
        self.command_targets = dict(tr["points"][-1][1])

    def control_loop(self):
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            next_tick += self.dt
            now = time.monotonic()
            publish = False
            with self.lock:
                if (self.enable_hardware and self.state in ("WAITING_FOR_STATE", "ACTIVATING", "ACTIVE", "RELEASING")
                    and self.last_mode_time is not None and now - self.last_mode_time > self.mode_timeout):
                    self.last_mode_time = None
                    self.last_mode = None
                    self.requested_mode = None
                    self.pending_activation = False
                    self.start_release_locked(now, "timeout del tópico de modo")
                if (self.enable_hardware and self.state in ("ACTIVATING", "ACTIVE")
                    and self.last_lowstate_time is not None
                    and now - self.last_lowstate_time > self.lowstate_timeout):
                    self.last_lowstate_time = None
                    self.start_release_locked(now, "timeout de rt/lowstate; requiere cambiar modo para reactivar")
                if self.state == "ACTIVE":
                    self.advance_trajectory_locked(now)
                old_state = self.state
                if old_state in ("ACTIVATING", "RELEASING"):
                    alpha = float(np.clip((now-self.transition_start)/self.transition_duration, 0.0, 1.0))
                    self.weight = self.transition_from + alpha*(self.transition_to-self.transition_from)
                    if alpha >= 1.0 and old_state == "ACTIVATING":
                        self.weight, self.state = 1.0, "ACTIVE"
                        self.status_text = "ACTIVE: arm_sdk habilitado; esperando/ejecutando waypoints."
                    elif alpha >= 1.0 and old_state == "RELEASING":
                        self.weight, self.state, self.trajectory = 0.0, "IDLE", None
                        self.active_arms = {"left", "right"}
                        self.status_text = "IDLE: peso arm_sdk=0; locomoción puede recuperar el control."
                publish = self.enable_hardware and old_state in ("ACTIVATING", "ACTIVE", "RELEASING")
                targets = self.command_targets.copy() if publish else {}
                active_arms = set(self.active_arms)
                weight = float(self.weight)
                if not self.enable_hardware and self.state in ("ACTIVATING", "ACTIVE", "RELEASING"):
                    if self.measured_positions is None:
                        self.measured_positions = {}
                    for i in CONTROLLED_INDICES:
                        self.measured_positions[i] = self.command_targets[i]
                    qpos = self.solver.qpos_home.copy()
                    for i in range(29):
                        if i in self.measured_positions:
                            qpos[self.qadr[i]] = self.measured_positions[i]
                    self.measured_qpos = qpos
            if publish and targets:
                try:
                    self.write_low_command(targets, active_arms, weight)
                except Exception as exc:
                    if now - self.last_write_error > 1.0:
                        self.last_write_error = now
                        self.get_logger().error(f"Error publicando rt/arm_sdk: {exc}")
            delay = next_tick - time.monotonic()
            if delay > 0:
                self.stop_event.wait(delay)
            else:
                next_tick = time.monotonic()

    def write_low_command(self, targets, active_arms, weight):
        if self.low_cmd is None or self.sdk_pub is None:
            return
        for i in CONTROLLED_INDICES:
            name = MOTOR_JOINTS[i]
            arm = "left" if name.startswith("left_") else "right" if name.startswith("right_") else None
            active = arm in active_arms if arm else False
            motor = self.low_cmd.motor_cmd[i]
            motor.q, motor.dq, motor.tau = float(targets[i]), 0.0, 0.0
            motor.kp = self.kp_active if active else self.kp_hold
            motor.kd = self.kd_active if active else self.kd_hold
        self.low_cmd.motor_cmd[WEIGHT_INDEX].q = float(np.clip(weight, 0.0, 1.0))
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.sdk_pub.Write(self.low_cmd)

    def shutdown_hardware(self):
        if self.stop_event.is_set():
            return

        #Detener la locomoción antes de cerrar DDS
        if self.enable_hardware and self.loco_client is not None:
            try:
                self.loco_client.StopMove()
                self.get_logger().info(
                    "StopMove enviado: deteniendo la locomoción."
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Error al detener la locomoción: {exc}"
                )
        # apagar la transicion de los brazos
        if self.enable_hardware:
            with self.lock:
                now = time.monotonic()
                if self.state in ("ACTIVATING", "ACTIVE", "RELEASING"):
                    self.requested_mode = MODE_LOCOMOTION
                    self.last_mode = MODE_LOCOMOTION
                    self.last_mode_time = now
                    self.start_release_locked(now, "cierre del nodo")
                elif self.state == "WAITING_FOR_STATE":
                    self.state = "IDLE"
                    self.pending_activation = False
            deadline = time.monotonic() + self.transition_duration + 0.5
            while time.monotonic() < deadline:
                with self.lock:
                    if self.state == "IDLE" and self.weight <= 1e-4:
                        break
                time.sleep(0.02)
        self.stop_event.set()
        if self.control_thread is not None:
            self.control_thread.join(timeout=1.0)
        for handle in (self.sdk_sub, self.sdk_pub):
            if handle is not None:
                try:
                    handle.Close()
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = G1CartesianModeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_hardware()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
