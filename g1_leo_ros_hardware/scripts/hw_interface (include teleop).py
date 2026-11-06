#!/usr/bin/env python3
"""Mode-driven Cartesian waypoint controller for Unitree G1 (29 DoF).

Requires G1IKSolver.py from this project. Topics:
  /mode (std_msgs/Int8): 1=locomocion, 2=IK brazos, 3=HOLDING+locomocion.
  /arm_cartesian_waypoints (trajectory_msgs/MultiDOFJointTrajectory).

Integracion inspirada en el G1_29_ArmController oficial de xr_teleoperate:
- Captura las 29 posiciones antes de la rampa de arm_sdk.
- Los objetivos dinamicos por IK se aplican exclusivamente a los brazos 15..28.
- La cintura 12..14 conserva su pose inicial en LowCmd (no queda a cero).
- mode_pr=0; mode_machine copiado desde LowState; rt/arm_sdk a 250 Hz.
- El modo interno Regular Motion Mode debe seleccionarse externamente.

Perfil opcional xr_reference_pd=True replica los PD del codigo de Unitree
(300/3.0 en motores fuertes; 80/3.0 en debiles; 40/1.5 munecas).
Por defecto mantiene ganancias conservadoras del programa que funcionaba;
no debe activarse el perfil de altas ganancias en hardware sin validarlo.

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
MODE_HOLD_WALK = 3
WEIGHT_INDEX = 29
CONTROLLED_INDICES = tuple(range(12, 29))  # estado y referencia IK
ARM_INDICES = tuple(range(15, 29))       # solo estos reciben objetivos nuevos, como xr_teleoperate
WAIST_INDICES = (12, 13, 14)           # se capturan y se mantienen en el LowCmd
XR_WEAK_INDICES = frozenset((4, 10, 15, 16, 17, 18, 22, 23, 24, 25))
XR_WRIST_INDICES = frozenset((19, 20, 21, 26, 27, 28))
LINK_TO_ARM = {"left_wrist_yaw_link": "left", "right_wrist_yaw_link": "right"}
ARM_MOTORS = {"left": tuple(range(15, 22)), "right": tuple(range(22, 29))}


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


def wrap_angle(a):
    """Lleva ángulos a (-pi, pi]; admite escalares y arrays."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


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
            # "network_interface": "enx00e04c360213", # PEDRO
            "network_interface": "enx00e04c360213", #DAVID
            "model_path": DEFAULT_MODEL_PATH,
            "control_rate": 250.0,  # xr_teleoperate usa 250 Hz
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
            "kp_hold": 80.0,
            "kd_hold": 2.0,
            # El ejemplo oficial usa ganancias de cuerpo muy altas.
            # Solo habilitarlas tras validar en simulacion y con soporte fisico.
            "xr_reference_pd": False,
            # El modo 3 exige que el operador haya confirmado el modo REGULAR
            # adecuado en el robot; ROS /mode=3 NO configura el FSM de Unitree.
            "require_regular_motion_confirmation": True,
            "regular_motion_confirmed": True,
            "cmd_vel_topic": "/cmd_vel",
            "cmd_vel_timeout": 0.5,
            "cmd_vel_rate": 10.0,
            "arm_velocity_topic": "/arm_cartesian_velocities",
            "arm_velocity_timeout": 0.3,      # s sin mensaje → velocidad 0
            "arm_velocity_rate": 50.0,        # Hz del bucle de IK por velocidad
            "max_arm_linear_speed": 0.15,     # m/s, saturación de seguridad
            "max_cartesian_offset": 0.25,     # m, caja de trabajo alrededor de la pose inicial
            "cartesian_windup_limit": 0.03,   # m, anti-acumulación del objetivo
            # Diagnóstico no intrusivo: desactívalo tras localizar el fallo.
            "debug_logs": True,
            "debug_period": 1.0,
            "max_vx": 0.4,
            "max_vy": 0.2,
            "max_wz": 0.5,
            # Lower limits in mode 3: walking while arm_sdk holds the upper body.
            "mode3_max_vx": 0.5,
            "mode3_max_vy": 0.5,
            "mode3_max_wz": 0.5,

            "max_arm_angular_speed": 1.0,      # rad/s, saturación angular de muñeca
            "max_angular_offset": 0.8,         # rad, giro máximo por eje respecto a la orientación inicial
            "orientation_windup_limit": 0.15,  # rad, anti-acumulación de orientación


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
        self.xr_reference_pd = bool(p("xr_reference_pd"))
        self.require_regular_motion_confirmation = bool(p("require_regular_motion_confirmation"))
        self.regular_motion_confirmed = bool(p("regular_motion_confirmed"))
        self.mode3_motion_permitted = (
            self.regular_motion_confirmed or not self.require_regular_motion_confirmation
        )
        self.cmd_vel_topic = str(p("cmd_vel_topic"))
        self.cmd_vel_timeout = float(p("cmd_vel_timeout"))
        self.cmd_vel_rate = float(p("cmd_vel_rate"))
        self.debug_logs = bool(p("debug_logs"))
        self.debug_period = float(p("debug_period"))
        self.velocity_topic = str(p("arm_velocity_topic"))
        self.arm_velocity_timeout = float(p("arm_velocity_timeout"))
        self.arm_velocity_rate = float(p("arm_velocity_rate"))
        self.max_arm_linear_speed = float(p("max_arm_linear_speed"))
        self.max_cartesian_offset = float(p("max_cartesian_offset"))
        self.windup_limit = float(p("cartesian_windup_limit"))

        


        self.max_vx = float(p("max_vx"))
        self.max_vy = float(p("max_vy"))
        self.max_wz = float(p("max_wz"))
        self.mode3_max_vx = float(p("mode3_max_vx"))
        self.mode3_max_vy = float(p("mode3_max_vy"))
        self.mode3_max_wz = float(p("mode3_max_wz"))

        self.max_arm_angular_speed = float(p("max_arm_angular_speed"))
        self.max_angular_offset = float(p("max_angular_offset"))
        self.orientation_windup_limit = float(p("orientation_windup_limit"))


        if (self.rate <= 0 or self.transition_duration <= 0 or self.mode_timeout <= 0
                or self.lowstate_timeout <= 0):
            raise ValueError("control_rate, transition_duration y timeouts deben ser positivos")
        if self.arm_velocity_rate <= 0 or self.arm_velocity_timeout <= 0:
            raise ValueError("arm_velocity_rate y arm_velocity_timeout deben ser positivos")    
        if self.debug_period <= 0.0:
            raise ValueError("debug_period debe ser positivo")
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
        # Protect LowCmd against simultaneous initialisation/write from ROS/DDS threads.
        self.sdk_cmd_lock = threading.Lock()
        self.lowcmd_initialized = False
        self.lowstate_mode_machine = None
        self.lowstate_mode_pr = None
        self.lowcmd_init_mode_machine = None
        self.lowcmd_init_mode_pr = None
        self.cartesian_reference = {}
        self.arm_velocity = {"left": np.zeros(3), "right": np.zeros(3)}

        self.arm_angular = {"left": np.zeros(3), "right": np.zeros(3)}

        self.arm_velocity_time = {"left": None, "right": None}
        self.velocity_targets = {}      # arm -> {"position","euler","origin"}; solo vive durante un movimiento
        self.last_velocity_tick = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.loco_client = None

        self.last_cmd_vel = None
        self.last_cmd_vel_time = None

        self.loco_stop_sent = True
        self.stop_move_pending = False
        self.arm_activation_after_stop = False
        self.arm_resume_after_stop = False   # 3 -> 2, keep arm_sdk at weight 1
        self.arm_release_after_stop = False  # 3 -> 1, first stop walking

        # Solo métricas/logs. No alteran estados, consignas ni llamadas al SDK.
        self._dbg_last = {}
        self._dbg_cmd_received = 0
        self._dbg_cmd_accepted = 0
        self._dbg_cmd_rejected = 0
        self._dbg_mode_received = 0
        self._dbg_move_count = 0
        self._dbg_stop_count = 0
        self._dbg_arm_writes = 0
        self._dbg_last_cmd_received_at = None
        self._dbg_last_move_ms = None
        self._dbg_last_stop_ms = None
        self._dbg_last_control_state = None

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.targets_pub = self.create_publisher(JointState, self.targets_topic, 10)
        self.mode_sub = self.create_subscription(Int8, self.mode_topic, self.mode_callback, 10)
        self.waypoint_sub = self.create_subscription(
            MultiDOFJointTrajectory, self.waypoints_topic, self.waypoint_callback, 10
        )
        self.maintenance_timer = self.create_timer(0.05, self.maintenance_callback)

        self.cmd_vel_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10,)

        self.locomotion_timer = self.create_timer(1.0 / self.cmd_vel_rate, self.locomotion_timer_callback)
        self.create_subscription(
            MultiDOFJointTrajectory, self.velocity_topic, self.arm_velocity_callback, 10)
        self.arm_velocity_timer = self.create_timer(
            1.0 / self.arm_velocity_rate, self.arm_velocity_timer_callback)
        if self.debug_logs:
            self.debug_timer = self.create_timer(self.debug_period, self.debug_status_callback)

        if self.enable_hardware:
            self.init_unitree()
        else:
            qpos = self.solver.qpos_home.copy()
            self.measured_qpos = qpos
            self.measured_positions = {i: float(qpos[self.qadr[i]]) for i in range(29)}
            self.last_lowstate_time = time.monotonic()
            # The dry run has no DDS mode_machine: no LowCmd is ever published.
            self.get_logger().warning("DRY RUN: DDS Unitree deshabilitado; no se enviarán comandos físicos.")
        self.control_thread = threading.Thread(target=self.control_loop, name="g1_arm_sdk_control", daemon=True)
        self.control_thread.start()
        self.get_logger().info(
            f"Nodo listo: {self.mode_topic} (1=locomoción, 2=brazos, 3=HOLDING+locomoción), "
            f"waypoints={self.waypoints_topic}, hardware={self.enable_hardware}, {self.rate:.0f} Hz, "
            f"xr_reference_pd={self.xr_reference_pd}, mode3_confirmed={self.mode3_motion_permitted}"
        )
        if self.enable_hardware and not self.mode3_motion_permitted:
            self.get_logger().warning(
                "[XR:REGULAR_MODE] /mode=3 mantendrá los brazos, pero NO enviará velocidades. "
                "Tras comprobar con el mando el Regular Motion Mode compatible con el firmware, "
                "reinicia con -p regular_motion_confirmed:=true."
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
        """Capture joint positions AND the real G1 mode fields from rt/lowstate."""
        if len(msg.motor_state) < 29:
            return
        try:
            qpos = self.solver.qpos_from_motor_state(
                msg.motor_state, base_qpos=self.solver.qpos_home)
            positions = {i: float(msg.motor_state[i].q) for i in range(29)}
            if not np.all(np.isfinite(list(positions.values()))):
                raise ValueError("LowState contiene una posición articular no finita")
            # Both fields are defined by Unitree's HG LowState IDL.
            mode_machine = int(msg.mode_machine)
            mode_pr = int(msg.mode_pr)
        except Exception as exc:
            self.get_logger().error(f"Error leyendo LowState: {exc}")
            return
        with self.lock:
            old_machine = self.lowstate_mode_machine
            old_pr = self.lowstate_mode_pr
            self.measured_qpos = qpos
            self.measured_positions = positions
            self.lowstate_mode_machine = mode_machine
            self.lowstate_mode_pr = mode_pr
            self.last_lowstate_time = time.monotonic()
            was_active = self.state in ("ACTIVATING", "ACTIVE", "HOLDING")
        if old_machine is not None and old_machine != mode_machine:
            self.get_logger().warning(
                f"[LOWCMD:MODE_MACHINE_CAMBIO] LowState {old_machine} -> "
                f"{mode_machine}; arm_sdk_activo={was_active}. "
                "Revisa el modo de movimiento del G1.")
        if old_pr is not None and old_pr != mode_pr:
            self.get_logger().warning(
                f"[LOWCMD:MODE_PR_CAMBIO] LowState {old_pr} -> {mode_pr}.")

    def xr_pd_for_motor(self, i):
        """Ganancias del perfil del controlador oficial G1_29_ArmController."""
        if i in XR_WRIST_INDICES:
            return 40.0, 1.5
        if i in XR_WEAK_INDICES:
            return 80.0, 3.0
        return 300.0, 3.0

    def initialize_lowcmd_from_lowstate(self, measured, mode_machine, mode_pr):
        """Inicializa un LowCmd completo al estilo xr_teleoperate.

        No hay acciones sobre la locomocion/FSM. Para evitar repetir el fallo
        observado al excluir 12..14, cada motor recibe q real y la cintura
        mantiene q de referencia con PD incluso cuando solo 15..28 se actualizan.

        El perfil xr_reference_pd replica los valores oficiales y modifica
        significativamente las ganancias: por defecto se conserva el perfil
        conservador del programa anterior.
        """
        if self.low_cmd is None:
            raise RuntimeError("No hay LowCmd de hardware para inicializar")
        if mode_machine is None:
            raise RuntimeError("Sin mode_machine desde LowState")
        if not all(i in measured and math.isfinite(float(measured[i])) for i in range(29)):
            raise ValueError("No hay 29 posiciones articulares medidas validas")
        if len(self.low_cmd.motor_cmd) <= WEIGHT_INDEX:
            raise RuntimeError("El LowCmd del SDK tiene menos de 30 entradas")

        with self.sdk_cmd_lock:
            # xr_teleoperate usa mode_pr=0, NO el valor del LowState.
            self.low_cmd.mode_machine = int(mode_machine)
            self.low_cmd.mode_pr = 0
            for i in range(29):
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.q = float(measured[i])
                motor.dq = 0.0
                motor.tau = 0.0
                if self.xr_reference_pd:
                    motor.kp, motor.kd = self.xr_pd_for_motor(i)
                elif i in CONTROLLED_INDICES:
                    motor.kp = float(self.kp_hold)
                    motor.kd = float(self.kd_hold)
                else:
                    # Mantener perfil conocido: las piernas siguen bajo el
                    # controlador de marcha. Contrasta con el perfil xr exacto.
                    motor.kp = 0.0
                    motor.kd = 0.0
            self.low_cmd.motor_cmd[WEIGHT_INDEX].q = 0.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_initialized = True
            self.lowcmd_init_mode_machine = int(mode_machine)
            self.lowcmd_init_mode_pr = 0

        self.get_logger().info(
            "[XR:LOWCMD_INIT] "
            f"mode_machine={mode_machine}, mode_pr=0 (LowState={mode_pr}), "
            f"q_waist={[round(measured[i], 4) for i in WAIST_INDICES]}, "
            f"PD_waist={[(self.low_cmd.motor_cmd[i].kp, self.low_cmd.motor_cmd[i].kd) for i in WAIST_INDICES]}, "
            f"perfil={'xr_reference' if self.xr_reference_pd else 'conservador'}, "
            "q[0:29] y motor.mode=1 inicializados; actuacion de brazos=15..28."
        )

    def debug_log(self, key, message, level="info", period=None):
        """Log limitado en frecuencia; no cambia el estado de control."""
        if not self.debug_logs:
            return
        now = time.monotonic()
        gap = self.debug_period if period is None else period
        with self.lock:
            prev = self._dbg_last.get(key, float("-inf"))
            if now - prev < gap:
                return
            self._dbg_last[key] = now
        logger = self.get_logger()
        log_message = f"[DBG:{key}] {message}"
        # rclpy asocia un nivel de severidad fijo a cada punto de llamada.
        # Usar getattr(logger, level)(...) desde la misma línea provoca:
        # ValueError: Logger severity cannot be changed between calls.
        # Las llamadas de cada nivel van en líneas diferentes.
        if level in ("warning", "warn"):
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
        elif level == "debug":
            logger.debug(log_message)
        else:
            logger.info(log_message)

    @staticmethod
    def debug_age(timestamp, now):
        return "NUNCA" if timestamp is None else f"{max(0.0, now-timestamp):.3f}s"

    def debug_status_callback(self):
        """Resumen 1 Hz: muestra en qué filtro se pierden /mode y /cmd_vel."""
        now = time.monotonic()
        with self.lock:
            mode = self.requested_mode
            state = self.state
            weight = self.weight
            mode_age = self.debug_age(self.last_mode_time, now)
            rx_age = self.debug_age(self._dbg_last_cmd_received_at, now)
            cmd_age = self.debug_age(self.last_cmd_vel_time, now)
            low_age = self.debug_age(self.last_lowstate_time, now)
            pending = self.stop_move_pending
            loco_stopped = self.loco_stop_sent
            handover = (self.arm_activation_after_stop,
                        self.arm_resume_after_stop, self.arm_release_after_stop)
            velocity = self.last_cmd_vel
            mode_count = self._dbg_mode_received
            rx_count = self._dbg_cmd_received
            accept_count = self._dbg_cmd_accepted
            reject_count = self._dbg_cmd_rejected
            move_count = self._dbg_move_count
            stop_count = self._dbg_stop_count
            write_count = self._dbg_arm_writes
            move_ms = self._dbg_last_move_ms
            stop_ms = self._dbg_last_stop_ms
            # Contadores del último intervalo de diagnóstico.
            self._dbg_mode_received = 0
            self._dbg_cmd_received = 0
            self._dbg_cmd_accepted = 0
            self._dbg_cmd_rejected = 0
            self._dbg_move_count = 0
            self._dbg_stop_count = 0
            self._dbg_arm_writes = 0
            lowcmd_info = (
                f"lowcmd_init={self.lowcmd_initialized} "
                f"lowstate_modes=({self.lowstate_mode_machine},{self.lowstate_mode_pr}) "
                f"lowcmd_modes=({self.lowcmd_init_mode_machine},{self.lowcmd_init_mode_pr})"
            )
            waist_error = None
            if self.measured_positions is not None and self.command_targets:
                if all(i in self.command_targets and i in self.measured_positions for i in (12,13,14)):
                    waist_error = max(abs(self.command_targets[i] - self.measured_positions[i])
                                      for i in (12,13,14))
            hold_error = None
            if self.measured_positions is not None and self.command_targets:
                indexes = [i for i in CONTROLLED_INDICES
                           if i in self.measured_positions and i in self.command_targets]
                if indexes:
                    hold_error = max(abs(self.command_targets[i] - self.measured_positions[i])
                                     for i in indexes)
        speed_text = "None" if velocity is None else str(tuple(round(v, 3) for v in velocity))
        self.get_logger().info(
            f"[DBG:RESUMEN] mode={mode} state={state} weight={weight:.3f} "
            f"/mode_age={mode_age} /cmd_vel_rx_age={rx_age} "
            f"cmd_aceptado_age={cmd_age} LowState_age={low_age} "
            f"stop_pending={pending} loco_stop_sent={loco_stopped} "
            f"handover(activar,reanudar,liberar)={handover} cmd={speed_text} | "
            f"recibidos/s mode={mode_count} cmd={rx_count} aceptados={accept_count} "
            f"rechazados={reject_count} Move={move_count} StopMove={stop_count} "
            f"arm_sdk={write_count} | ultima_llamada_Move={move_ms}ms "
            f"StopMove={stop_ms}ms "
            f"error_max_articular={hold_error if hold_error is not None else 'N/A'}rad "
            f"error_max_cintura={waist_error if waist_error is not None else 'N/A'}rad "
            f"{lowcmd_info}"
        )

    def cmd_vel_callback(self, msg: Twist):
        """Accept fresh cmd_vel in mode 1/IDLE or mode 3/HOLDING."""
        now = time.monotonic()
        values = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
        with self.lock:
            self._dbg_cmd_received += 1
            self._dbg_last_cmd_received_at = now
        if not np.all(np.isfinite(values)):
            with self.lock:
                self._dbg_cmd_rejected += 1
            self.get_logger().warning("[DBG:CMD_RECHAZADO] cmd_vel no finito")
            return

        with self.lock:
            mode = self.requested_mode
            allowed = (
                (mode == MODE_LOCOMOTION and self.state == "IDLE")
                or (mode == MODE_HOLD_WALK and self.state == "HOLDING"
                 and self.weight >= 0.999 and self.mode3_motion_permitted)
            )
            if not allowed or self.stop_move_pending:
                self._dbg_cmd_rejected += 1
                self.debug_log(
                    "CMD_RECHAZADO",
                    f"rx={tuple(round(v,3) for v in values)} mode={mode} "
                    f"state={self.state} weight={self.weight:.3f} "
                    f"allowed={allowed} stop_pending={self.stop_move_pending} "
                    f"regular_confirmed={self.mode3_motion_permitted}",
                    level="warning")
                return
            if self.last_mode_time is None or now - self.last_mode_time > self.mode_timeout:
                self._dbg_cmd_rejected += 1
                self.debug_log("CMD_RECHAZADO", "Último /mode caducado", level="warning")
                return

            # The 29-DoF G1 is not guaranteed to walk stably when arm_sdk
            # holds the upper body. Mode 3 uses lower configurable speeds.
            limits = (
                (self.mode3_max_vx, self.mode3_max_vy, self.mode3_max_wz)
                if mode == MODE_HOLD_WALK
                else (self.max_vx, self.max_vy, self.max_wz)
            )
            self.last_cmd_vel = tuple(
                float(np.clip(v, -lim, lim)) for v, lim in zip(values, limits)
            )
            self.last_cmd_vel_time = now
            self._dbg_cmd_accepted += 1
            self.debug_log(
                "CMD_ACEPTADO",
                f"mode={mode} state={self.state} recibido={tuple(round(v,3) for v in values)} "
                f"tras_limites={tuple(round(v,3) for v in self.last_cmd_vel)} "
                f"limites={limits}")

    
    def arm_velocity_callback(self, msg: MultiDOFJointTrajectory):
        if len(msg.points) != 1:
            self.debug_log("VEL_MSG", "Se esperaba exactamente 1 punto", level="warning")
            return
        point = msg.points[0]
        if len(point.velocities) != len(msg.joint_names):
            self.debug_log("VEL_MSG", "velocities y joint_names difieren en tamaño", level="warning")
            return

        # Un brazo que no aparece en joint_names queda con velocidad 0
        velocities = {"left": (np.zeros(3), np.zeros(3)), "right": (np.zeros(3), np.zeros(3))}
        try:
            seen = set()
            for name, twist in zip(msg.joint_names, point.velocities):
                arm = LINK_TO_ARM.get(name)
                if arm is None or arm in seen:
                    raise ValueError(f"joint_name no válido o repetido: {name}")
                seen.add(arm)
                lin = np.array([twist.linear.x, twist.linear.y, twist.linear.z], dtype=float)
                ang = np.array([twist.angular.x, twist.angular.y, twist.angular.z], dtype=float)
                if not (np.all(np.isfinite(lin)) and np.all(np.isfinite(ang))):
                    raise ValueError(f"velocidad no finita en {arm}")
                velocities[arm] = (lin, ang)

            # Si el marco no es el de la IK, solo se rota (son vectores libres)
            frame = msg.header.frame_id
            if frame and frame != self.ik_frame_id:
                tf = self.tf_buffer.lookup_transform(self.ik_frame_id, frame, Time())
                r = tf.transform.rotation
                q = [r.x, r.y, r.z, r.w]
                velocities = {a: (quat_rotate(q, l), quat_rotate(q, w))
                            for a, (l, w) in velocities.items()}
        except Exception as exc:
            self.debug_log("VEL_MSG", f"Mensaje de velocidades rechazado: {exc}", level="warning")
            return

        now = time.monotonic()
        with self.lock:
            for arm, (lin, ang) in velocities.items():
                self.arm_velocity[arm] = np.clip(
                    lin, -self.max_arm_linear_speed, self.max_arm_linear_speed)
                self.arm_angular[arm] = np.clip(
                    ang, -self.max_arm_angular_speed, self.max_arm_angular_speed)
                self.arm_velocity_time[arm] = now
    

    def arm_velocity_timer_callback(self):
        now = time.monotonic()
        with self.lock:
            # Solo en modo 2 y con arm_sdk completamente activo
            if not (self.requested_mode == MODE_ARMS and self.state == "ACTIVE"
                    and self.cartesian_reference and self.command_targets
                    and self.measured_qpos is not None):
                self.velocity_targets = {}
                self.last_velocity_tick = None
                return

            moving = {}
            for arm in ("left", "right"):
                stamp = self.arm_velocity_time[arm]
                fresh = stamp is not None and now - stamp <= self.arm_velocity_timeout
                lin, ang = self.arm_velocity[arm], self.arm_angular[arm]          
                if fresh and (np.linalg.norm(lin) > 1e-4 or np.linalg.norm(ang) > 1e-4):  
                    moving[arm] = (lin.copy(), ang.copy())                        
                else:
                    self.velocity_targets.pop(arm, None)   # brazo quieto: olvidar su objetivo
            if not moving:
                self.last_velocity_tick = None
                return

            nominal = 1.0 / self.arm_velocity_rate
            dt = nominal if self.last_velocity_tick is None else float(
                np.clip(now - self.last_velocity_tick, 1e-3, 0.1))
            self.last_velocity_tick = now

            # El joystick tiene prioridad sobre cualquier trayectoria de waypoints
            self.cancel_trajectory_locked()
            generation = self.plan_generation
            start_targets = self.command_targets.copy()
            qpos = self.measured_qpos.copy()
            for i in CONTROLLED_INDICES:
                qpos[self.qadr[i]] = start_targets[i]
            refs = self.cartesian_reference
            velocity_targets = self.velocity_targets

        # --- IK fuera del lock: el hilo de 250 Hz no debe esperar a MuJoCo ---
        try:
            max_step = self.max_joint_velocity * dt
            for arm, (v, w) in moving.items():                                    
                tgt = velocity_targets.get(arm)
                if tgt is None:   # inicio de un movimiento: partir de donde está el brazo
                    pos, euler = self.solver.forward_kinematics(qpos, arm)
                    tgt = {"position": pos.copy(), "euler": euler.copy(),
                        "origin": refs[arm]["position"].copy(),
                        "origin_euler": np.asarray(refs[arm]["euler"], dtype=float).copy()}  
                    velocity_targets[arm] = tgt

                tgt["position"] = np.clip(
                    tgt["position"] + v * dt,
                    tgt["origin"] - self.max_cartesian_offset,
                    tgt["origin"] + self.max_cartesian_offset)

                # Orientación: roll/pitch/yaw += velocidad·dt, dentro de ±max_angular_offset
                delta = wrap_angle(tgt["euler"] + w * dt - tgt["origin_euler"])  
                tgt["euler"] = tgt["origin_euler"] + np.clip(                    
                    delta, -self.max_angular_offset, self.max_angular_offset)    

                qpos = self.solver.solve(tgt["position"], tgt["euler"], qpos, arm,
                                        max_iterations=self.max_ik_iterations)

                # Límite de velocidad articular respecto a la consigna anterior
                for i in ARM_MOTORS[arm]:
                    a = self.qadr[i]
                    q0 = start_targets[i]
                    qpos[a] = q0 + float(np.clip(qpos[a] - q0, -max_step, max_step))

                # Anti-acumulación: si el objetivo se separó demasiado de lo alcanzado
                # (límite articular, singularidad, límite de velocidad), reanclarlo
                actual, actual_euler = self.solver.forward_kinematics(qpos, arm)  
                if np.linalg.norm(tgt["position"] - actual) > self.windup_limit:
                    tgt["position"] = actual.copy()
                if np.linalg.norm(wrap_angle(tgt["euler"] - actual_euler)) > self.orientation_windup_limit:  
                    tgt["euler"] = np.asarray(actual_euler, dtype=float).copy()                              

            targets = {i: float(qpos[self.qadr[i]]) for i in CONTROLLED_INDICES}
        except Exception as exc:
            self.debug_log("VEL_IK", f"IK por velocidad falló: {exc}", level="warning")
            return

        with self.lock:
            if (generation != self.plan_generation or self.requested_mode != MODE_ARMS
                    or self.state != "ACTIVE"):
                return
            self.command_targets = targets
    
    def locomotion_timer_callback(self):
        """Send Move in mode 1/3; StopMove is processed before mode handovers."""
        if not self.enable_hardware or self.loco_client is None:
            return

        now = time.monotonic()
        with self.lock:
            mode_fresh = (
                self.last_mode_time is not None
                and now - self.last_mode_time <= self.mode_timeout
            )
            cmd_fresh = (
                self.last_cmd_vel is not None
                and self.last_cmd_vel_time is not None
                and now - self.last_cmd_vel_time <= self.cmd_vel_timeout
            )
            walking_state = (
                (self.requested_mode == MODE_LOCOMOTION and self.state == "IDLE")
                or (self.requested_mode == MODE_HOLD_WALK
                    and self.state == "HOLDING" and self.weight >= 0.999
                    and self.mode3_motion_permitted)
            )
            can_move = walking_state and mode_fresh and cmd_fresh and not self.stop_move_pending
            velocity = self.last_cmd_vel if can_move else None
            need_stop = self.stop_move_pending or (not can_move and not self.loco_stop_sent)
            if self.requested_mode in (MODE_LOCOMOTION, MODE_HOLD_WALK):
                self.debug_log(
                    "FILTRO_MOVE",
                    f"can_move={can_move} state={self.state} weight={self.weight:.3f} "
                    f"mode_fresh={mode_fresh} cmd_fresh={cmd_fresh} "
                    f"stop_pending={self.stop_move_pending} need_stop={need_stop} "
                    f"regular_confirmed={self.mode3_motion_permitted} "
                    f"last_cmd={self.last_cmd_vel}",
                    level="info" if can_move else "warning")

        if need_stop:
            stop_start = time.monotonic()
            self.debug_log(
                "STOP_START",
                f"Enviando StopMove; state={self.state}, mode={self.requested_mode}, "
                f"pending={self.stop_move_pending}", period=0)
            try:
                stop_result = self.loco_client.StopMove()
                if isinstance(stop_result, (int, np.integer)) and stop_result != 0:
                    raise RuntimeError(f"StopMove devolvió código {stop_result}")
            except Exception as exc:
                # Do not acknowledge the stop or start a handover on failure.
                self.get_logger().error(
                    f"[DBG:STOP_ERROR] StopMove falló tras "
                    f"{(time.monotonic()-stop_start)*1000:.1f} ms: {exc}")
                return

            stop_ms = (time.monotonic() - stop_start) * 1000.0
            with self.lock:
                self._dbg_stop_count += 1
                self._dbg_last_stop_ms = round(stop_ms, 1)
            self.debug_log(
                "STOP_FIN",
                f"StopMove retornó {stop_result!r} en {stop_ms:.1f} ms "
                "(retorno no garantiza parada física)", period=0)
            if stop_ms > 100.0:
                self.debug_log(
                    "BLOQUEO_SDK", f"StopMove bloqueó el executor {stop_ms:.1f} ms",
                    level="warning", period=0)

            with self.lock:
                self.loco_stop_sent = True
                self.stop_move_pending = False
                self.last_cmd_vel = None
                self.last_cmd_vel_time = None

                if self.arm_release_after_stop:
                    # Mode 3 held the last target while the gait was stopped.
                    # Now it is safe to START the original weight-down sequence.
                    self.arm_release_after_stop = False
                    self.debug_log("HANDOVER", "StopMove retornó; comenzando RELEASING 3 -> 1", period=0)
                    self.start_release_locked(
                        time.monotonic(), "locomoción detenida; liberación solicitada"
                    )

                elif self.arm_resume_after_stop and self.requested_mode == MODE_ARMS:
                    self.arm_resume_after_stop = False
                    if self.state == "HOLDING":
                        # Keep the same command_targets and weight; do NOT
                        # recapture the robot pose or restart activation.
                        self.state = "ACTIVE"
                        self.status_text = (
                            "ACTIVE: locomoción detenida; IK reanudada sin mover los brazos."
                        )
                        self.debug_log("HANDOVER", "StopMove retornó; HOLDING -> ACTIVE", period=0)

                elif (
                    self.arm_activation_after_stop
                    and self.requested_mode in (MODE_ARMS, MODE_HOLD_WALK)
                ):
                    # maintenance_callback will call begin_activation as soon
                    # as the first valid LowState is available.
                    self.arm_activation_after_stop = False
                    self.pending_activation = True
                    self.status_text = (
                        "StopMove enviado; esperando/preparando activación de arm_sdk."
                    )
                    self.debug_log("HANDOVER", "StopMove retornó; activación pendiente", period=0)
            return

        if velocity is not None:
            vx, vy, wz = velocity
            move_start = time.monotonic()
            with self.lock:
                requested_for_call = self.requested_mode
            self.debug_log("MOVE_START", f"SDK locomocion mode={requested_for_call} "
                           f"vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")
            try:
                if requested_for_call == MODE_HOLD_WALK:
                    # Move() retorna siempre None en SDK2; SetVelocity() SI
                    # devuelve el codigo de respuesta de la API (0 = aceptado).
                    move_result = self.loco_client.SetVelocity(vx, vy, wz, duration=1.0)
                    self.debug_log(
                        "LOCO_RESULT", f"SetVelocity codigo={move_result} "
                        f"mode={requested_for_call} vel=({vx:.3f},{vy:.3f},{wz:.3f})"
                    )
                    if move_result != 0:
                        raise RuntimeError(f"SetVelocity en modo 3 devolvio codigo {move_result}")
                else:
                    # Mantener exactamente la llamada que ya funcionaba en modo 1.
                    move_result = self.loco_client.Move(vx, vy, wz)
                    if isinstance(move_result, (int, np.integer)) and move_result != 0:
                        raise RuntimeError(f"Move devolvio codigo {move_result}")
            except Exception as exc:
                self.get_logger().error(
                    f"[DBG:MOVE_ERROR] Move falló tras "
                    f"{(time.monotonic()-move_start)*1000:.1f} ms: {exc}")
                with self.lock:
                    self.stop_move_pending = True
                return
            move_ms = (time.monotonic() - move_start) * 1000.0
            with self.lock:
                self._dbg_move_count += 1
                self._dbg_last_move_ms = round(move_ms, 1)
            self.debug_log(
                "MOVE_FIN", f"Move retornó {move_result!r} en {move_ms:.1f} ms "
                "(retorno no garantiza desplazamiento físico)")
            if move_ms > 100.0:
                self.debug_log(
                    "BLOQUEO_SDK", f"Move bloqueó el executor {move_ms:.1f} ms; "
                    "podrían caducar /mode o /cmd_vel", level="warning", period=0)
            with self.lock:
                self.loco_stop_sent = False

    def mode_callback(self, msg: Int8):
        """1: locomotion/release; 2: arm IK; 3: hold arms + locomotion."""
        mode = int(msg.data)
        now = time.monotonic()
        activate_directly = False

        with self.lock:
            self._dbg_mode_received += 1
            # /mode is continuously published at 10 Hz; only edges matter.
            self.last_mode_time = now
            if mode == self.last_mode:
                return

            previous = self.last_mode
            self.last_mode = mode
            self.requested_mode = mode
            self.arm_velocity = {"left": np.zeros(3), "right": np.zeros(3)}

            self.arm_angular = {"left": np.zeros(3), "right": np.zeros(3)}

            self.arm_velocity_time = {"left": None, "right": None}
            self.velocity_targets = {}

            # Require a fresh cmd_vel after EVERY mode switch.
            self.last_cmd_vel = None
            self.last_cmd_vel_time = None
            self.pending_activation = False
            self.arm_activation_after_stop = False
            self.arm_resume_after_stop = False
            self.arm_release_after_stop = False

            if mode == MODE_ARMS:
                if self.state == "HOLDING":
                    # 3 -> 2: KEEP the held joints at their last command.
                    # Stop gait before allowing new Cartesian trajectories.
                    if self.enable_hardware and self.loco_client is not None:
                        self.stop_move_pending = True
                        self.arm_resume_after_stop = True
                        self.status_text = (
                            "3 -> 2: deteniendo locomoción; conservando postura HOLDING."
                        )
                    else:
                        self.state = "ACTIVE"
                        self.status_text = "ACTIVE: IK reanudada desde la última postura."
                elif self.state == "ACTIVE":
                    self.status_text = "ACTIVE: teleoperación cartesiana."
                elif self.state == "ACTIVATING":
                    # The ongoing ramp finishes in ACTIVE (not HOLDING).
                    self.status_text = "ACTIVATING: completando transferencia a modo 2."
                else:
                    # Fresh startup or reactivation after weight has been released.
                    if self.enable_hardware and self.loco_client is not None:
                        self.stop_move_pending = True
                        self.arm_activation_after_stop = True
                        self.status_text = (
                            f"{previous} -> 2: StopMove antes de activar arm_sdk."
                        )
                    else:
                        self.pending_activation = True
                        activate_directly = self.measured_qpos is not None
                        self.status_text = "Modo 2: solicitando control cartesiano."

            elif mode == MODE_HOLD_WALK:
                if self.state == "ACTIVE":
                    # 2 -> 3: freeze trajectory at the current commanded point.
                    self.advance_trajectory_locked(now)
                    self.cancel_trajectory_locked()
                    self.state = "HOLDING"
                    self.weight = 1.0
                    self.status_text = (
                        "HOLDING: brazos fijos; locomocion "
                        + ("habilitada" if self.mode3_motion_permitted else "BLOQUEADA (regular no confirmado)")
                    )
                elif self.state == "HOLDING":
                    self.status_text = "HOLDING: manteniendo consignas de brazos."
                elif self.state == "ACTIVATING":
                    # A mode 2 -> 3 switch during ramp ends in HOLDING.
                    self.cancel_trajectory_locked()
                    self.status_text = (
                        "ACTIVATING: completando rampa antes de habilitar modo 3."
                    )
                else:
                    # 1 -> 3 (including startup): stop any previous gait
                    # before capturing the measured arm pose and ramping weight.
                    if self.enable_hardware and self.loco_client is not None:
                        self.stop_move_pending = True
                        self.arm_activation_after_stop = True
                        self.status_text = (
                            f"{previous} -> 3: deteniendo marcha antes de fijar postura."
                        )
                    else:
                        self.pending_activation = True
                        activate_directly = self.measured_qpos is not None
                        self.status_text = "Modo 3: inicializando HOLDING."

            else:
                # Mode 1 preserves original behavior: release arm_sdk.
                # Unknown modes are treated as an inactive/safe stop.
                if mode != MODE_LOCOMOTION and self.enable_hardware:
                    self.stop_move_pending = True

                if self.state == "HOLDING":
                    # While in mode 3, stop walking BEFORE reducing the
                    # upper-body weight. Keep the joints held until StopMove.
                    self.cancel_trajectory_locked()
                    if self.enable_hardware and self.loco_client is not None:
                        self.stop_move_pending = True
                        self.arm_release_after_stop = True
                        self.status_text = (
                            f"3 -> {mode}: StopMove antes de liberar arm_sdk."
                        )
                    else:
                        self.start_release_locked(now, f"modo cambiado a {mode}")
                elif self.state == "WAITING_FOR_STATE":
                    self.cancel_trajectory_locked()
                    self.state = "IDLE"
                    self.weight = 0.0
                    self.status_text = f"Modo {mode}: IDLE, sin arm_sdk."
                elif self.state in ("ACTIVATING", "ACTIVE", "RELEASING"):
                    self.start_release_locked(now, f"modo cambiado a {mode}")
                else:
                    self.cancel_trajectory_locked()
                    self.state = "IDLE"
                    self.weight = 0.0
                    self.status_text = f"Modo {mode}: IDLE, sin arm_sdk."

        self.debug_log(
            "RESULTADO_MODO",
            f"mode={mode} state={self.state} weight={self.weight:.3f} "
            f"stop_pending={self.stop_move_pending} "
            f"resume_after_stop={self.arm_resume_after_stop} "
            f"activate_after_stop={self.arm_activation_after_stop} "
            f"release_after_stop={self.arm_release_after_stop}",
            period=0)
        if activate_directly:
            self.begin_activation()

    def maintenance_callback(self):
        with self.lock:
            # Si StopMove ya se ha enviado, pero todavía no había LowState,
            # permitir la activación cuando llegue un estado articular válido.
            if (
                self.arm_activation_after_stop
                and not self.stop_move_pending
                and self.requested_mode in (MODE_ARMS, MODE_HOLD_WALK)
                and self.measured_qpos is not None
            ):
                self.arm_activation_after_stop = False
                self.pending_activation = True

            pending = (
                self.pending_activation
                and self.requested_mode in (MODE_ARMS, MODE_HOLD_WALK)
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
            if self.requested_mode not in (MODE_ARMS, MODE_HOLD_WALK) or self.measured_qpos is None:
                return
            if (self.enable_hardware and
                    (self.last_lowstate_time is None or
                     time.monotonic() - self.last_lowstate_time > self.lowstate_timeout)):
                self.state = "WAITING_FOR_STATE"
                self.pending_activation = True
                self.status_text = "Modo 2/3 recibido; esperando LowState reciente antes de activar arm_sdk."
                return
            qpos = self.measured_qpos.copy()
            measured = self.measured_positions.copy()
            mode_machine = self.lowstate_mode_machine
            mode_pr = self.lowstate_mode_pr
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
            if self.requested_mode not in (MODE_ARMS, MODE_HOLD_WALK):
                return
            # Initialize the whole rt/arm_sdk payload BEFORE weight ramps up.
            # This is done on each fresh activation, not on 3 -> 2 handovers.
            if self.enable_hardware:
                try:
                    self.initialize_lowcmd_from_lowstate(
                        measured, mode_machine, mode_pr)
                except Exception as exc:
                    self.state = "WAITING_FOR_STATE"
                    self.pending_activation = False
                    self.status_text = f"LowCmd no inicializado: {exc}"
                    self.get_logger().error(f"[LOWCMD:INIT_ERROR] {exc}")
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
            self.debug_log("ARM_STATE", "Entrando a ACTIVATING; referencia capturada", period=0)

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
        self.debug_log("ARM_STATE", self.status_text, period=0)

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
                # En XR la cintura conserva el q inicial. El solucionador IK
                # externo podria modificar cintura sin avisar. No ejecutamos
                # soluciones que dependan de ese movimiento no comandado.
                max_waist_delta = max(abs(targets[i] - start_targets[i]) for i in WAIST_INDICES)
                if max_waist_delta > 0.03:
                    raise ValueError(
                        f"IK necesita mover la cintura {max_waist_delta:.3f} rad; "
                        "perfil XR solo ejecuta brazos 15..28. Revisar G1IKSolver.")
                for i in WAIST_INDICES:
                    targets[i] = start_targets[i]
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
                if (self.enable_hardware and self.state in (
                        "WAITING_FOR_STATE", "ACTIVATING", "ACTIVE", "HOLDING", "RELEASING")
                    and self.last_mode_time is not None
                    and now - self.last_mode_time > self.mode_timeout):
                    self.last_mode_time = None
                    self.last_mode = None
                    self.requested_mode = None
                    self.pending_activation = False
                    self.arm_activation_after_stop = False
                    self.arm_resume_after_stop = False
                    if self.state == "HOLDING":
                        # If moving while holding, request a gait stop before
                        # releasing upper-body control.
                        self.stop_move_pending = True
                        self.arm_release_after_stop = True
                        self.status_text = "Timeout modo 3: StopMove pendiente antes de liberar."
                        self.debug_log("WATCHDOG", "Timeout /mode en HOLDING; solicitando StopMove", level="warning", period=0)
                    else:
                        self.debug_log("WATCHDOG", "Timeout /mode; liberación arm_sdk", level="warning", period=0)
                        self.start_release_locked(now, "timeout del tópico de modo")
                if (self.enable_hardware and self.state in (
                        "ACTIVATING", "ACTIVE", "HOLDING")
                    and self.last_lowstate_time is not None
                    and now - self.last_lowstate_time > self.lowstate_timeout):
                    self.last_lowstate_time = None
                    self.pending_activation = False
                    self.arm_activation_after_stop = False
                    if self.state == "HOLDING":
                        self.stop_move_pending = True
                        self.arm_release_after_stop = True
                        self.status_text = "Timeout LowState modo 3: StopMove pendiente."
                        self.debug_log(
                            "WATCHDOG", "Timeout rt/lowstate en HOLDING; StopMove pendiente",
                            level="warning", period=0)
                    else:
                        self.debug_log(
                            "WATCHDOG", "Timeout rt/lowstate; liberación arm_sdk",
                            level="warning", period=0)
                        self.start_release_locked(
                            now, "timeout de rt/lowstate; requiere cambiar modo para reactivar")
                if self.state == "ACTIVE":
                    self.advance_trajectory_locked(now)
                old_state = self.state
                if old_state in ("ACTIVATING", "RELEASING"):
                    alpha = float(np.clip((now-self.transition_start)/self.transition_duration, 0.0, 1.0))
                    self.weight = self.transition_from + alpha*(self.transition_to-self.transition_from)
                    if alpha >= 1.0 and old_state == "ACTIVATING":
                        self.weight = 1.0
                        if self.requested_mode == MODE_HOLD_WALK:
                            self.state = "HOLDING"
                            self.status_text = (
                                "HOLDING: arm_sdk activo; cmd_vel "
                                + ("habilitado" if self.mode3_motion_permitted else "bloqueado (Regular no confirmado)")
                            )
                            self.debug_log("ARM_STATE", "ACTIVATING -> HOLDING; weight=1.0", period=0)
                        elif self.requested_mode == MODE_ARMS:
                            self.state = "ACTIVE"
                            self.status_text = (
                                "ACTIVE: arm_sdk habilitado; esperando/ejecutando waypoints."
                            )
                            self.debug_log("ARM_STATE", "ACTIVATING -> ACTIVE; weight=1.0", period=0)
                        else:
                            self.start_release_locked(now, "modo ya no activo")
                    elif alpha >= 1.0 and old_state == "RELEASING":
                        self.weight, self.state, self.trajectory = 0.0, "IDLE", None
                        self.active_arms = {"left", "right"}
                        self.status_text = "IDLE: peso arm_sdk=0; locomoción puede recuperar el control."
                        self.debug_log("ARM_STATE", "RELEASING -> IDLE; weight=0.0", period=0)
                publish = self.enable_hardware and old_state in ("ACTIVATING", "ACTIVE", "HOLDING", "RELEASING")
                targets = self.command_targets.copy() if publish else {}
                active_arms = set(self.active_arms)
                weight = float(self.weight)
                if not self.enable_hardware and self.state in ("ACTIVATING", "ACTIVE", "HOLDING", "RELEASING"):
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
                    if self.debug_logs:
                        with self.lock:
                            self._dbg_arm_writes += 1
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
        if not self.lowcmd_initialized:
            raise RuntimeError("LowCmd no inicializado; se inhibe rt/arm_sdk")
        with self.sdk_cmd_lock:
            # xr_teleoperate: las posiciones no-brazo (incluida cintura 12..14)
            # se capturan una vez y no se sobrescriben en cada ciclo.
            for i in ARM_INDICES:
                name = MOTOR_JOINTS[i]
                arm = "left" if name.startswith("left_") else "right"
                active = arm in active_arms
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.q = float(targets[i])
                motor.dq = 0.0
                motor.tau = 0.0
                if self.xr_reference_pd:
                    motor.kp, motor.kd = self.xr_pd_for_motor(i)
                else:
                    motor.kp = self.kp_active if active else self.kp_hold
                    motor.kd = self.kd_active if active else self.kd_hold
            # La cintura mantiene pose+PD inicializados, nunca se pone a cero.
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
                self.pending_activation = False
                self.arm_activation_after_stop = False
                self.arm_resume_after_stop = False
                self.arm_release_after_stop = False
                self.stop_move_pending = False
                self.last_cmd_vel = None
                self.last_cmd_vel_time = None
                if self.state in ("ACTIVATING", "ACTIVE", "HOLDING", "RELEASING"):
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
