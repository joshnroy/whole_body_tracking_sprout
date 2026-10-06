"""Sprout motor parameters and DC motor actuator configs.

The vendor models each motor as an Isaac Lab ``DelayedDCMotor``: a PD law whose torque is clipped by a linear
torque-speed curve (``saturation_effort`` at stall, zero at ``velocity_limit``) and by the continuous
``effort_limit``. mjlab's ``DcMotorActuatorCfg`` implements the same law, and its ``delay_min_lag``/``delay_max_lag``
cover the command delay.

``MOTOR_PARAMETERS`` is the vendor table as given; the comments record where each value came from. The armatures are
all vendor guesses.
"""

from mjlab.actuator import DcMotorActuatorCfg

MOTOR_PARAMETERS = {
    "48V-HTDW-5047-36": {
        "effort_limit": 30,  # experimental
        "saturation_effort": 114.72,  # calculated
        "velocity_limit": 15.72,  # calculated
        "rated_torque": 4.0,  # datasheet
        "bus_voltage": 54.6,  # fully-charged battery
        "max_temp": 85,  # experimental
        "winding_resistance": 0.950,  # vendor supplied
        "torque_constant": 2.39,  # experimental
        "idle_current": 0.65,  # experimental
        "thermal_resistance": 6.07,  # experimental
        "thermal_capacitance": 159.04,  # experimental
        "armature": 0.027,  # guess
    },
    "48V-HTDW-4438-30": {
        "effort_limit": 18,  # datasheet
        "saturation_effort": 28.50,  # calculated
        "velocity_limit": 37.48,  # calculated
        "rated_torque": 2.0,  # datasheet
        "bus_voltage": 54.6,  # fully-charged battery
        "max_temp": 85,  # experimental
        "winding_resistance": 0.770,  # vendor supplied
        "torque_constant": 0.819,  # experimental
        "idle_current": 0.0,
        "thermal_resistance": 5.44,  # experimental
        "thermal_capacitance": 234.43,  # experimental
        "armature": 0.027,  # guess
    },
    "48V-HTDW-3536-32": {
        "effort_limit": 3.5,  # hand-tuned
        "saturation_effort": 4.5,  # hand-tuned
        "velocity_limit": 17.5,  # hand-tuned
        "rated_torque": 1.0,  # datasheet
        "bus_voltage": 54.6,  # fully-charged battery
        "max_temp": 85,  # experimental
        "winding_resistance": 0.8,  # experimental, directly measured
        "torque_constant": 1.44,  # datasheet
        "idle_current": 0.0,
        "thermal_resistance": 4.80,  # experimental
        "thermal_capacitance": 227.29,  # experimental
        "armature": 0.027,  # guess
    },
    "48V-EC-A4310": {
        "effort_limit": 24,  # datasheet
        "saturation_effort": 105.12,  # calculated
        "velocity_limit": 19.04,  # calculated
        "rated_torque": 9,
        "bus_voltage": 54.6,  # fully-charged battery
        "max_temp": 103,  # experimental
        "winding_resistance": 0.616,  # experimental, directly measured
        "torque_constant": 1.42,  # experimental
        "thermal_resistance": 5.63,  # experimental
        "thermal_capacitance": 221.00,  # experimental
        "armature": 0.0068,  # guess
        "idle_current": 0.036,  # experimental
    },
    "48V-EC-A6408": {
        "effort_limit": 30,  # datasheet
        "saturation_effort": 128.2,  # calculated
        "velocity_limit": 15.4,  # calculated
        "rated_torque": 8,  # experimental
        "bus_voltage": 54.6,  # fully-charged battery
        "max_temp": 80,  # experimental
        "winding_resistance": 0.835,  # experimental, directly measured
        "torque_constant": 2.35,  # datasheet
        "thermal_resistance": 4.86,  # experimental
        "thermal_capacitance": 399.50,  # experimental
        "armature": 0.0068,  # guess
        "idle_current": 0.034,  # experimental
    },
}

# The subset of MOTOR_PARAMETERS that the simulated actuator uses.
SIM_MOTOR_PARAMETERS = {
    motor: {key: params[key] for key in ("effort_limit", "saturation_effort", "velocity_limit", "armature")}
    for motor, params in MOTOR_PARAMETERS.items()
}

# Joints driven by each motor type.
SPROUT_MOTOR_JOINTS = {
    "48V-EC-A6408": (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint", ".*_ankle_joint"),
    "48V-EC-A4310": (".*_shoulder_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    "48V-HTDW-5047-36": (".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"),
    "48V-HTDW-4438-30": ("neck_yaw_joint", ".*_wrist_roll_joint"),
    "48V-HTDW-3536-32": ("neck_pitch_joint", ".*_gripper_joint", ".*_wrist_pitch_joint"),
}

# Vendor PD gains, shared by every motor type.
STIFFNESS = 32.5
DAMPING = 1.0

# Command delay in physics steps (0.005 s each, so 10-25 ms). Each env samples a lag on reset and resamples it every 4 s
# (800 steps). Without a per-env phase offset the first sample lands on the first step after a reset; with one, a reset
# env would run with zero lag until its offset came round.
DELAY_MIN_LAG = 2
DELAY_MAX_LAG = 5
DELAY_UPDATE_PERIOD = 800

SPROUT_ACTUATORS = tuple(
    DcMotorActuatorCfg(
        target_names_expr=joint_names,
        stiffness=STIFFNESS,
        damping=DAMPING,
        delay_min_lag=DELAY_MIN_LAG,
        delay_max_lag=DELAY_MAX_LAG,
        delay_update_period=DELAY_UPDATE_PERIOD,
        delay_per_env_phase=False,
        **SIM_MOTOR_PARAMETERS[motor],
    )
    for motor, joint_names in SPROUT_MOTOR_JOINTS.items()
)
