from __future__ import annotations

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_USD_PATH = os.path.join(_THIS_DIR, "usd", "robot.usd")

try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
except ModuleNotFoundError as exc:
    # Allow lightweight imports (e.g., checking _USD_PATH) without Isaac Sim runtime.
    if exc.name != "pxr":
        raise
    CAR_ROBOT_CFG = None
else:
    CAR_ROBOT_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "front_left_knuckle_joint": 0.0,
                "front_right_knuckle_joint": 0.0,
                "fr_wheel_joint": 0.0,
                "fl_wheel_joint": 0.0,
                "rl_wheel_joint": 0.0,
                "rr_wheel_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_knuckle_joint", "front_right_knuckle_joint"],
                stiffness=1000.0,
                damping=100.0,
                effort_limit=200.0,
                velocity_limit=3.0,
            ),
            "rear_drive": ImplicitActuatorCfg(
                joint_names_expr=["rl_wheel_joint", "rr_wheel_joint"],
                stiffness=0.0,
                damping=10.0,
                effort_limit=400.0,
                velocity_limit=100.0,
            ),
            "front_free_wheels": ImplicitActuatorCfg(
                joint_names_expr=["fl_wheel_joint", "fr_wheel_joint"],
                stiffness=0.0,
                damping=1.0,
                effort_limit=50.0,
                velocity_limit=100.0,
            ),
        },
    )