import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer

# ---------- convex hull (monotonic chain) ----------
def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)

# ---------- contact -> support polygon & ZMP ----------
@dataclass
class ContactConfig:
    foot_geom_names: List[str]
    floor_geom_name: str
    min_fz: float = 10.0  # N

def _id(model, obj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise ValueError(f"Name not found: {name}")
    return idx

def compute_support_and_zmp(model, data, cfg: ContactConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    foot_ids = set(_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in cfg.foot_geom_names)
    floor_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.floor_geom_name)

    pts_xy = []
    F = np.zeros(3)
    tau = np.zeros(3)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        ok = ((g1 in foot_ids and g2 == floor_id) or (g2 in foot_ids and g1 == floor_id))
        if not ok:
            continue

        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)

        # con.frame is stored transposed; interpret as 3x3 then use R.T to map to world
        R = np.array(con.frame, dtype=np.float64).reshape(3, 3)
        f_world = R.T @ f6[:3]
        tau_world = R.T @ f6[3:]

        if f_world[2] < cfg.min_fz:
            continue

        p = np.array(con.pos, dtype=np.float64)
        pts_xy.append(p[:2])

        F += f_world
        tau += np.cross(p, f_world) + tau_world

    if len(pts_xy) == 0 or abs(F[2]) < 1e-6:
        return np.zeros((0, 2)), None

    pts_xy = np.array(pts_xy, dtype=np.float64)
    zmp = np.array([-tau[1] / F[2], tau[0] / F[2]], dtype=np.float64)
    return pts_xy, zmp

# ---------- draw in viewer.user_scn ----------
def draw_zmp_and_hull(viewer, zmp_xy: Optional[np.ndarray], hull_xy: np.ndarray, z=0.01):
    viewer.user_scn.ngeom = 0
    idx = 0

    if zmp_xy is not None:
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.015, 0, 0], dtype=np.float64),
            pos=np.array([zmp_xy[0], zmp_xy[1], z], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0.2, 0.2, 1], dtype=np.float64),
        )
        idx += 1

    if hull_xy is not None and len(hull_xy) >= 2:
        loop = np.vstack([hull_xy, hull_xy[0]])
        for a, b in zip(loop[:-1], loop[1:]):
            pa = np.array([a[0], a[1], z], dtype=np.float64)
            pb = np.array([b[0], b[1], z], dtype=np.float64)
            mujoco.mjv_connector(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=2,
                from_=pa,
                to=pb,
                rgba=np.array([0.2, 0.9, 0.2, 1], dtype=np.float64),
            )
            idx += 1

    viewer.user_scn.ngeom = idx

# ---------- PD control mapping ----------
@dataclass
class JointMap:
    joint_name: str
    actuator_name: str
    qpos_adr: int
    qvel_adr: int
    act_id: int
    tau_min: float
    tau_max: float

def build_joint_maps(model: mujoco.MjModel, joint_to_act: List[Tuple[str, str]]) -> List[JointMap]:
    maps = []
    for jn, an in joint_to_act:
        jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        # actuator ctrlrange
        r = model.actuator_ctrlrange[aid]
        maps.append(JointMap(jn, an, qadr, vadr, aid, float(r[0]), float(r[1])))
    return maps

# ---------- Simple Tk UI for q_des, kp, kd ----------
def start_tk_ui(shared):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("PD Tuner (q_des / kp / kd)")

    # global kp/kd
    kp_var = tk.DoubleVar(value=shared["kp"])
    kd_var = tk.DoubleVar(value=shared["kd"])

    def set_kp(v):
        shared["kp"] = float(v)

    def set_kd(v):
        shared["kd"] = float(v)

    ttk.Label(root, text="Global Kp").grid(row=0, column=0, sticky="w")
    ttk.Scale(root, from_=0, to=600, orient="horizontal", variable=kp_var, command=set_kp, length=300).grid(row=0, column=1, sticky="ew")

    ttk.Label(root, text="Global Kd").grid(row=1, column=0, sticky="w")
    ttk.Scale(root, from_=0, to=50, orient="horizontal", variable=kd_var, command=set_kd, length=300).grid(row=1, column=1, sticky="ew")

    ttk.Separator(root, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)

    # per-joint desired
    for i, name in enumerate(shared["q_des"].keys()):
        var = tk.DoubleVar(value=shared["q_des"][name])

        def make_cb(nm):
            def _cb(v):
                shared["q_des"][nm] = float(v)
            return _cb

        ttk.Label(root, text=name).grid(row=3+i, column=0, sticky="w")
        ttk.Scale(root, from_=-1.5, to=1.5, orient="horizontal", variable=var, command=make_cb(name), length=300)\
            .grid(row=3+i, column=1, sticky="ew")

    root.columnconfigure(1, weight=1)
    root.mainloop()

# ---------- Main loop ----------
@dataclass
class AppState:
    reload_requested: bool = False
    paused: bool = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--floor_geom", default="floor")
    parser.add_argument("--foot_geoms", nargs="+", default=["left_sole", "right_sole"])
    args = parser.parse_args()

    xml_path = args.xml
    contact_cfg = ContactConfig(args.foot_geoms, args.floor_geom, min_fz=10.0)

    # Joint list (your 12 DOF)
    joint_to_act = [
        ("Hip_L_Roll_Joint",   "m_Hip_L_Roll"),
        ("Hip_L_Yaw_Joint",    "m_Hip_L_Yaw"),
        ("Hip_L_Pitch_Joint",  "m_Hip_L_Pitch"),
        ("Knee_L_Pitch_Joint", "m_Knee_L_Pitch"),
        ("Ankle_L_Pitch_Joint","m_Ankle_L_Pitch"),
        ("Ankle_L_Roll_Joint", "m_Ankle_L_Roll"),
        ("Hip_R_Roll_Joint",   "m_Hip_R_Roll"),
        ("Hip_R_Yaw_Joint",    "m_Hip_R_Yaw"),
        ("Hip_R_Pitch_Joint",  "m_Hip_R_Pitch"),
        ("Knee_R_Pitch_Joint", "m_Knee_R_Pitch"),
        ("Ankle_R_Pitch_Joint","m_Ankle_R_Pitch"),
        ("Ankle_R_Roll_Joint", "m_Ankle_R_Roll"),
    ]

    # Shared UI state
    shared = {
        "kp": 220.0,
        "kd": 10.0,
        "q_des": {jn: 0.0 for jn, _ in joint_to_act},
    }

    # A reasonable initial “stand-ish” seed (你可以在 UI 里继续调)
    shared["q_des"]["Hip_L_Pitch_Joint"] = -0.15
    shared["q_des"]["Knee_L_Pitch_Joint"] = 0.45
    shared["q_des"]["Ankle_L_Pitch_Joint"] = -0.25
    shared["q_des"]["Hip_R_Pitch_Joint"] = -0.15
    shared["q_des"]["Knee_R_Pitch_Joint"] = 0.45
    shared["q_des"]["Ankle_R_Pitch_Joint"] = -0.25

    # start Tk UI in another thread
    ui_thread = threading.Thread(target=start_tk_ui, args=(shared,), daemon=True)
    ui_thread.start()

    state = AppState()

    def key_callback(key):
        if key in (ord("r"), ord("R")):
            state.reload_requested = True
        elif key == ord(" "):
            state.paused = not state.paused

    def load():
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        maps = build_joint_maps(model, joint_to_act)
        return model, data, maps

    while True:
        model, data, maps = load()

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            while viewer.is_running():
                with viewer.lock():
                    if state.reload_requested:
                        state.reload_requested = False
                        viewer.close()
                        break

                    if not state.paused:
                        # PD torque -> data.ctrl
                        kp = float(shared["kp"])
                        kd = float(shared["kd"])
                        for m in maps:
                            q = float(data.qpos[m.qpos_adr])
                            qd = float(data.qvel[m.qvel_adr])
                            q_des = float(shared["q_des"][m.joint_name])
                            tau = kp * (q_des - q) - kd * qd
                            data.ctrl[m.act_id] = np.clip(tau, m.tau_min, m.tau_max)

                        mujoco.mj_step(model, data)

                    # ZMP + support polygon
                    pts_xy, zmp_xy = compute_support_and_zmp(model, data, contact_cfg)
                    hull_xy = convex_hull_2d(pts_xy) if len(pts_xy) >= 3 else pts_xy
                    draw_zmp_and_hull(viewer, zmp_xy, hull_xy, z=0.01)

                    viewer.sync()

                time.sleep(0.001)

if __name__ == "__main__":
    main()
