import mujoco
import mujoco.viewer
import numpy as np
import time

class SimulationEnvironment:
    def __init__(self, xml_path="robotstudio_so101/VLA_SCENE.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        self._configure_physics()
        self._cache_ids()
        
        self.mocap_id = -1
        self._setup_mocap()

        self.grasped_object = None
        print(f"Simulation initialized | Actuators: {self.model.nu} | Mode: Mocap")

    def _configure_physics(self):
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 30
        self.model.opt.tolerance = 1e-10
        self.model.opt.impratio = 10
        
        if len(self.model.actuator_gainprm) > 0:
            self.model.actuator_gainprm[:, 0] = 0.001
            self.model.actuator_biasprm[:, 1] = 0.001

            gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
            if gripper_id >= 0:
                self.model.actuator_gainprm[gripper_id, 0] = 20.0
                self.model.actuator_biasprm[gripper_id, 1] = 2.0

    def _cache_ids(self):
        self.object_ids = {}
        for name in ["red_block", "blue_block", "green_block", "target_zone"]:
            try:
                self.object_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            except ValueError:
                self.object_ids[name] = -1
        
        self.grasp_welds = {}
        for color in ["red", "blue", "green"]:
            name = f"{color}_block"
            weld_name = f"grasp_{color}"
            try:
                self.grasp_welds[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            except ValueError:
                pass

    def _setup_mocap(self):
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
            self.mocap_id = self.model.body_mocapid[body_id]
        except ValueError:
            print("Warning: Mocap body 'hand_target' not found.")

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def launch_viewer(self):
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.distance = 1.0
                self.viewer.cam.lookat[:] = [0.35, 0, 0.1]
                self.viewer.cam.azimuth = 140
                self.viewer.cam.elevation = -25
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        except RuntimeError:
            print("Warning: Could not launch viewer (likely headless environment).")

    def set_mocap_target(self, pos, gripper_state="open"):
        if self.mocap_id == -1: return

        current_pos = self.data.mocap_pos[self.mocap_id]
        target_pos = np.array(pos, dtype=float)
        
        max_step = 0.05
        delta = target_pos - current_pos
        dist = np.linalg.norm(delta)
        
        if dist > max_step:
            current_pos += (delta / dist) * max_step
        else:
            current_pos = target_pos
            
        self.data.mocap_pos[self.mocap_id] = current_pos
        self._update_gripper(gripper_state)
        self._check_grasp(gripper_state)

    def _update_gripper(self, state):
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if actuator_id >= 0:
            target = 1.2 if state == "closed" else -0.15
            self.data.ctrl[actuator_id] = target

    def _check_grasp(self, state):
        if state == "closed":
            if self.grasped_object: return

            gripper_pos = self.data.body("gripper").xpos
            threshold = 0.05
            
            for name, weld_id in self.grasp_welds.items():
                obj_id = self.object_ids.get(name)
                if obj_id == -1: continue
                
                obj_pos = self.data.body(obj_id).xpos
                if np.linalg.norm(gripper_pos - obj_pos) < threshold:
                    if self.data.eq_active[weld_id] == 0:
                        self.data.eq_active[weld_id] = 1.0
                        self.grasped_object = name
                        print(f"Grasped: {name}")
                        return
                        
        elif state == "open" and self.grasped_object:
            weld_id = self.grasp_welds.get(self.grasped_object)
            if weld_id is not None:
                self.data.eq_active[weld_id] = 0.0
                print(f"Released: {self.grasped_object}")
            self.grasped_object = None

    def wait(self, duration):
        steps = int(duration / self.model.opt.timestep)
        for _ in range(steps):
            self.step()
            if self.viewer:
                self.viewer.sync()

    def wait_for_stability(self, max_time=3.0):
        start = time.time()
        while time.time() - start < max_time:
            self.step()
            if self.viewer: self.viewer.sync()
            
            if np.max(np.abs(self.data.qvel)) < 0.01:
                break

    def get_object_position(self, name):
        try:
            return self.data.body(name).xpos.copy()
        except ValueError:
            return None
