import mujoco
import mujoco.viewer
import numpy as np
import time

class SimulationEnvironment:
    def __init__(self, xml_path="robotstudio_so101/VLA_SCENE.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.xml_path = xml_path
        self.viewer = None
        
        # Enhanced physics configuration
        self._configure_physics()
        
        # Cache important IDs
        self._cache_ids()
        
        # Initialize control state
        self.control_mode = "mocap"  # "mocap" or "actuator"
        self.gripper_state = "open"
        
        print(f"[SIM] Initialized with {self.model.nu} actuators")
        print(f"[SIM] Control mode: {self.control_mode}")

    def _configure_physics(self):
        """Configure physics parameters for stable simulation."""
        # Improve solver parameters for better contact handling
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 30
        self.model.opt.tolerance = 1e-10
        
        # Enhanced contact parameters
        self.model.opt.impratio = 10
        
        # Configure actuators based on control mode
        if len(self.model.actuator_gainprm) > 0:
            # For mocap mode: VERY low stiffness to prevent fighting
            self.model.actuator_gainprm[:, 0] = 0.001  # Ultra-low stiffness (was 0.01)
            self.model.actuator_biasprm[:, 1] = 0.001  # Ultra-low damping
            
            # Keep gripper actuator responsive but not too strong
            gripper_actuator_id = self._get_actuator_id("gripper")
            if gripper_actuator_id >= 0:
                self.model.actuator_gainprm[gripper_actuator_id, 0] = 20.0  # Reduced from 50
                self.model.actuator_biasprm[gripper_actuator_id, 1] = 2.0   # Reduced from 5

    def _cache_ids(self):
        """Cache frequently used MuJoCo IDs for performance."""
        try:
            # Cache grasp weld IDs for each block
            self.grasp_weld_ids = {}
            for color in ["red", "blue", "green"]:
                try:
                    weld_name = f"grasp_{color}"
                    self.grasp_weld_ids[f"{color}_block"] = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name
                    )
                except:
                    self.grasp_weld_ids[f"{color}_block"] = -1
                    print(f"[SIM WARNING] {weld_name} constraint not found")
            
            self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            self.hand_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
            
            # Cache object IDs
            self.object_ids = {}
            for obj_name in ["red_block", "blue_block", "green_block", "target_zone"]:
                try:
                    self.object_ids[obj_name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                except:
                    self.object_ids[obj_name] = -1
                    
            # Cache mocap ID
            if self.hand_target_id >= 0:
                self.mocap_id = self.model.body_mocapid[self.hand_target_id]
            else:
                self.mocap_id = -1
            
            # Track currently grasped object
            self.grasped_object = None
                
        except Exception as e:
            self.mocap_id = -1
            print(f"[SIM WARNING] ID caching failed: {e}")

    def _get_actuator_id(self, name):
        """Get actuator ID by name."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except:
            return -1

    def step(self):
        """Step the simulation forward."""
        mujoco.mj_step(self.model, self.data)

    def set_mocap_target(self, pos, gripper_state="open"):
        """
        Set mocap target with trajectory smoothing for 50Hz+ updates.
        Prevents jerky motion by limiting velocity and acceleration.
        
        Args:
            pos: Target position [x, y, z]
            gripper_state: 'open' or 'closed'
        """
        # 1. Update mocap target position with velocity limiting
        if self.mocap_id >= 0:
            current_pos = self.data.mocap_pos[self.mocap_id].copy()
            target_pos = np.array(pos, dtype=float)
            
            # Smooth trajectory: limit maximum step size (prevents jerky jumps)
            MAX_STEP_SIZE = 0.05  # 5cm max step per call (for 50Hz = 2.5 m/s max velocity)
            delta = target_pos - current_pos
            distance = np.linalg.norm(delta)
            
            if distance > MAX_STEP_SIZE:
                # Interpolate towards target smoothly
                direction = delta / distance
                smoothed_pos = current_pos + direction * MAX_STEP_SIZE
            else:
                smoothed_pos = target_pos
            
            # Apply smoothed position
            self.data.mocap_pos[self.mocap_id] = smoothed_pos
        else:
            print("[SIM WARNING] Mocap target not found")
            
        # 2. Update gripper state
        self.gripper_state = gripper_state
        self._update_gripper_control(gripper_state)
        
        # 3. Update grasp logic
        self._update_grasp_logic(gripper_state)

    def _update_gripper_control(self, gripper_state):
        """Update gripper control with improved precision."""
        gripper_actuator_id = self._get_actuator_id("gripper")
        
        if gripper_actuator_id >= 0:
            if gripper_state == "open":
                # Open position: negative angle
                self.data.ctrl[gripper_actuator_id] = -0.15
            else:
                # Closed position: positive angle for grasping
                self.data.ctrl[gripper_actuator_id] = 1.2
        else:
            print("[SIM WARNING] Gripper actuator not found")

    def _update_grasp_logic(self, gripper_state):
        """
        Enhanced grasp logic with proximity-based activation.
        Only activates when gripper leads are within specific proximity to object's center of mass.
        Prevents teleporting and glitching.
        """
        
        if gripper_state == "closed":
            # If already grasping something, maintain it
            if self.grasped_object is not None:
                return
            
            # Get gripper position (fingertip position, not wrist)
            hand_pos = self.data.body("gripper").xpos
            
            min_dist = float('inf')
            closest_object = None
            
            # Check distance to each graspable object's center of mass
            for obj_name in ["red_block", "blue_block", "green_block"]:
                if self.object_ids.get(obj_name, -1) >= 0:
                    try:
                        # Get object's center of mass position
                        obj_pos = self.data.body(obj_name).xpos
                        
                        # Calculate 3D Euclidean distance
                        dist = np.linalg.norm(hand_pos - obj_pos)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_object = obj_name
                    except:
                        continue
            
            # CRITICAL: Stricter proximity threshold (5cm) to prevent teleporting
            # Only activate grasp if gripper leads are very close to object COM
            GRASP_PROXIMITY_THRESHOLD = 0.05  # 5cm - very tight tolerance
            
            if closest_object and min_dist < GRASP_PROXIMITY_THRESHOLD:
                weld_id = self.grasp_weld_ids.get(closest_object, -1)
                
                if weld_id >= 0:
                    # Double-check object is not already attached
                    if self.data.eq_active[weld_id] == 0:
                        print(f"[SIM] 🔗 Proximity Grasp ACTIVATED on {closest_object}")
                        print(f"      Distance to COM: {min_dist*1000:.1f}mm (threshold: {GRASP_PROXIMITY_THRESHOLD*1000:.0f}mm)")
                        self.data.eq_active[weld_id] = 1.0
                        self.grasped_object = closest_object
                else:
                    print(f"[SIM WARNING] Weld constraint for {closest_object} not found")
            else:
                if closest_object and min_dist < 0.15:  # Within 15cm - give feedback
                    print(f"[SIM] ⚠️  Gripper too far from {closest_object} ({min_dist*1000:.0f}mm > {GRASP_PROXIMITY_THRESHOLD*1000:.0f}mm)", flush=True)
        
        else:  # gripper_state == "open"
            # Release any active grasp
            if self.grasped_object is not None:
                weld_id = self.grasp_weld_ids.get(self.grasped_object, -1)
                
                if weld_id >= 0 and self.data.eq_active[weld_id] == 1.0:
                    print(f"[SIM] 🔓 Grasp Released from {self.grasped_object}")
                    self.data.eq_active[weld_id] = 0.0
                
                self.grasped_object = None

    def get_object_position(self, name):
        """Get object position with error handling."""
        try:
            return self.data.body(name).xpos.copy()
        except:
            return None

    def get_object_velocity(self, name):
        """Get object velocity."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self.data.cvel[body_id].copy()
        except:
            return None

    def get_joint_positions(self):
        """Get current joint positions."""
        return self.data.qpos.copy()

    def get_joint_velocities(self):
        """Get current joint velocities."""
        return self.data.qvel.copy()

    def set_joint_positions(self, positions):
        """Set joint positions directly (for initialization)."""
        if len(positions) <= len(self.data.qpos):
            self.data.qpos[:len(positions)] = positions
            mujoco.mj_forward(self.model, self.data)

    def wait(self, duration):
        """Wait for specified duration with physics stepping."""
        steps = int(duration / self.model.opt.timestep)
        for _ in range(steps):
            self.step()
            if self.viewer:
                self.viewer.sync()
                
    def wait_for_stability(self, max_time=3.0, velocity_threshold=0.01):
        """Wait until objects are stable or timeout."""
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            # Check if all objects are stable
            all_stable = True
            for obj_name in ["red_block", "blue_block", "green_block"]:
                vel = self.get_object_velocity(obj_name)
                if vel is not None:
                    linear_vel = np.linalg.norm(vel[:3])
                    angular_vel = np.linalg.norm(vel[3:])
                    if linear_vel > velocity_threshold or angular_vel > velocity_threshold:
                        all_stable = False
                        break
            
            if all_stable:
                print(f"[SIM] Objects stabilized in {time.time() - start_time:.2f}s")
                break
                
            self.step()
            if self.viewer:
                self.viewer.sync()
        
        if time.time() - start_time >= max_time:
            print(f"[SIM] Stability timeout after {max_time}s")

    def launch_viewer(self):
        """Launch passive viewer with platform-specific handling."""
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Enhanced camera positioning for better workspace view
            self.viewer.cam.distance = 1.0
            self.viewer.cam.lookat[:] = [0.35, 0, 0.1]
            self.viewer.cam.azimuth = 140
            self.viewer.cam.elevation = -25
            
            # Enable better rendering
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            
            print("[SIM] Viewer launched with enhanced settings")
            
        except RuntimeError as e:
            if "mjpython" in str(e):
                print("[SIM WARNING] Passive viewer requires mjpython on macOS")
                print("[SIM] Run with: python .venv/bin/mjpython main.py")
                print("[SIM] Continuing without viewer...")
                self.viewer = None
            else:
                raise e
                
        return self.viewer

    def close_viewer(self):
        """Close the viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset_scene(self):
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset control state
        self.gripper_state = "open"
        
        # Let physics settle
        self.wait_for_stability()
        
        print("[SIM] Scene reset to initial state")

    def get_contact_info(self):
        """Get contact information for debugging."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            contacts.append({
                'geom1': geom1_name,
                'geom2': geom2_name,
                'pos': contact.pos.copy(),
                'force': contact.frame.copy()
            })
        return contacts
