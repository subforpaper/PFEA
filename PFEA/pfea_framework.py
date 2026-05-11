
import os           
import random      
import threading    
import time         
import cv2
from matplotlib import use
import numpy as np
from PIL import Image
from heapq import nlargest    
import matplotlib.pyplot as plt 
import pybullet    
import pybullet_data  
from urdf_models import models_data
import torch        
if os.path.exists('ur5e/ur5e.urdf'):
    print("加载完成！")
import tempfile
import string
from vlm_planner import *
from vision_utils import *
from agent_utils import * 


PICK_TARGETS = {
  "blue block": None, 
  "red block": None,   
  "green block": None,  
  "yellow block": None,
}


COLORS01 = {
    "blue":   (78/255,  121/255, 167/255, 255/255),  
    "red":    (255/255,  87/255,  89/255, 255/255), 
    "green":  (89/255,  169/255,  79/255, 255/255), 
    "yellow": (237/255, 201/255,  72/255, 255/255),
    'brown': (156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0, 255/255),
}

COLORS = {
    'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    'red': [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
    'green': [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
    'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
    'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
    'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    'brown': [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
    'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0]
}


PLACE_TARGETS = {

  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,


  "blue bowl": None,  
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,


  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),  
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),  
  "middle":              (0,           -0.5,        0),  
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0), 
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}


PIXEL_SIZE = 0.00267857  

BOUNDS = np.float32([
    [-0.3, 0.3],    
    [-0.8, -0.2],   
    [0, 0.15]      
])


class Robotiq2F85:
    """Gripper handling for Robotiq 2F85."""
    def __init__(self, robot, tool):

        self.robot = robot  
        self.tool = tool 

        pos = [0.1339999999999999, -0.49199999999872496, 0.5]  
        rot = pybullet.getQuaternionFromEuler([np.pi, 0, np.pi])  

        urdf = "robotiq_2f_85/robotiq_2f_85.urdf"
        self.body = pybullet.loadURDF(urdf, pos, rot)
        self.n_joints = pybullet.getNumJoints(self.body)  
        self.activated = False  


        pybullet.createConstraint(self.robot, 
                                  tool, 
                                  self.body,  
                                  0,  
                                  jointType=pybullet.JOINT_FIXED,  
                                  jointAxis=[0, 0, 0],  
                                  parentFramePosition=[0, 0, 0],  
                                  childFramePosition=[0, 0, -0.07],  
                                  childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]))  

        for i in range(pybullet.getNumJoints(self.body)):
            pybullet.changeDynamics(self.body,
                                    i,
                                    lateralFriction=10.0, 
                                    spinningFriction=1.0, 
                                    rollingFriction=1.0, 
                                    frictionAnchor=True)  


        self.motor_joint = 1  
        self.constraints_thread = threading.Thread(target=self.step)
        self.constraints_thread.daemon = True  
        self.constraints_thread.start()  


    def step(self):

        while True:
            try:
                currj = [pybullet.getJointState(self.body, i)[0] for i in range(self.n_joints)]
                indj = [6, 3, 8, 5, 10]  
                targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]   
                pybullet.setJointMotorControlArray(self.body,
                                                   indj,
                                                   pybullet.POSITION_CONTROL,  
                                                   targj,
                                                   positionGains=np.ones(5)  
                                                   )
            except:
                return
            time.sleep(0.001)  


    def activate(self):

        pybullet.setJointMotorControl2(self.body,
                                       self.motor_joint,
                                       pybullet.VELOCITY_CONTROL, 
                                       targetVelocity=1, 
                                       force=10  
                                       )
        self.activated = True

    def release(self):

        pybullet.setJointMotorControl2(self.body,
                                       self.motor_joint,
                                       pybullet.VELOCITY_CONTROL,
                                       targetVelocity=-1,  
                                       force=10)
        self.activated = False


    def detect_contact(self):

        obj, _, ray_frac = self.check_proximity()  
        if self.activated:

            empty = self.grasp_width() < 0.01  
            cbody = self.body if empty else obj  
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)



    def external_contact(self, body=None):

        if body is None:
            body = self.body

        pts = pybullet.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]  
        return len(pts) > 0  

    def check_grasp(self):

        while self.moving():
            time.sleep(0.001)  
        success = self.grasp_width() > 0.01  
        return success

    def grasp_width(self):

        lpad = np.array(pybullet.getLinkState(self.body, 4)[0]) 
        rpad = np.array(pybullet.getLinkState(self.body, 9)[0])  
        dist = np.linalg.norm(lpad - rpad) - 0.047813  
        return dist

    def check_proximity(self):

        ee_pos = np.array(pybullet.getLinkState(self.robot, self.tool)[0]) 
        tool_pos = np.array(pybullet.getLinkState(self.body, 0)[0])  
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos)) 
        ee_targ = ee_pos + vec  
        ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]  
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]  
        return obj, link, ray_frac



class PickPlaceEnv():
    def __init__(self):

        self.dt = 1 / 480  
        self.sim_step = 0  

        pybullet.connect(pybullet.GUI)
        
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)  
        
        assets_path = os.path.dirname(os.path.abspath(""))
        pybullet.setAdditionalSearchPath(assets_path)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setTimeStep(self.dt)  

        
        self.home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  
        self.home_ee_euler = (np.pi, 0, np.pi)  
        self.ee_link_id = 9  
        self.tip_link_id = 10  
        self.gripper = None  


    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname
    
    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def reset(self, config):
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)  
        pybullet.setGravity(0, 0, -9.8)   
        self.cache_video = []  

        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        pybullet.loadURDF("plane.urdf", [0, 0, -0.001])  
        self.robot_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)  
        self.ghost_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, 0, -10])  

        self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in range(pybullet.getNumJoints(self.robot_id))]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])

        if self.gripper is not None:
            while self.gripper.constraints_thread.is_alive():
                self.constraints_thread_active = False
        self.gripper = Robotiq2F85(self.robot_id, self.ee_link_id)
        self.gripper.release()  

        plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
        plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
        plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
        pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

        self.config = config
        self.obj_name_to_id = {}
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        obj_xyz = np.zeros((0, 3))  
        for obj_name in obj_names:
            
            if "kit" in obj_name:
                n_objects = 4

                self.train_set = [2,3,10,17]
                obj_shapes = [2,3,10,17]  
                # obj_shapes = [0,1,12,13] 
                colors = [
                    COLORS['purple'], COLORS['blue'], COLORS['green'],
                    COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['cyan'],
                    COLORS['brown'], COLORS['gray'], COLORS['orange'],
                ]
                template = 'g_kitting/object-template.urdf'
                for i in range(n_objects):
                    shape = obj_shapes[i]
                    size = (0.08, 0.08, 0.02)
                    fname = f'{shape:02d}.obj'
                    fname = os.path.join('g_kitting', fname)
                    
                    if shape == 3:
                        scale = [0.0025, 0.0025, 0.0013]
                    else:
                        scale = [0.003, 0.003, 0.0013]
                    
                    replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}
                    urdf = self.fill_template(template, replace)

                    while True:
                        rand_x = np.random.uniform(BOUNDS[0, 0] + 0.1, BOUNDS[0, 1] - 0.1)
                        rand_y = np.random.uniform(BOUNDS[1, 0] + 0.1, BOUNDS[1, 1] - 0.1)
                        rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                        if len(obj_xyz) == 0:
                            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                            break
                        else:
                            nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                            if nn_dist > 0.1:
                                obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                                break

                    object_position = rand_xyz.squeeze()

                    object_id = pybullet.loadURDF(urdf, object_position, useFixedBase=0)

                pybullet.changeVisualShape(object_id, -1)
                self.obj_name_to_id[obj_name] = object_id
                

        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        for _ in range(200):
            pybullet.stepSimulation()
        return self.get_observation()

    def servoj(self, joints):
        """Move to target joint positions with position control."""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.01] * 6)

    def movep(self, position):
        """Move to target end effector position."""
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=pybullet.getQuaternionFromEuler(self.home_ee_euler),
            maxNumIterations=100)  
        self.servoj(joints)  

    def step(self, action=None):
        """Do pick and place motion primitive."""
        pick_xyz, place_xyz = action["pick"].copy(), action["place"].copy()  

        hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.2])  
        pick_xyz[2] = 0.01  
        place_xyz[2] = 0.15

        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
            self.movep(hover_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])


        while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
            self.movep(pick_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])


        self.gripper.activate()
        for _ in range(240*2):
            self.step_sim_and_render() 

        while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
            self.movep(hover_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

 
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])


        while (not self.gripper.detect_contact()) and (place_xyz[2] > 0.08):
            place_xyz[2] -= 0.001
            self.movep(place_xyz)
            for _ in range(3):
                self.step_sim_and_render()
        self.gripper.release()
        for _ in range(240*4):
            self.step_sim_and_render()  
        place_xyz[2] = 0.2
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        place_xyz = np.float32([0, -0.5, 0.2])
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

        observation = self.get_observation()
        reward = self.get_reward()
        done = False
        info = {}
        return observation, reward, done, info

    def set_alpha_transparency(self, alpha: float) -> None:
        for id in range(20):
            visual_shape_data = pybullet.getVisualShapeData(id)
            for i in range(len(visual_shape_data)):
                object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
                rgba_color = list(rgba_color[0:3]) + [alpha]
                pybullet.changeVisualShape(
                    self.robot_id, linkIndex=i, rgbaColor=rgba_color)
                pybullet.changeVisualShape(
                    self.gripper.body, linkIndex=i, rgbaColor=rgba_color)

    def step_sim_and_render(self):
        pybullet.stepSimulation()
        self.sim_step += 1

        if self.sim_step % 60 == 0:
            self.cache_video.append(self.get_camera_image())

    def get_camera_image(self):
        
        image_size = (240, 240)
        intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)

        color, _, _, _, _ = env.render_image(image_size, intrinsics)
        return color

    def get_camera_image_save(self):
        
        image_size = (1560, 1560)
        intrinsics = (1000., 0, 1000., 0, 1000., 1000., 0, 0, 1)

        color, _, _, _, _ = env.render_image_save(image_size, intrinsics)
        return color

    def get_camera_image_save_top(self):
        
        image_size = (1560, 1560)
        intrinsics = (1000., 0, 1000., 0, 1000., 1000., 0, 0, 1)

        color, _, _, _, _ = env.render_image(image_size, intrinsics)
        return color

    def get_camera_image_top(self,
                             image_size=(240, 240),
                             intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                             position=(0, -0.5, 5),
                             orientation=(0, np.pi, -np.pi / 2),
                             zrange=(0.01, 1.),
                             set_alpha=True):
        set_alpha and self.set_alpha_transparency(0)
        color, _, _, _, _ = env.render_image_top(image_size,
                                                 intrinsics,
                                                 position,
                                                 orientation,
                                                 zrange)
        set_alpha and self.set_alpha_transparency(1)
        return color

    def get_reward(self):
        return 0  


    def get_observation(self):

        observation = {}

        color, depth, position, orientation, intrinsics = self.render_image()

        points = self.get_pointcloud(depth, intrinsics)
        position = np.float32(position).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotation = np.float32(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = self.transform_pointcloud(points, transform)
        heightmap, colormap, xyzmap = self.get_heightmap(points, color, BOUNDS, PIXEL_SIZE)


        observation["image"] = colormap
        observation["xyzmap"] = xyzmap
        observation["pick"] = list(self.config["pick"])  
        observation["place"] = list(self.config["place"])  
        return observation

    def render_image(self, image_size=(720, 720), intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1)):

        position = (0, -0.5, 0.4)  
        orientation=(0, np.pi, np.pi)  

        
        orientation = pybullet.getQuaternionFromEuler(orientation)
        zrange=(0.01, 1.)  
        noise = True


        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)


        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))


        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth


        if noise:
            depth += np.random.normal(0, 0.001, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics

    def render_image_save(self, image_size=(720, 720), intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1)):

        position = (0, -0.95, 0.7)
        orientation = (np.pi / 4 + np.pi / 65, np.pi, np.pi)


        orientation = pybullet.getQuaternionFromEuler(orientation)
        zrange=(0.01, 1.)  
        noise = True


        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi


        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)


        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)


        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))


        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth


        if noise:
            depth += np.random.normal(0, 0.001, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics


    def render_image_top(self,
                         image_size=(240, 240),
                         intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                         position=(0, -0.5, 5),
                         orientation=(0, np.pi, -np.pi / 2),
                         zrange=(0.01, 1.)):


        orientation = pybullet.getQuaternionFromEuler(orientation)
        noise = True


        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi


        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)


        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)


        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))


        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics

    def get_pointcloud(self, depth, intrinsics):
        """Get 3D pointcloud from perspective depth image.
        Args:
          depth: HxW float array of perspective depth in meters.
          intrinsics: 3x3 float array of camera intrinsics matrix.
        Returns:
          points: HxWx3 float array of 3D points in camera coordinates.
        """
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    def transform_pointcloud(self, points, transform):
        """Apply rigid transformation to 3D pointcloud.
        Args:
          points: HxWx3 float array of 3D points in camera coordinates.
          transform: 4x4 float array representing a rigid transformation matrix.
        Returns:
          points: HxWx3 float array of transformed 3D points.
        """
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(points.copy(), padding,
                                "constant", constant_values=1)
        for i in range(3):
            points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
        return points

    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
        Args:
          points: HxWx3 float array of 3D points in world coordinates.
          colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
          bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
          pixel_size: float defining size of each pixel in meters.
        Returns:
          heightmap: HxW float array of height (from lower z-bound) in meters.
          colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
          xyzmap: HxWx3 float array of XYZ points in world coordinates.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
        xyzmap = np.zeros((height, width, 3), dtype=np.float32)


        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]


        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
            xyzmap[py, px, c] = points[:, c]
        colormap = colormap[::-1, :, :]  
        xv, yv = np.meshgrid(np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
                             np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width))
        xyzmap[:, :, 0] = xv
        xyzmap[:, :, 1] = yv
        xyzmap = xyzmap[::-1, :, :]  
        heightmap = heightmap[::-1, :]  
        return heightmap, colormap, xyzmap



if 'env' in locals():
    env.gripper.running = False 
    while env.gripper.constraints_thread.isAlive():  
        time.sleep(0.01)
env = PickPlaceEnv()   


config = {
    'pick':  ['kit'],
    'place': []
}


np.random.seed(42)  
obs = env.reset(config)  


plt.subplot(1, 2, 1)
img = env.get_camera_image()

img_file_path = "temp_camera_image.png"  
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(img_file_path, img)    


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = env.get_camera_image_top()
img = np.flipud(img.transpose(1, 0, 2))  



img = env.get_camera_image_save()   

img_file_path = f"temp_camera_image01.png"  
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(img_file_path, img)  


img = env.get_camera_image_save_top()   

img_file_path = f"temp_camera_image02.png"  
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(img_file_path, img)  




def reset_robot():

    home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  
    pybullet.loadURDF("plane.urdf", [0, 0, -0.001])  
    robot_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)  
    ghost_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, 0, -10])  

    joint_ids = [pybullet.getJointInfo(robot_id, i) for i in range(pybullet.getNumJoints(robot_id))]
    joint_ids = [j[0] for j in joint_ids if j[2] == pybullet.JOINT_REVOLUTE]


    for i in range(len(joint_ids)):
        pybullet.resetJointState(robot_id, joint_ids[i], home_joints[i])


def camera_detection(texts, frame_pil, color, points):
    global orange_flag, move, guanjieyi


    results = detect_objects(texts, frame_pil)
    

    if results[0]['scores'].numel() > 0:
        max_score_index = torch.argmax(results[0]['scores'])
    else:
        print("张量为空，无法找到最大值的索引。")
        return 

    box, score, label = results[0]["boxes"][max_score_index], results[0]["scores"][max_score_index], results[0]["labels"][max_score_index]

    if label is not None:

            box = [int(i) for i in box.tolist()] 
            cv2.rectangle(color, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) 
            cv2.putText(color, f"{texts[0][label]}: {score.item():.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)  
            mid_x = (box[0] + box[2]) / 2

            mid_y = (box[1] + box[3]) / 2

            mid_point = [mid_x, mid_y]
            print(mid_point)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  

            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
            move = 1

            if move == 1:
                depth_pixel = [int(mid_point[0]), int(mid_point[1])]  

                target_xyz = points[depth_pixel[1], depth_pixel[0]]  

              
                print('depth: ', target_xyz[2])  
                
                return [target_xyz[0], target_xyz[1], target_xyz[2]]


action = {
    "pick": None,  
    "place": None  
}
def vla_move(start, end):

    global action

    env.set_alpha_transparency(0)

    color, depth, position, orientation, intrinsics = env.render_image()
    env.set_alpha_transparency(1)


    points = env.get_pointcloud(depth, intrinsics)
    position = np.float32(position).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.float32(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    points = env.transform_pointcloud(points, transform)
  
    frame_pil = Image.fromarray(color)


    start01 = camera_detection(start, frame_pil, color, points)
    action["pick"] = start01  
    print("start01",start01)

    end01 = camera_detection(end, frame_pil, color, points)
    action["place"] = end01
    print("end01",end01)
    print("action",action)
    env.step(action)  


def agent_play(start_record_ok):

    if str.isnumeric(start_record_ok):
      
        order = start_record_ok 
    else:
    
        order = start_record_ok
  
    re = agent_plan(order)
    json_str_without_json = re.replace('json', '')
    json_str_without_json = json_str_without_json.replace('```', '')
    agent_plan_output = eval(json_str_without_json)
    print('智能体编排动作如下\n', agent_plan_output)

    plan_ok = 'c'
    if plan_ok == 'c':
        for each in agent_plan_output['function']:  
            print('开始执行动作', each)
            eval(each)
        response = agent_plan_output['response'] 
        print(response) 

    elif plan_ok == 'q':
        raise NameError('按q退出')


# 本场景是场景二，seen和unseen任务：
# Stack objects in descending order based on the number of corners
# Stack the objects in ascending order based on the number of sides
if __name__ == '__main__':
    while True:
        TEXT = input('Input: \n')
        ok = 0
        while TEXT and ok == 0:
            re = vlm_coOR(TEXT, img_file_path)
            json_str_without_json = re.replace('json', '')
            json_str_without_json = json_str_without_json.replace('```', '')
            agent_plan_output = eval(json_str_without_json)
            for i, each in enumerate(agent_plan_output['function']):  
                print('执行动作：', each)
                agent_play(each)

                img = env.get_camera_image_save()   

                img_file_path = f"temp_camera_image_{i}.png"  
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_file_path, img) 
            

            env.set_alpha_transparency(0)
            img = env.get_camera_image()  
            env.set_alpha_transparency(1) 

            img_file_path = "temp_camera_image.png"  
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_file_path, img) 


            PROMPT_back = PROMPT_back01
            TEXT = ' '
            re = vlm_ROF(PROMPT_back, TEXT, img_file_path)
            json_str_without_json = re.replace('json', '')
            json_str_without_json = json_str_without_json.replace('```', '')
            agent_plan_output = eval(json_str_without_json)
            for each in agent_plan_output['function']:  
                if each == "任务完成":
                    print('任务完成')
                    ok = 1
                    break
                else:
                    print('任务未完成')
                    ok = 0

