import metaworld
import random
import glfw
import time
import numpy as np
import mujoco
import cv2
import json
import imageio
import os

from metaworld.policies import *
from tqdm import tqdm

policy_dict = {
    'window-open-v2': SawyerWindowOpenV2Policy,
    'soccer-v2': SawyerSoccerV2Policy,
    'hammer-v2': SawyerHammerV2Policy,
    'drawer-open-v2': SawyerDrawerOpenV2Policy,
}

cams = [
    'cam_1_1',
    'cam_7_4',
    'cam_14_2',
]

cam_fovy = 90.0 # field of view of all cams

base_semantic_mapping = {
    -1: 0, # background
    28: 1, # sawyerbase
    29: 1, # sawyergrip
    31: 1, # sawyergrip
    33: 1, # sawyergrip
}

soccer_semantic_mapping = {
    36: 2, # ball white
    37: 2, # ball black
    39: 3, # goal frame
    40: 3, # goal net
}

hammer_semantic_mapping = {
    36: 2, # hammer handle
    37: 3, # hammer head
    38: 3, # hammer head
    39: 3, # hammer head
    40: 3, # hammer head
    41: 3, # hammer head
    48: 4, # wood box
    53: 5, # nail
}

drawer_semantic_mapping = {
    43: 2, # drawer handle
    42: 3, # drawer movable
    36: 4, # drawe frame
}

window_semantic_mapping = {
    48: 2, # right window frame
    61: 2, # left window frame
    36: 2, # window frame bottom
    37: 2, # window frame all
    43: 3, # window handle
    44: 3, # window handle
    45: 3, # window handle
    46: 3, # window handle
    47: 3, # window handle
}

window_semantic_mapping.update(base_semantic_mapping)
soccer_semantic_mapping.update(base_semantic_mapping)
hammer_semantic_mapping.update(base_semantic_mapping)
drawer_semantic_mapping.update(base_semantic_mapping)

semantic_mapping = {
    'window-open-v2': window_semantic_mapping,
    'soccer-v2': soccer_semantic_mapping,
    'hammer-v2': hammer_semantic_mapping,
    'drawer-open-v2': drawer_semantic_mapping,
}


def generate_dataset(env_name, out_dir, num_ep=120, timestep_per_ep=120, resolution=(128, 128)):
    if env_name not in policy_dict:
        raise NotImplementedError(f'env_name {env_name} not implemented')

    ml = metaworld.ML1(env_name) # Construct the benchmark, sampling tasks
    env = ml.train_classes[env_name]()  # Create an environment with the corresponding task
    policy = policy_dict[env_name]()  # Instantiate a policy
    env.sim.model.vis.quality.offsamples = 0 # disable anti-aliasing so that the segmentation is clean

    def init_env():
        task = random.choice(ml.train_tasks)
        env.set_task(task)  # Set task
        obs = env.reset()  # Reset environment
        return obs
    
    obs = init_env()
    for ep in tqdm(range(num_ep)):
        for t in range(timestep_per_ep):
            # a = env.action_space.sample()  # Sample a random action
            a = policy.get_action(obs)  # Get an action from the policy
            obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
            data = {
                'camera_angle_x': cam_fovy,
                'frames': []
            }
            file_path = f'e_{ep:03d}_t_{t:03d}'

            for cam in cams:
                rgb = env.render(camera_name=cam, resolution=resolution, offscreen=True, segmentation=False)
                semantic = env.render(camera_name=cam, resolution=resolution, offscreen=True, segmentation=True)[:, :, 1]
                cam_rot = env.sim.data.get_camera_xmat(cam)
                cam_pos = env.sim.data.get_camera_xpos(cam)
                cam_mat = np.concatenate([cam_rot, cam_pos.reshape(3, 1)], axis=1)

                semantic_pixels = np.zeros_like(semantic)
                for key, val in semantic_mapping[env_name].items():
                    semantic_pixels[semantic == key] = val * 50
                semantic_pixels = semantic_pixels.astype(np.uint8)
                
                img_file_path = f'{file_path}_{cam}'
                rgb_path = f'{img_file_path}.jpg'
                semantic_path = f'{img_file_path}_semantic.jpg'
                imageio.imsave(os.path.join(out_dir, rgb_path), rgb)
                imageio.imsave(os.path.join(out_dir, semantic_path), semantic_pixels)
                data['frames'].append({
                    'file_path': img_file_path,
                    'transform_matrix': cam_mat.tolist(),
                })
            
            with open(os.path.join(out_dir, f'{file_path}.json'), 'w') as fp:
                json.dump(data, fp)

        obs = init_env()  # Reset environment after each episode

if __name__ == '__main__':
    generate_dataset('window-open-v2', '/media/steve/hdd/Dataset/SNERL/window-open-v2', resolution=(800, 800))
    # generate_dataset('