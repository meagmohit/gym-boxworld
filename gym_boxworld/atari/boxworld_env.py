import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep
from  boxworld_graphics import *

# Action Codes: 0,1,2,3 : Up, Right, Left and Down respectively
# Stimulation code: [0, agent_x, agent_y, agent_key, agent_prev_x, agent_prev_y, agent_prev_key, action]

# [start_key,. .. mid keys, losing key, winning key]
key_locs = [[0,1],[4,4],[2,5],[3,2],[5,1]]


class ALEInterface(object):
    def __init__(self):
      self._lives_left = 0

    def lives(self):
      return 0 #self.lives_left

class BoxWorldEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}

    def __init__(self, map_size=6, total_keys=5, tcp_tagging=False, tcp_port=15361):

        # Atari-platform related parameters
        self._atari_dims = (210,160,3)		# Specifies standard atari resolution
        (self._atari_height, self._atari_width, self._atari_channels) = self._atari_dims

        #  Game-related paramteres
        self._screen_height = map_size + 4
        self._screen_width = map_size + 4
        self._total_keys = total_keys
        self._map_size = map_size
        self._screen_dims = [self._screen_height, self._screen_width]

        self._map = np.zeros((self._map_size, self._map_size), dtype=np.int32)

        self._agent_pos = None  # agent position
        self._agent_key = None  # key number in agent's hand

        # Sanity Checks
        self._actions = [[-1,0],[0,1],[0,-1],[1,0]]     # Up, Right, Left, DOWN
        self._score = 0.0
        self._state = None  # [agent_x, agent_y, agent_key]

        self._gem = 5
        self._dead_set = [4]
        self._agent_init_pos = None
        self._steps_beyond_done = None

        # Gym-related variables [must be defined]
        self.action_set = np.array([0,1,2,3],dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._atari_height, self._atari_width, 3), dtype=np.uint8)
        self.viewer = None

        # Display based variables
        # self._offset = 25  # need to display square screen, 210x160 -> 160x160, so 25 pixels
        self._color_box = [[128, 128, 128], [0, 0, 200], [128, 0, 128], [192, 192, 32], [200, 0, 0], [0, 255, 0], [0, 255, 255],
                          [255, 0, 255], [128, 128, 255], [128, 0, 128], [255, 128, 128]]
        self._color_back = [255, 255, 255]  # background color
        self._color_agent, self._color_gem = None, [0, 0, 0]
        self._offset = 25  # need to display square screen, 210x160 -> 160x160, so 25 pixels

        # Code for TCP Tagging
        self._tcp_tagging = tcp_tagging
        if (self._tcp_tagging):
            self._host = '127.0.0.1'
            self._port = tcp_port
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._s.connect((self._host, self._port))

        # Methods
        self._ale = ALEInterface()
        self.seed()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        if isinstance(action, np.ndarray):
          action = action[0]
        assert self.action_space.contains(action)   # makes sure the action is valid

        # Updating the state, state is hidden from observation
        [agent_x, agent_y, agent_key, agent_prev_x, agent_prev_y, agent_prev_key, prev_action] = self._state
        agent_prev_x = agent_x
        agent_prev_y = agent_y
        agent_prev_key = agent_key

        # reward definition
        reward_step = -0.5  # one step movement
        reward_invalid = -1  # movement beyond boundary or opening wrong lock
        reward_open = 1.  # next lock is opened
        reward_fail = -10.  # game fail
        reward_done = 10.  # game success

        current_action = self._actions[action]
        ghost_x = agent_x + current_action[0]
        ghost_y = agent_y + current_action[1]

        reward, done = 0.0, False
        # check that action is valid or not
        if action == 0 and ghost_x < 0: #up
            reward = reward_invalid
        elif action == 1 and ghost_y > self._map_size - 1: #right
            reward = reward_invalid
        elif action == 2 and ghost_y < 0: # left
            reward = reward_invalid
        elif action == 3 and ghost_x > self._map_size - 1: # down
            reward = reward_invalid
        # check that intended grid is empty
        elif self._map[ghost_x, ghost_y ] == -1:
            agent_x, agent_y = ghost_x, ghost_y
            reward = reward_step
        elif  (ghost_y  > 0) and (self._map[ghost_x, ghost_y] == self._agent_key) and (self._map[ghost_x, ghost_y -1] > -1):
            agent_x, agent_y = ghost_x, ghost_y - 1
            self._agent_key = self._map[ghost_x, ghost_y - 1]
            agent_key = self._agent_key
            self._color_agent = self._color_box[self._agent_key]
            if self._agent_key == self._gem:
                reward, done = reward_done, True
                print("Game Finished Successfully")
            elif self._agent_key in self._dead_set:
                reward, done = reward_fail, True
                print("Agent reaches dead end!")
            else:# lock is open
                print("One More Lock Opened")
                self._map[agent_x, agent_y+1] = -1
                self._map[agent_x, agent_y] = -1
                reward = reward_open


        self._score = self._score + reward
        self._state = [agent_x, agent_y, agent_key, agent_prev_x, agent_prev_y, agent_prev_key, action]
        print self._state

        # Sending the external stimulation over TCP port
        if self._tcp_tagging:
            padding=[0]*8
            event_id = [0, agent_x, agent_y, agent_key, agent_prev_x, agent_prev_y, agent_prev_key, action]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self._s.sendall(bytearray(padding+event_id+timestamp))


        return self._get_observation(), reward, done, {"ale.lives": self._ale.lives(), "internal_state": self._state}

    def deploy_box(self, k, b_x, b_y, color_key, color_content):
        self._map[b_x, b_y] = color_key
        self._map[b_x, b_y - 1] = color_content

    def reset(self):

        # generate the box map
        self._map = np.zeros((self._map_size, self._map_size), dtype=np.int32) - 1  # -1 denotes empty on the map
        self.deploy_box(0, key_locs[0][0], key_locs[0][1], 0, 1)  # box 0
        self.deploy_box(1, key_locs[1][0], key_locs[1][1], 1, 2)  # box 1
        self.deploy_box(2, key_locs[2][0], key_locs[2][1], 2, 3)  # box 2
        self.deploy_box(3, key_locs[3][0], key_locs[3][1], 3, 4)  # box 3
        self.deploy_box(4, key_locs[4][0], key_locs[4][1], 3, 5)  # box 4

        # randomly select agent's starting position
        agent_pos = np.random.choice(np.arange(self._map_size ** 2))
        agent_x, agent_y = int(agent_pos/self._map_size), agent_pos % self._map_size

        while self._map[agent_x, agent_y] > -1:
            agent_pos = np.random.choice(np.arange(self._map_size ** 2))
            agent_x, agent_y = int(agent_pos / self._map_size), agent_pos % self._map_size

        self._agent_key = 0
        self._color_agent = self._color_box[0]

        # define score and state
        self._score = 0.0
        self._state = [agent_x, agent_y, self._agent_key, agent_x, agent_y, self._agent_key, -1]
        return self._get_observation()


    def _get_observation(self, dstate=0):
        img = 255*np.ones(self._atari_dims, dtype=np.uint8) # White screen
        block_width = int(self._atari_width/self._screen_width)
        [agent_x, agent_y, agent_key, agent_prev_x, agent_prev_y, agent_prev_key, prev_action] = self._state

        boundary_offset, game_offset = 1, 2
        #boundary
        for id in range(0,self._map_size+2):
            idx_x, idx_y = id+boundary_offset, 0+boundary_offset
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 200*arr_brick
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = 0
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 0
            idx_x, idx_y = id+boundary_offset, self._map_size+2-1+boundary_offset
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 200*arr_brick
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = 0
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 0
            idx_x, idx_y = 0+boundary_offset, id+boundary_offset
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 200*arr_brick
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = 0
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 0
            idx_x, idx_y = self._map_size+2-1+boundary_offset, id+boundary_offset
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 200*arr_brick
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = 0
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 0

        # dstate=0 : agent is displayed
        # dstate=1 : agent is not displayed if same position
        # agent drawing
        if dstate==0 or (not((agent_x == agent_prev_x) and (agent_y == agent_prev_y)) and dstate==1):
            idx_x, idx_y = agent_x+game_offset, agent_y+game_offset
            x_c, y_c = idx_x*block_width + self._offset, idx_y*block_width
            for idx_x in range(arr_ghost.shape[0]):
                    for idx_y in range(arr_ghost.shape[1]):
                        if arr_ghost[idx_x, idx_y] == 0:    # Boundary
                            img[x_c+idx_x, y_c+idx_y, :] = 0
                        elif arr_ghost[idx_x, idx_y] == 1:    # Body
                            img[x_c+idx_x, y_c+idx_y, 2] = self._color_box[agent_key][2]
                            img[x_c+idx_x, y_c+idx_y, 0] = self._color_box[agent_key][0]
                            img[x_c+idx_x, y_c+idx_y, 1] = self._color_box[agent_key][1]
                            # img[x_c+idx_x, y_c+idx_y, 1] = 0
                        elif arr_ghost[idx_x, idx_y] == 2:    # Eyes
                            img[x_c+idx_x, y_c+idx_y, 1] = 0
                            img[x_c+idx_x, y_c+idx_y, 0] = 0
                            img[x_c+idx_x, y_c+idx_y, 2] = 0
                        else:   # background
                            img[x_c+idx_x, y_c+idx_y, :] = 255

        # Display all other blocks/locks/keys
        for key_id in range(agent_key, self._total_keys):
            idx_x, idx_y = key_locs[key_id][0]+game_offset, key_locs[key_id][1]+game_offset
            if key_id == self._total_keys - 1:
                color_id = key_id - 1
            else:
                color_id = key_id
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = self._color_box[color_id][0]
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = self._color_box[color_id][1]
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = self._color_box[color_id][2]

            idx_x, idx_y = key_locs[key_id][0]+game_offset, key_locs[key_id][1]-1+game_offset
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = self._color_box[key_id+1][0]
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = self._color_box[key_id+1][1]
            img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = self._color_box[key_id+1][2]

        return img

    def render(self, mode='human', close=False, dstate=0):
        img = self._get_observation(dstate)
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=1920)
            self.viewer.imshow(np.repeat(np.repeat(img, 5, axis=0), 5, axis=1))
            return self.viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(BoxWorldEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._tcp_tagging:
            self._s.close()

    def _get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.action_set]

    @property
    def _n_actions(self):
        return len(self.action_set)

    def seed(self, seed=None):
        self._np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    # A function for TCP_tagging in openvibe
    # transform a value into an array of byte values in little-endian order.
    def to_byte(self, value, length):
        for x in range(length):
            yield value%256
            value//=256


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self._get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


ACTION_MEANING = {
    0 : "UP",
    1 : "RIGHT",
    2 : "LEFT",
    3 : "DOWN",
}
