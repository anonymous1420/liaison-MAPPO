import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import wandb
import torch


class UAVEnv(gym.Env):
    def __init__(self, config):
        super(UAVEnv, self).__init__()
        self.sum=0
        self.num_uavs = config.num_agents
        self.num_red = 7
        self.num_blue = 7
        self.max_steps = config.max_steps
        self.step_count = 0
        self.rnd=0
        self.base_hit_probability = 0.6
        # 定义动作和观测空间的上下限
        obs_low = np.array([0, 0, -0.2, -0.5236,-1], dtype=np.float32)
        obs_high = np.array([100, 100, 0.2, 0.5236,1], dtype=np.float32)
        action_bounds = np.array([-0.2, -0.5236], dtype=np.float32), np.array([0.2, 0.5236], dtype=np.float32)

        # 创建观测和动作空间
        self.individual_observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.individual_action_space = spaces.Box(low=action_bounds[0], high=action_bounds[1], dtype=np.float32)
        obs_low_all = np.tile(obs_low, self.num_uavs)  # 将 obs_low 重复 14 次
        obs_high_all = np.tile(obs_high, self.num_uavs)  # 将 obs_high 重复 14 次

        # individual_cent_observation_space 是 14 个智能体全局的观测空间
        self.individual_cent_observation_space = spaces.Box(
            low=obs_low_all,
            high=obs_high_all,
            dtype=np.float32
        )

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for idx in range(self.num_uavs):
            self.observation_space.append(self.individual_observation_space)
            self.share_observation_space.append(self.individual_cent_observation_space)

        for idx in range(self.num_red):
            self.action_space.append(self.individual_action_space)
        self.uav_state = np.zeros((14, 5), dtype=np.float32)
        self.reset()

        self.cent_obs_space = self._get_observation()
        self.obs1_space = np.concatenate([self._get_obs(0), self._get_obs(1), self._get_obs(2)])
        self.obs2_space = np.concatenate([self._get_obs(1), self._get_obs(3), self._get_obs(4)])
        self.obs3_space = np.concatenate([self._get_obs(2), self._get_obs(5), self._get_obs(6)])
        self.obs_space = self.cent_obs_space
        # Setup for visualization with pygame
        # pygame.init()
        # self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        # pygame.display.set_caption("UAV Environment")
        # self.clock = pygame.time.Clock()

    def reset(self):
        for i in range(7,14):
            self.uav_state[i, 0] = np.random.uniform(70, 100)  # x position
            self.uav_state[i, 1] = np.random.uniform(70,100)  # y position
            self.uav_state[i, 2] = np.random.uniform(5, 10)  # speed
            self.uav_state[i, 3] = np.random.uniform(-np.pi, np.pi)  # theta
            self.uav_state[i, 4] = 1  # alive
        for i in range(7):
            self.uav_state[i, 0] = np.random.uniform(0, 0)  # x position
            self.uav_state[i, 1] = np.random.uniform(30, 30)  # y position
            self.uav_state[i, 2] = np.random.uniform(5, 10)  # speed
            self.uav_state[i, 3] = np.random.uniform(-np.pi, np.pi)  # theta
            self.uav_state[i, 4] = 1  # alive
        self.sum = 0
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):
        # 只提取 x 和 y 位置（第 0 列和第 1 列）
        xy_positions = self.uav_state[:, :]
        # 将 x 和 y 位置展平成一维数组
        return xy_positions

    def _get_obs(self, i):
        obs_space = np.array([self.uav_state[i,0],self.uav_state[i,1],
                             self.uav_state[i,2], self.uav_state[i,3],self.uav_state[i,4]])
        return obs_space

    def _check_status(self):
        done = False
        reward = np.zeros(7)

        # Attack angle in radians (20 degrees)
        attack_angle = np.deg2rad(20)
        observe_angle = np.deg2rad(35)

        for i in range(7):
            for j in range(7, 14):
                if self.uav_state[i, 4] == 1 and self.uav_state[j, 4] == 1:
                    distance = np.sqrt((self.uav_state[i, 0] - self.uav_state[j, 0]) ** 2 +
                                       (self.uav_state[i, 1] - self.uav_state[j, 1]) ** 2)
                    if distance < 5:
                        # 对两个 UAV 进行相对角度的计算
                        for target, attacker in [(j, i), (i, j)]:
                            dx = self.uav_state[target, 0] - self.uav_state[attacker, 0]
                            dy = self.uav_state[target, 1] - self.uav_state[attacker, 1]
                            angle_to_target = np.arctan2(dy, dx)
                            relative_angle = np.abs(angle_to_target - self.uav_state[attacker, 3])
                            relative_angle = np.minimum(relative_angle, 2 * np.pi - relative_angle)

                            if relative_angle < attack_angle:  # 检查目标是否在攻击角范围内
                                hit_probability = self.base_hit_probability
                                if np.random.rand() > hit_probability:
                                    self.uav_state[target, 4] = 0
                                    if attacker <7 :
                                        reward[attacker] += 20
                                    else:
                                        reward[target] -= 20
        done = False
        red_win = False
        blue_win = False
        base_alive = True
        for t in range(7,14):
            if self.uav_state[t,4]==1:
                if -100<=self.uav_state[t,0]<=-90 and -10<=self.uav_state[t,1]<=10:
                    base_alive=False
                distance = np.sqrt((self.uav_state[t, 0] -95) ** 2 + (self.uav_state[t, 1]) ** 2)
                for r in range(0,7):
                    if self.uav_state[r,4]==1:
                        reward[r] += (distance-100)/1000
        if self.step_count >= self.max_steps:
            done = True
            if base_alive:
                red_win = True
            else:
                blue_win = True
        else:
            if not base_alive:
                blue_win = True
                done = True
            elif np.sum(self.uav_state[7:14, 4]) == 0:
                red_win = True
                done = True
        reward = reward.reshape(7,1)
        return done, red_win, blue_win, reward

    def step(self, actions):
        self.step_count += 1
        #screen_width, screen_height = self.screen.get_size()
        for i in range(self.num_blue):
            if self.uav_state[i, 4] == 1:  # Only update if UAV is alive
                v, delta_theta = actions[i]
                self.uav_state[i, 2] = np.clip(v, 0, 5) #speed
                self.uav_state[i, 3] +=np.clip(delta_theta, -np.pi/6, np.pi/6)
                if 0<= self.uav_state[i, 0] + self.uav_state[i, 2] * np.cos(self.uav_state[i, 3]) <=100:
                    self.uav_state[i, 0] += self.uav_state[i, 2] * np.cos(self.uav_state[i, 3])
                if 0<=  self.uav_state[i, 1] + self.uav_state[i, 2] * np.sin(self.uav_state[i, 3]) <=100:
                    self.uav_state[i, 1] += self.uav_state[i, 2] * np.sin(self.uav_state[i, 3])

        # 为 7-13 号智能体生成随机动作
        for i in range(7, 14):
            if self.uav_state[i, 4] == 1:  # 仅在 UAV 存活时执行随机动作
                random_v = np.random.uniform(0, 5)  # 随机速度
                random_delta_theta = np.random.uniform(-np.pi/6, np.pi/6)  # 限制为 ±30 度
                self.uav_state[i, 2] = random_v
                self.uav_state[i, 3] = (self.uav_state[i, 3]+random_delta_theta) % (2 * np.pi)
                if 0 <= self.uav_state[i, 0] + self.uav_state[i, 2] * np.cos(self.uav_state[i, 3]) <= 100:
                    self.uav_state[i, 0] += self.uav_state[i, 2] * np.cos(self.uav_state[i, 3])
                if 0 <= self.uav_state[i, 1] + self.uav_state[i, 2] * np.sin(self.uav_state[i, 3]) <= 100:
                    self.uav_state[i, 1] += self.uav_state[i, 2] * np.sin(self.uav_state[i, 3])
        done, red_win, blue_win, reward = self._check_status()
        a = reward.sum()
        self.sum = self.sum + a
        info = {}
        #self.sum = (self.sum*(self.step_count-1)+a)/self.step_count
        if done is True and red_win:
            #print('average rewards: {}'.format(self.sum))
            self.rnd+=1
            info = {'round rewards': self.sum, 'step_average rewards':self.sum/self.step_count,'winner': 1, 'round':self.rnd, 'round_step':self.step_count}
            #wandb.log({'red average rewards': self.sum, 'winner': 1, 'round':self.rnd, 'round_step':self.step_count})
        elif done is True and blue_win:
            self.rnd += 1
            #print('average rewards: {}'.format(self.sum))
            info = {'round rewards': self.sum, 'step_average rewards':self.sum/self.step_count,'winner': 0, 'round':self.rnd, 'round_step':self.step_count}
            #wandb.log({'red average rewards': self.sum, 'winner': 0, 'round':self.rnd, 'round_step':self.step_count})
        cent_obs = self._get_observation()
        obs = np.array([self.obs1_space,self.obs1_space,self.obs1_space,self.obs2_space,self.obs2_space,self.obs3_space,self.obs3_space])
        masks = np.ones((7, 1), dtype=np.integer)
        for i in range(7):
            if self.uav_state[i, 4] == 0:
                masks[i] = 0
        #self.render()
        if done is True:
            self.reset()
        return cent_obs, reward, done, info

    # def render(self, mode='human'):
    #     self.screen.fill((255, 255, 255))  # Clear screen
    #     # Define airplane shape (triangle)
    #     airplane_shape = np.array([[0, -10], [5, 5], [-5, 5]])  # Simple triangle shape
    #     for i in range(self.num_red):
    #         if self.uav_state[i, 4] == 1:  # If UAV is alive
    #             x = int(self.uav_state[i, 0] * 8)  # Scale to fit screen
    #             y = int(self.uav_state[i, 1] * 6)  # Scale to fit screen
    #             level = self.uav_state[i, 5]  # UAV level
    #             color_intensity = min(max(int(255 * level / 3), 0), 255)  # Ensure within range 0-255
    #             # Calculate rotated position of the airplane
    #             rotated_shape = []
    #             for point in airplane_shape:
    #                 # Rotate point around the UAV's position
    #                 rotated_x = point[0] * np.cos(self.uav_state[i, 3]) - point[1] * np.sin(self.uav_state[i, 3])
    #                 rotated_y = point[0] * np.sin(self.uav_state[i, 3]) + point[1] * np.cos(self.uav_state[i, 3])
    #                 # Translate to UAV position
    #                 rotated_shape.append((x + rotated_x, y + rotated_y))
    #             # Draw the airplane
    #             pygame.draw.polygon(self.screen, (255, 0, 0), rotated_shape)
    #     for i in range(self.num_blue):
    #         if self.uav_state[self.num_red + i, 4] == 1:
    #             x = int(self.uav_state[self.num_red + i, 0] * 8)  # Scale to fit screen
    #             y = int(self.uav_state[self.num_red + i, 1] * 6)  # Scale to fit screen
    #             level = self.uav_state[self.num_red + i, 5]  # UAV level
    #             color_intensity = min(max(int(255 * level / 3), 0), 255)  # Ensure within range 0-255
    #             # Calculate rotated position of the airplane
    #             rotated_shape = []
    #             for point in airplane_shape:
    #                 rotated_x = point[0] * np.cos(self.uav_state[self.num_red + i, 3]) - point[1] * np.sin(
    #                     self.uav_state[self.num_red + i, 3])
    #                 rotated_y = point[0] * np.sin(self.uav_state[self.num_red + i, 3]) + point[1] * np.cos(
    #                     self.uav_state[self.num_red + i, 3])
    #                 rotated_shape.append((x + rotated_x, y + rotated_y))
    #             # Draw the airplane
    #             pygame.draw.polygon(self.screen, (0, 0, 255), rotated_shape)
    #     pygame.display.flip()
    #     self.clock.tick(30)  # Control the frame rate

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)






