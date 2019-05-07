import math
import numpy as np
import gym
import copy
from gym import spaces
from gym.utils import seeding


class gameObject():
    def __init__(self, coordinates, radius, color, reward, name):
        self.coordinates = coordinates
        self.radius = radius
        self.color = color
        self.reward = reward
        self.name = name


class HypercubeEnv(gym.Env):
    """
    Description:
        An agent is in a square (2 dimension hypercube). The goal is to collect
         the maximum possible reward via preferring real target object upon fake
         objects. This is doneby increasing and reducing his velocity in the n
         directions.

    Observation: 
        Type: Box(2n)
        Num     Observation                         Min     Max
        0       Agent 1st direction position        -1      1
        1       Agent 2nd direction position        -1      1
        ...     ...                                 ...     ...
        n-1     Agent nth direction position        -1      1
        n       Agent 1st direction velocity        -0.1    0.1
        ...     ...                                 ...     ...
        2n-2    Agent (n-1)th direction velocity    -0.1    0.1
        2n-1    Agent nth direction velocity        -0.1    0.1

    Actions:
        Type: Discrete(2n+1)
        Num     Action
        0       Don't Accelerate the agent
        1       Accelerate the agent to -1 orientation in the 1st direction
        2       Accelerate the agent to 1 orientation in the 1st direction
        3       Accelerate the agent to -1 orientation in the 2nd direction
        4       Accelerate the agent to 1 orientation in the 2nd  direction
        ...     ...
        2n-1    Accelerate the agent to -1 orientation in the nth direction
        2n      Accelerate the agent to 1 orientation in the nth direction

    Reward:
        Reward is 1 if the agent find the green (n-1)-face
        Reward is 0.1 if the agent find a red (n-1)-face

    Starting State
        Agent 1st direction position is assigned a uniform random value between -0.6 and-0.4
        Agent 2nd direction position is assigned to 0
        ...
        Agent nth direction position is assigned to 0
        Agent 1st direction velocity is assigned to 0
        Agent 2nd direction velocity is assigned to 0
        ...
        Agent nth direction velocity is assigned to 0

    Episode Termination:
        Agent find any (n-1)-face
    """

    metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 30
            }

    """ Default environement description , to be customized by the user """

    default_description = {'high_reward_value': 1,
                           'low_reward_value': 0.1,
                           'high_reward_count': 'half',
                           'low_reward_count': 'half',
                           'mode': 'deterministic',
                           'agent_starting': 'fixed',
                           'speed_limit_mode': 'vector_norm'}

    def __init__(self, n_dimensions=2, env_description=default_description, continuous=True,acceleration=True, reset_radius=None):

        """ Sanity checks """
        # if n_dimensions <= 0:
        #    raise ValueError('Number of dimension must be strictly positive')

        if self._validate_env_description(env_description) is False:
            raise ValueError("Missing descriptors in environement description")

        self.env_description = env_description

        self.n = n_dimensions
        self.objects = None
        self.continuous = continuous
        self.acceleration = acceleration
        self.reset_radius = reset_radius

        self.max_position = 1
        self.max_velocity = 0.1

        if self.acceleration:
            self.high_observation = np.r_[
                np.ones(self.n)*self.max_position,
                np.ones(self.n)*self.max_velocity,
                np.ones(self.n) * self.max_position,
                np.ones(self.n) * self.max_velocity
                ]

        else:
            self.high_observation = np.ones(self.n)*self.max_position

        self.low_observation = -self.high_observation

        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation, dtype=np.float32)
        if self.continuous:
            self.max_action = 1
            self.high_action = np.ones(self.n)*self.max_action
            self.low_action = -self.high_action
            self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        else: # if discrete
            self.action_space = spaces.Discrete(2*self.n+1)

        if self.acceleration:
            self.power = 0.01

        self.friction = 0.001
        self.nb_target = env_description['nb_target']
        self.high_reward = env_description['high_reward_value']
        self.low_reward = env_description['low_reward_value']
        self.agent_starting = env_description['agent_starting']
        if self.nb_target == 1:
            self.target_position = np.asarray([0.6, 0.6])
        else:
            self.target_position = None
        self.action_cost = 0.01
        self._max_episode_steps = 100
        self._current_episode_step = 0
        self.position = None
        self.velocity = None
        self.viewer = None
        self.state = None
        self.seed()

        """ Keep IDs of dimensions used for environement description """
        self.wall_position = np.array([]).reshape(-1, 2)
        self.wall_position = np.vstack((self.wall_position, [(0, self.max_position), (1, self.max_position), (0, -self.max_position), (1, -self.max_position)]))
        if self.reset_radius is None:
            self.reset_radius = np.sqrt(self.n * (self.max_position ** 2))

        elif self.reset_radius < 0 or abs(self.reset_radius) > self.max_position:
            raise ValueError('Reset radius must be a positive number between 0 and max_position ({})'.format(self.max_position))

        # self.reset()

    def _validate_env_description(self, env_description):

        if env_description.get("high_reward_value") is None:
            return False
        if env_description.get("low_reward_value") is None:
            return False
        if env_description.get("nb_target") is None:
            return False
        if env_description.get("mode") is None:
            return False
        if env_description.get("speed_limit_mode") is None:
            return False

        return True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _discrete_velocity_step(self, state, action):
        if action == 0:
            orientation = 1
            direction = -1

        else:
            action -= 1
            if action % 2 == 0:
                orientation = -1
            elif action % 2 == 1:
                orientation = 1
            direction = action//2

        if direction > -1:
            velocity[direction] = orientation*self.max_velocity
        
        return self.state, velocity

    def _continuous_velocity_step(self, state, action):
        return state, action*self.max_velocity

    def _discrete_acceleration_step(self, state, action):
        position = state[:state.shape[0]//2]
        velocity = state[state.shape[0]//2:]

        if action == 0:
            orientation = 1
            direction = -1

        else:
            action -= 1
            if action % 2 == 0:
                orientation = -1
            elif action % 2 == 1:
                orientation = 1
            direction = action//2

        if direction > -1:
            accel = orientation*self.power
            velocity[direction] += accel
        velocity = self._apply_friction(state, velocity)
        for direction in range(self.n):
            velocity[direction] = np.clip(velocity[direction], -self.max_velocity, self.max_velocity)

        return position, velocity

    def _continuous_acceleration_step(self, state, action):
        position = state[:state.shape[0]//2]
        velocity = state[state.shape[0]//2:]

        accel = action * self.power
        velocity += accel
        velocity = self._apply_friction(state, velocity)
        for direction in range(self.n):
            velocity[direction] = np.clip(velocity[direction], -self.max_velocity, self.max_velocity)

        return position, velocity

    def _apply_friction(self, state, velocity):
        for direction in range(self.n):
            if velocity[direction] > 0:
                velocity[direction] = max(0, velocity[direction]-self.friction)
            else:
                velocity[direction] = min(0, velocity[direction]+self.friction)
        return velocity

    def _compute_step(self, current_state, action):
        if self.acceleration:
            if self.continuous:
                position, velocity = self._continuous_acceleration_step(current_state, action)
            else:
                position, velocity = self._discrete_acceleration_step(current_state, action)
        else:
            if self.continuous:
                position, velocity = self._continuous_velocity_step(current_state, action)
            else:
                position, velocity = self._discrete_velocity_step(current_state, action)

        return position, velocity

    def _apply_state(self, position, velocity):
        position += velocity
        for direction in range(self.n):
            position[direction] = np.clip(position[direction], -self.max_position, self.max_position)
        return position

    def _check_object(self, position):
        reward = self.low_reward
        reach = False
        info = ""
        """ Check object in n dimensional space """
        for object in self.objects:
            ob_coor = object.coordinates
            # d = np.sqrt(pow(position[0]-ob_coor[0], 2) + pow(position[1]-ob_coor[1], 2))
            if (np.abs(position[0]-ob_coor[0]) < 0.3) and (np.abs(position[1]-ob_coor[1]) < 0.3):
                if object.name == "target":
                    reach = True
                    info = "Target collected "
                else:
                    reach = False
                    info = "Fake collected "
                reward = object.reward
                object.coordinates = np.random.uniform(low=max(-self.reset_radius, -self.max_position),high=min(self.reset_radius, self.max_position),size=self.n)
        return reward, reach, info

    def _check_walls(self, position, velocity):
        """ Check for wall in n dimensional space """
        for infos in self.wall_position:
            nth, boundary = infos
            if abs(position[int(nth)] + boundary) >= 2 * self.max_position:
                velocity[int(nth)] = 0
        return velocity

    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(low=max(-self.reset_radius, -self.max_position),high=min(self.reset_radius, self.max_position), size=(num_tasks, self.n))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def sample_step(self, state, action):

        position, velocity = self._compute_step(state, action)
        position = self._apply_state(position, velocity)

        reward, reach_target, info = self._check_object(position)
        velocity = self._check_walls(position, velocity)

        done = reach_target or self._current_episode_step >= self._max_episode_steps
        state = np.array(position)

        if self.acceleration:
            state = np.r_[position, velocity]

        return state, reward, done, info

    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        current_state = self.state[:4]
        current_goal = self.state[4:]
        position, velocity = self._compute_step(current_state, action)

        if self.env_description['speed_limit_mode'] == 'vector_norm':
            if np.linalg.norm(velocity) > self.max_velocity:
                velocity = (velocity / np.linalg.norm(velocity)) * self.max_velocity

        position = self._apply_state(position, velocity)

        reward, reach_target, info = self._check_object(position)
        velocity = self._check_walls(position, velocity)

        done = reach_target or self._current_episode_step >= self._max_episode_steps
        if self.acceleration:
            self.state = np.r_[position, velocity]

        self._current_episode_step += 1
        self.position = position
        self.velocity = velocity
        self.state = np.r_[self.state, current_goal]

        return self.state, reward, done, info

    def reset_task(self, task):
        if self.env_description['agent_starting'] == 'fixed':
            self.position = np.asarray([-0.6, -0.6])
        else:
            self.position = np.random.uniform(low=max(-self.reset_radius, -self.max_position),high=min(self.reset_radius, self.max_position),size=self.n)
        self.velocity = np.zeros(self.n)
        self.objects = []
        self.target_position = task['goal']
        target_color = [0, 1, 0]
        self.objects.append(gameObject(self.target_position, 30, target_color, self.high_reward, 'target'))
        if self.acceleration:
            self.state = np.r_[self.position, self.velocity]
        else:
            self.state = self.position
        self.state = np.r_[self.state, np.concatenate((self.objects[0].coordinates, np.asarray([0, 0])))]
        self._current_episode_step = 0

        return self.state

    def reset(self):
        if self.env_description['agent_starting'] == 'fixed':
            self.position = np.asarray([-0.6, -0.6])
        else:
            self.position = np.random.uniform(low=max(-self.reset_radius, -self.max_position),high=min(self.reset_radius, self.max_position),size=self.n)
        self.velocity = np.zeros(self.n)
        target_color = [0, 1, 0]
        self.objects = []
        if self.env_description['mode'] == 'deterministic':
            self.objects.append(gameObject(self.target_position, 30, target_color, self.high_reward, 'target'))
        else:
            self.target_position = np.random.uniform(low=max(-self.reset_radius, -self.max_position), high=min(self.reset_radius, self.max_position),size=self.n)
            self.objects.append(gameObject(self.target_position, 30, target_color, self.high_reward, 'target'))
        if self.acceleration:
            self.state = np.r_[self.position, self.velocity]
        else:
            self.state = self.position
        self.state = np.r_[self.state, np.concatenate((self.objects[0].coordinates, np.asarray([0, 0])))]
        self._current_episode_step = 0

        return self.state

    def get_vertices_list(self, nth, boundary, width, height, line=0):
        if self.n == 1:

            y1 = line + 20
            y2 = line - 20

            if nth == 0 and boundary == self.max_position:
                x1 = width - 10
                x2 = width

            if nth == 0 and boundary == -self.max_position:
                x1 = 0
                x2 = 10

            return x1, y1, x2, y2

        if self.n == 2:

            if nth == 0 and boundary == self.max_position:
                x1 = width - 10
                x2 = width
                y1 = 0
                y2 = height

            if nth == 0 and boundary == -self.max_position:

                x1 = 0
                x2 = 10
                y1 = 0
                y2 = height

            if nth == 1 and boundary == self.max_position:

                x1 = 0
                x2 = width
                y1 = height - 10
                y2 = height

            if nth == 1 and boundary == -self.max_position:

                x1 = 0
                x2 = width
                y1 = 10
                y2 = 0
            
            return x1, y1, x2, y2

        return None

    def render(self, mode='human'):
        if self.n == 2:
            screen_width = 200
            screen_height = 200

            world_width = self.max_position*2
            scale = screen_width/world_width

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)

                """ Build agent sprite """
                self.agent = rendering.make_circle(radius=20,res=30,filled=True)
                self.agent.set_color(1,0,0)
                self.agenttrans = rendering.Transform()
                self.agent.add_attr(self.agenttrans)
                self.viewer.add_geom(self.agent)
                self.agentline = rendering.FilledPolygon([
                    (0, 0),(0, 2),(15, 2),(15, -2),(0,-2)
                    ])
                self.agentline.set_color(0,0,0)
                self.agentlinetrans = rendering.Transform()
                self.agentline.add_attr(self.agentlinetrans)
                self.viewer.add_geom(self.agentline)

                """ Build objects """
                self.targettrans = []
                for i in range(len(self.objects)):
                    object = rendering.make_circle(radius=self.objects[i].radius, res=30, filled=True)
                    object.set_color(self.objects[i].color[0],self.objects[i].color[1],self.objects[i].color[2])
                    self.targettrans.append(rendering.Transform())
                    object.add_attr(self.targettrans[i])
                    self.viewer.add_geom(object)

        if self.state is None:
            return None

        pos = self.position
        vel = self.velocity
        agent_position = (pos+self.max_position)*scale
        self.agenttrans.set_translation(agent_position[0],agent_position[1])
        self.agentlinetrans.set_translation(agent_position[0],agent_position[1])
        self.agentlinetrans.set_rotation(math.atan2(vel[1],vel[0]))
        for i in range(len(self.objects)):
            object_position = (self.objects[i].coordinates+self.max_position)*scale
            self.targettrans[i].set_translation(object_position[0], object_position[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
