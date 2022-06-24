import numpy as np
import copy
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room, RandGen
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS


class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        max_episode_steps=None,
        task=None,
        **kwargs
    ):
        if task is None:
            task = {}
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        # Initialize the meta learning variables
        self.rand = RandGen()  # I need to declare twice the random generator due to how the super class is initialized
        self._task = task
        self._generator = task.get('generator', self.rand.subset([(0, 1), (0, -1), (-1, 0), (1, 0)], 4))

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            obs_width=224,
            obs_height=224,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    #floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self._generator

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        X = (self.num_cols - 0.5) * self.room_size + (self.num_cols - 1) * self.gap_size
        Z = (self.num_rows - 0.5) * self.room_size + (self.num_rows - 1) * self.gap_size
        self.box = self.place_entity(Box(color='red'), pos=np.array([X, 0, Z]))

        X = 0.5 * self.room_size
        Z = 0.5 * self.room_size
        self.place_entity(self.agent, pos=np.array([X, 0, Z]), dir=0)

    def reward(self, done, fixed_penalty=0.1):
        """
        Custom reward per step including geometric distance and penalty per step
        """
        geo_dist = np.linalg.norm(self.box.pos - self.agent.pos)
        reward = - np.log(geo_dist) + 1 - fixed_penalty

        # If the agent reaches the target, give good amount of reward hehe
        if done:
            reward += 10

        return reward

    def step(self, action, resnet=True):
        obs, _, done, info = super().step(action, resnet=resnet)

        if self.near(self.box):
            # reward += self._reward()
            done = True

        reward = self.reward(done)

        return obs, reward, done, info

    def sample_tasks(self, num_tasks):
        generators = [self.rand.subset([(0, 1), (0, -1), (-1, 0), (1, 0)], 4) for _ in range(num_tasks)]
        tasks = [{'generator': generator} for generator in generators]
        return tasks

    def set_task(self, task):
        self._task = task
        self._generator = task['generator']

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        return dict(task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(task=state['task'])


class MazeS2(Maze):
    def __init__(self, task={}):
        super().__init__(num_rows=2, num_cols=2, task=task)


class MazeS3(Maze):
    def __init__(self, task={}):
        super().__init__(num_rows=3, num_cols=3, task=task)


class MazeS5(Maze):
    def __init__(self, task={}):
        super().__init__(num_rows=5, num_cols=5, task=task)


class MazeS3Fast(Maze):
    def __init__(self, task={}, forward_step=0.8, turn_step=45):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = 300

        super().__init__(
            num_rows=3,
            num_cols=3,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False,
            task=task
        )
