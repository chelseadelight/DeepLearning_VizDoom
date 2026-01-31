
from os import wait
from typing import Any, List, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType
from sample_factory.enjoy import gym
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv
import math

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Player(Point):
    def __init__(self, x: float, y: float, angle: float):
        super().__init__(x, y)
        self.angle = angle

class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    def orientation(self, p: Point) -> int:
        val = (self.p2.y - self.p1.y) * (p.x - self.p2.x) - (self.p2.x - self.p1.x) * (p.y - self.p2.y)
        if val == 0:
            return 0
        return 1 if val > 0 else 2 # Clockwise or Counterclockwise

    def angle(self) -> float:
        return math.atan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x)

    def on_segment(self, p: Point) -> bool:
        return (min(self.p1.x, self.p2.x) <= p.x <= max(self.p1.x, self.p2.x) and
                min(self.p1.y, self.p2.y) <= p.y <= max(self.p1.y, self.p2.y))

    def intersects(self, other: 'Line') -> bool:
        o1 = self.orientation(other.p1)
        o2 = self.orientation(other.p2)
        o3 = other.orientation(self.p1)
        o4 = other.orientation(self.p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # self.p1, self.p2 and other.p1 are collinear and other.p1 lies on segment self.p1self.p2
        if o1 == 0 and self.on_segment(other.p1):
            return True

        # self.p1, self.p2 and other.p2 are collinear and other.p2 lies on segment self.p1self.p2
        if o2 == 0 and self.on_segment(other.p2):
            return True

        # other.p1, other.p2 and self.p1 are collinear and self.p1 lies on segment other.p1other.p2
        if o3 == 0 and other.on_segment(self.p1):
            return True

        # other.p1, other.p2 and self.p2 are collinear and self.p2 lies on segment other.p1other.p2
        if o4 == 0 and other.on_segment(self.p2):
            return True

        return False




class HideAndSeekWrapper(gym.Wrapper):
    def __init__(self, env):
        super(HideAndSeekWrapper, self).__init__(env)

        # set curr_policy_idx on unwrapped env so the sampling algo will set it.
        setattr(self.unwrapped, 'curr_policy_idx', 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset accumulators.
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # get self.unwrapped and cast it to VizdoomEnv
        self_env = self.unwrapped
        if not isinstance(self_env, VizdoomEnv):
            raise TypeError("Expected self.env.unwrapped to be an instance of VizdoomEnv")

        game = self_env.game
        if game is None:
            print(">>> Game is None, returning default step result.")
            return obs, reward, terminated, truncated, info

        state = self_env.game.get_state()

        if state is None:
            print(">>> State is None, returning default step result.")
            return obs, reward, terminated, truncated, info


        vars = state.game_variables;

        player = Player(vars[0], vars[1], vars[2])
        enemies: List[Player] = []
        walls: List[Line] = []

        for label in state.labels:
            if label.object_name == "DoomPlayer":
                enemies.append(Player(label.object_position_x, label.object_position_y, label.object_angle))


        ## can probably be done once.
        for sector in state.sectors:
            for line in sector.lines:
                if line.is_blocking:
                    walls.append(Line(Point(line.x1, line.y1), Point(line.x2, line.y2)))

        # smaller_own_angle means the enemy is in front of the player.
        # smaller_enemy_angle means the player is in front of the enemy.
        # (distance, own_angle_delta, enemy_angle_delta)
        metrics = []

        # compute lines of sight.
        for enemy in enemies:
            los = Line(player, enemy)

            # check if any wall intersects the line of sight.
            visible = True
            for wall in walls:
                if los.intersects(wall):
                    visible = False
                    break
            if not visible:
                continue

            distance = player.distance_to(enemy)
            # own angle delta is the difference between player's angle and the angle of the line of sight. 
            own_angle = abs((los.angle() - player.angle + math.pi) % (2 * math.pi) - math.pi)
            # enemy angle delta is the difference between enemy's angle and the inverse angle of the line of sight.
            enemy_angle = abs((los.angle() + math.pi - enemy.angle + math.pi) % (2 * math.pi) - math.pi)

            metrics.append((distance, own_angle, enemy_angle))

        # The game is symmetric, so everyone is simultaneously hider and seeker.
        # The reward is reminiscent of the prisoner's dilemma. 
        # If you can see an enemy while they cannot see you, you get a positive reward.
        # If you are seen by an enemy while you cannot see them, you get a negative reward.
        # If you can see each other or neither can see each other, the reward is zero.
        # The reward is scaled by distance and angle deltas to encourage stealthy behavior.

        if reward is None:
            reward = 0

        for distance, own_angle, enemy_angle in metrics:
            if own_angle < (math.pi / 4) and enemy_angle >= (math.pi / 4):
                reward += 2.0 / (1.0 + distance)
            elif own_angle >= (math.pi / 4) and enemy_angle < (math.pi / 4):
                reward -= 1.0 / (1.0 + distance)


        return obs, reward, terminated, truncated, info

