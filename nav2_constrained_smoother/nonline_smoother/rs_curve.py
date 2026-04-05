import math
import numpy as np
from typing import List, Optional, Tuple

class Command:
    def __init__(self, length: float, curvature: float):
        self.length = length
        self.curvature = curvature
        
    def __repr__(self) -> str:
        return f"Command(len={self.length:.3f}, kappa={self.curvature:.3f})"

class Path:
    def __init__(self, commands: List[Command]):
        self.commands = commands

    @property
    def cost(self) -> float:
        return sum(abs(cmd.length) for cmd in self.commands)
        
    @staticmethod
    def _mod2pi(theta: float) -> float:
        return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))
        
    def generate_trajectory(self, sx: float, sy: float, syaw: float, step_size: float = 0.1) -> Tuple[List[float], List[float], List[float], List[int]]:
        x, y, yaw = sx, sy, syaw
        x_list, y_list, yaw_list, dir_list = [x], [y], [yaw], [1]
        
        for cmd in self.commands:
            if abs(cmd.length) < 1e-6: continue
            
            direction = 1 if cmd.length > 0 else -1
            steps = max(int(abs(cmd.length) / step_size), 1)
            ds = cmd.length / steps
            
            for _ in range(steps):
                if abs(cmd.curvature) < 1e-6:
                    x += ds * math.cos(yaw)
                    y += ds * math.sin(yaw)
                else:
                    new_yaw = yaw + ds * cmd.curvature
                    # Precise integration of circular arcs
                    x += (math.sin(new_yaw) - math.sin(yaw)) / cmd.curvature
                    y += (math.cos(yaw) - math.cos(new_yaw)) / cmd.curvature
                    yaw = new_yaw
                    
                yaw = self._mod2pi(yaw)
                x_list.append(x)
                y_list.append(y)
                yaw_list.append(yaw)
                dir_list.append(direction)

        return x_list, y_list, yaw_list, dir_list

class PlanningResult:
    def __init__(self, paths: List[Path]):
        self.all_paths = sorted(paths, key=lambda p: p.cost)
        self.best_path = self.all_paths[0] if self.all_paths else None
        
    @property
    def best_commands(self) -> Optional[List[Command]]:
        return self.best_path.commands if self.best_path else None

class ReedsSheppPlanner:
    def __init__(self, turning_radius: float):
        self.turning_radius = turning_radius

    @staticmethod
    def _pi_2_pi(theta: float) -> float:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _polar(x: float, y: float) -> Tuple[float, float]:
        return math.hypot(x, y), math.atan2(y, x)

    def _LSL(self, x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
        u, t = self._polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if t >= 0.0:
            v = self._pi_2_pi(phi - t)
            if v >= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _LSR(self, x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
        u1, t1 = self._polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
        if u1**2 >= 4.0:
            u = math.sqrt(u1**2 - 4.0)
            t = self._pi_2_pi(t1 + math.atan2(2.0, u))
            v = self._pi_2_pi(t - phi)
            if t >= 0.0 and v >= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _LRL(self, x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
        u1, t1 = self._polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if u1 <= 4.0:
            u = -2.0 * math.asin(0.25 * u1)
            t = self._pi_2_pi(t1 + 0.5 * u + math.pi)
            v = self._pi_2_pi(phi - t + u)
            if t >= 0.0 and u <= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def get_all_paths(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> List[Path]:
        dx, dy = ex - sx, ey - sy
        c, s = math.cos(syaw), math.sin(syaw)
        x = (c * dx + s * dy) / self.turning_radius
        y = (-s * dx + c * dy) / self.turning_radius
        phi = eyaw - syaw

        paths = []

        # 12 core recipes from Reeds-Shepp (simplified)
        # Using symmetries to generate all 48 possible optimal types
        
        def set_path(lengths, types):
            commands = []
            for l, t in zip(lengths, types):
                kappa = 0.0 if t == 'S' else (1.0 if t == 'L' else -1.0) / self.turning_radius
                commands.append(Command(l * self.turning_radius, kappa))
            
            # Simple validation to ensure it reaches (x, y, phi)
            # Starting at (0,0,0) locally
            px, py, pyaw = 0.0, 0.0, 0.0
            for cmd in commands:
                l, k = cmd.length, cmd.curvature
                if abs(k) < 1e-6:
                    px += l * math.cos(pyaw)
                    py += l * math.sin(pyaw)
                else:
                    new_yaw = pyaw + l * k
                    px += (math.sin(new_yaw) - math.sin(pyaw)) / k
                    py += (math.cos(pyaw) - math.cos(new_yaw)) / k
                    pyaw = new_yaw
            
            dist_err = math.hypot(px - dx, py - dy)
            yaw_err = abs(self._pi_2_pi(pyaw - (eyaw - syaw)))
            if dist_err < 0.1 and yaw_err < 0.1:
                paths.append(Path(commands))

        # CSC (LSL, LSR, etc)
        for t_flip in [1, -1]:
            for reflect in [1, -1]:
                # Try LSL
                ok, t, u, v = self._LSL(x * t_flip, y * reflect, phi * t_flip * reflect)
                if ok: 
                    set_path([t*t_flip, u*t_flip, v*t_flip], 
                             ['L' if reflect > 0 else 'R', 'S', 'L' if reflect > 0 else 'R'])
                
                # Try LSR
                ok, t, u, v = self._LSR(x * t_flip, y * reflect, phi * t_flip * reflect)
                if ok:
                    set_path([t*t_flip, u*t_flip, v*t_flip],
                             ['L' if reflect > 0 else 'R', 'S', 'R' if reflect > 0 else 'L'])
                
                # Try LRL
                ok, t, u, v = self._LRL(x * t_flip, y * reflect, phi * t_flip * reflect)
                if ok:
                    set_path([t*t_flip, u*t_flip, v*t_flip],
                             ['L' if reflect > 0 else 'R', 'R' if reflect > 0 else 'L', 'L' if reflect > 0 else 'R'])

        return paths

    def plan(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> PlanningResult:
        return PlanningResult(self.get_all_paths(sx, sy, syaw, ex, ey, eyaw))

if __name__ == '__main__':
    # Test problematic case
    sx, sy, syaw = 0.0, 0.0, 0.0
    ex, ey, eyaw = 20.0, 0.0, 1.42
    planner = ReedsSheppPlanner(turning_radius=5.0)
    result = planner.plan(sx, sy, syaw, ex, ey, eyaw)
    if result.best_path:
        print(f"Success! Cost: {result.best_path.cost:.2f}")
        for cmd in result.best_commands:
            print(f"  {cmd}")
    else:
        print("Failed to find path")
