import math
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
                    yaw += ds * cmd.curvature
                    x += ds * math.cos(yaw)
                    y += ds * math.sin(yaw)
                    
                yaw = self._mod2pi(yaw)
                x_list.append(x)
                y_list.append(y)
                yaw_list.append(yaw)
                dir_list.append(direction)

        return x_list, y_list, yaw_list, dir_list

class PlanningResult:
    """Container for planning results to avoid ugly tuples."""
    def __init__(self, paths: List[Path]):
        self.all_paths = sorted(paths, key=lambda p: p.cost)
        self.best_path = self.all_paths[0] if self.all_paths else None
        
    @property
    def best_commands(self) -> Optional[List[Command]]:
        return self.best_path.commands if self.best_path else None
        
    @property
    def all_commands(self) -> List[List[Command]]:
        return [p.commands for p in self.all_paths]

class ReedsSheppPlanner:
    def __init__(self, turning_radius: float):
        self.turning_radius = turning_radius

    @staticmethod
    def _mod2pi(theta: float) -> float:
        v = theta % (2.0 * math.pi)
        if v < -math.pi: v += 2.0 * math.pi
        elif v > math.pi: v -= 2.0 * math.pi
        return v

    @staticmethod
    def _polar(x: float, y: float) -> Tuple[float, float]:
        return math.hypot(x, y), math.atan2(y, x)

    def _evaluate_LSL(self, x: float, y: float, phi: float):
        u, t = self._polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if t >= 0.0:
            v = self._mod2pi(phi - t)
            if v >= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _evaluate_LSR(self, x: float, y: float, phi: float):
        u1, t1 = self._polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
        if u1 >= 4.0:
            u = math.sqrt(u1**2 - 4.0)
            t = self._mod2pi(t1 + math.atan2(2.0, u))
            v = self._mod2pi(t - phi)
            if t >= 0.0 and v >= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _evaluate_LRL(self, x: float, y: float, phi: float):
        u1, t1 = self._polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if u1 <= 4.0:
            u = 2.0 * math.asin(0.25 * u1)
            t = self._mod2pi(t1 + 0.5 * u + math.pi)
            v = self._mod2pi(phi - t + u)
            if t >= 0.0 and u >= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _evaluate_LRL_neg(self, x: float, y: float, phi: float):
        u1, t1 = self._polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if u1 <= 4.0:
            u = -2.0 * math.asin(0.25 * u1)
            t = self._mod2pi(t1 + 0.5 * u + math.pi)
            v = self._mod2pi(phi - t + u)
            if t >= 0.0 and u <= 0.0: return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def get_all_paths(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> List[Path]:
        """Finds all valid Reeds-Shepp paths connecting start to goal."""
        dx, dy = ex - sx, ey - sy
        c, s = math.cos(syaw), math.sin(syaw)
        local_x, local_y = (c * dx + s * dy) / self.turning_radius, (-s * dx + c * dy) / self.turning_radius
        local_yaw = self._mod2pi(eyaw - syaw)

        paths = []

        def ingest(f, char_types, x_, y_, phi_, t_flip=False, reflect=False):
            ok, t, u, v = f(x_, y_, phi_)
            if ok:
                lengths = [-l if t_flip else l for l in [t, u, v]]
                types = ['R' if c == 'L' else 'L' if c == 'R' else 'S' for c in char_types] if reflect else char_types
                commands = []
                for l_norm, c_type in zip(lengths, types):
                    kappa = 0.0 if c_type == 'S' else (1.0 if c_type == 'L' else -1.0) / self.turning_radius
                    commands.append(Command(l_norm * self.turning_radius, kappa))
                paths.append(Path(commands))

        for f, labels in [
            (self._evaluate_LSL, "LSL"), (self._evaluate_LSR, "LSR"),
            (self._evaluate_LRL, "LRL"), (self._evaluate_LRL_neg, "LRL")
        ]:
            ingest(f, labels, local_x, local_y, local_yaw)
            ingest(f, labels, -local_x, local_y, -local_yaw, t_flip=True)
            ingest(f, labels, local_x, -local_y, -local_yaw, reflect=True)
            ingest(f, labels, -local_x, -local_y, local_yaw, t_flip=True, reflect=True)
            
        return paths

    def plan(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> PlanningResult:
        """Returns a result object containing all paths and the optimal one."""
        return PlanningResult(self.get_all_paths(sx, sy, syaw, ex, ey, eyaw))

if __name__ == '__main__':
    sx, sy, syaw = 0.0, 0.0, 0.0
    ex, ey, eyaw = 0.0, 5.0, math.radians(180)
    
    planner = ReedsSheppPlanner(turning_radius=3.0)
    result = planner.plan(sx, sy, syaw, ex, ey, eyaw)
    
    if result.best_path:
        print(f"Total Paths Found: {len(result.all_paths)}")
        print(f"Best Reeds-Shepp Path Cost: {result.best_path.cost:.2f}")
        for idx, cmd in enumerate(result.best_commands):
            print(f"  Segment {idx}: {cmd}")
        
        rx, ry, ryaw, dlist = result.best_path.generate_trajectory(sx, sy, syaw, step_size=0.1)
        
        import matplotlib.pyplot as plt
        plt.plot(rx, ry, "-b", label="Optimal Reeds-Shepp Path")
        plt.arrow(sx, sy, math.cos(syaw), math.sin(syaw), head_width=0.5, color="green")
        plt.arrow(ex, ey, math.cos(eyaw), math.sin(eyaw), head_width=0.5, color="red")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Failed to find a Reeds-Shepp path")
