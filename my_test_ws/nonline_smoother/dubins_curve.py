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
        x_list, y_list, yaw_list, gears = [x], [y], [yaw], []
        
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
                    x += (math.sin(new_yaw) - math.sin(yaw)) / cmd.curvature
                    y += (math.cos(yaw) - math.cos(new_yaw)) / cmd.curvature
                    yaw = new_yaw
                    
                yaw = self._mod2pi(yaw)
                x_list.append(x)
                y_list.append(y)
                yaw_list.append(yaw)
                gears.append(direction)

        return x_list, y_list, yaw_list, gears

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

class DubinsPlanner:
    def __init__(self, turning_radius: float):
        self.turning_radius = turning_radius

    @staticmethod
    def _mod2pi(theta: float) -> float:
        return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))

    def _evaluate_LSL(self, alpha: float, beta: float, d: float):
        p_sq = 2.0 + d**2 - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
        if p_sq < 0: return None
        tmp = math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) - math.sin(beta))
        return self._mod2pi(-alpha + tmp), math.sqrt(p_sq), self._mod2pi(beta - tmp), "LSL"

    def _evaluate_RSR(self, alpha: float, beta: float, d: float):
        p_sq = 2.0 + d**2 - 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))
        if p_sq < 0: return None
        tmp = math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
        return self._mod2pi(alpha - tmp), math.sqrt(p_sq), self._mod2pi(-beta + tmp), "RSR"

    def _evaluate_LSR(self, alpha: float, beta: float, d: float):
        p_sq = -2.0 + d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) + math.sin(beta))
        if p_sq < 0: return None
        tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, math.sqrt(p_sq))
        return self._mod2pi(-alpha + tmp), math.sqrt(p_sq), self._mod2pi(-beta + tmp), "LSR"

    def _evaluate_RSL(self, alpha: float, beta: float, d: float):
        p_sq = -2.0 + d**2 + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) + math.sin(beta))
        if p_sq < 0: return None
        tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, math.sqrt(p_sq))
        return self._mod2pi(alpha - tmp), math.sqrt(p_sq), self._mod2pi(beta - tmp), "RSL"

    def _evaluate_RLR(self, alpha: float, beta: float, d: float):
        tmp = (6.0 - d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
        if abs(tmp) > 1.0: return None
        p = self._mod2pi(math.acos(tmp))
        t = self._mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
        return t, p, self._mod2pi(alpha - beta - t + p), "RLR"

    def _evaluate_LRL(self, alpha: float, beta: float, d: float):
        tmp = (6.0 - d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
        if abs(tmp) > 1.0: return None
        p = self._mod2pi(math.acos(tmp))
        t = self._mod2pi(-alpha + math.atan2(-math.cos(alpha) + math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
        return t, p, self._mod2pi(beta - alpha - t + p), "LRL"

    def get_all_paths(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> List[Path]:
        """Finds all valid Dubins paths connecting start to goal."""
        dx, dy = ex - sx, ey - sy
        d_val = math.hypot(dx, dy) / self.turning_radius
        theta = self._mod2pi(math.atan2(dy, dx))
        alpha = self._mod2pi(syaw - theta)
        beta = self._mod2pi(eyaw - theta)

        evaluators = [
            self._evaluate_LSL, self._evaluate_RSR,
            self._evaluate_LSR, self._evaluate_RSL,
            self._evaluate_RLR, self._evaluate_LRL
        ]

        paths = []
        for evaluate in evaluators:
            res = evaluate(alpha, beta, d_val)
            if res:
                t, p, q, types_str = res
                commands = []
                for m_type, length_norm in zip(types_str, [t, p, q]):
                    kappa = 0.0 if m_type == 'S' else (1.0 if m_type == 'L' else -1.0) / self.turning_radius
                    commands.append(Command(length_norm * self.turning_radius, kappa))
                paths.append(Path(commands))
                
        return paths

    def plan(self, sx: float, sy: float, syaw: float, ex: float, ey: float, eyaw: float) -> PlanningResult:
        """Returns a result object containing all paths and the optimal one."""
        return PlanningResult(self.get_all_paths(sx, sy, syaw, ex, ey, eyaw))

if __name__ == '__main__':
    sx, sy, syaw = 0.0, 0.0, 0.0
    ex, ey, eyaw = 10.0, -10.0, math.radians(-180)
    
    planner = DubinsPlanner(turning_radius=3.0)
    result = planner.plan(sx, sy, syaw, ex, ey, eyaw)
    
    if result.best_path:
        print(f"Total Paths Found: {len(result.all_paths)}")
        print(f"Best Dubins Path Cost: {result.best_path.cost:.2f}")
        for idx, cmd in enumerate(result.best_commands):
            print(f"  Segment {idx}: {cmd}")
        
        rx, ry, ryaw, dlist = result.best_path.generate_trajectory(sx, sy, syaw, step_size=0.1)
        
        import matplotlib.pyplot as plt
        plt.plot(rx, ry, "-r", label="Optimal Dubins Path")
        plt.arrow(sx, sy, math.cos(syaw), math.sin(syaw), head_width=0.5, color="green")
        plt.arrow(ex, ey, math.cos(eyaw), math.sin(eyaw), head_width=0.5, color="blue")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Failed to find a Dubins path")
