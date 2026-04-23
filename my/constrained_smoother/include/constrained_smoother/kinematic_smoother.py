import numpy as np
from scipy.optimize import OptimizeResult, least_squares


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a, b):
    return normalize_angle(a - b)


class KinematicSmoother:
    def __init__(
        self,
        w_model=10.0,
        w_smooth=10.0,
        w_s=1.0,
        ref_weight=1.0,
        w_fix=100.0,
        target_spacing=0.2,
        max_kappa=0.5,
        max_iter=50,
    ):
        self.w_model = w_model
        self.w_smooth = w_smooth
        self.w_s = w_s
        self.w_fix = w_fix
        self.ref_weight = ref_weight
        self.target_spacing = target_spacing
        self.max_kappa = max_kappa
        self.max_iter = max_iter

    def optimize(self, raw_path, gear_directions=None, return_result=False, verbose=1):
        raw_path = np.asarray(raw_path, dtype=float)
        if raw_path.ndim != 2 or raw_path.shape[1] not in (2, 3):
            raise ValueError("raw_path must have shape (N, 2) or (N, 3)")

        num_points = len(raw_path)
        if num_points == 0:
            raise ValueError("raw_path must contain at least one pose")

        if num_points == 1:
            theta0 = raw_path[0, 2] if raw_path.shape[1] == 3 else 0.0
            single_state = np.array([[raw_path[0, 0], raw_path[0, 1], theta0, 0.0, 0.0]])
            if return_result:
                return OptimizeResult(
                    x=single_state.flatten(),
                    success=True,
                    status=0,
                    message="Single-point path requires no optimization.",
                    cost=0.0,
                    fun=np.zeros(0),
                    nfev=0,
                )
            return single_state

        if gear_directions is None:
            gear_directions = np.ones(num_points - 1, dtype=float)
        else:
            gear_directions = np.asarray(gear_directions, dtype=float)
            if gear_directions.shape != (num_points - 1,):
                raise ValueError("gear_directions must have shape (N-1,)")

        processed_path = [raw_path[0]]
        processed_gears = []
        is_cusp_segment = []

        for index in range(num_points - 1):
            current_gear = gear_directions[index]
            next_gear = gear_directions[index + 1] if index + 1 < len(gear_directions) else current_gear

            processed_gears.append(current_gear)
            is_cusp_segment.append(False)
            processed_path.append(raw_path[index + 1])

            if index < num_points - 2 and current_gear != next_gear:
                processed_gears.append(0.0)
                is_cusp_segment.append(True)
                processed_path.append(raw_path[index + 1])

        processed_path = np.asarray(processed_path, dtype=float)
        processed_gears = np.asarray(processed_gears, dtype=float)
        is_cusp_segment = np.asarray(is_cusp_segment, dtype=bool)
        state_count = len(processed_path)

        x_init = processed_path[:, 0]
        y_init = processed_path[:, 1]
        theta_init = np.zeros(state_count)

        for index in range(state_count - 1):
            dx = processed_path[index + 1, 0] - processed_path[index, 0]
            dy = processed_path[index + 1, 1] - processed_path[index, 1]
            if is_cusp_segment[index]:
                theta_init[index] = theta_init[index - 1] if index > 0 else 0.0
                continue

            segment_norm = np.hypot(dx, dy)
            if segment_norm > 1e-6:
                heading = np.arctan2(dy, dx)
                if processed_gears[index] < 0:
                    heading += np.pi
                theta_init[index] = normalize_angle(heading)
            else:
                theta_init[index] = theta_init[index - 1] if index > 0 else 0.0

        theta_init[-1] = theta_init[-2]
        if raw_path.shape[1] == 3:
            theta_init[0] = raw_path[0, 2]

        kappa_init = np.zeros(state_count)
        ds_init = np.zeros(state_count)
        for index in range(state_count - 1):
            if is_cusp_segment[index]:
                ds_init[index] = 0.0
            else:
                ds_init[index] = np.hypot(
                    processed_path[index + 1, 0] - processed_path[index, 0],
                    processed_path[index + 1, 1] - processed_path[index, 1],
                )

        initial_guess = np.column_stack((x_init, y_init, theta_init, kappa_init, ds_init)).flatten()

        start_pose = np.zeros(3)
        start_pose[:2] = processed_path[0, :2]
        start_pose[2] = theta_init[0] if raw_path.shape[1] < 3 else raw_path[0, 2]

        end_pose = np.zeros(3)
        end_pose[:2] = processed_path[-1, :2]
        end_pose[2] = theta_init[-1] if raw_path.shape[1] < 3 else raw_path[-1, 2]

        lower_bounds = np.full(5 * state_count, -np.inf)
        upper_bounds = np.full(5 * state_count, np.inf)

        ds_indices = np.arange(4, 5 * state_count, 5)
        lower_bounds[ds_indices] = 0.0

        kappa_indices = np.arange(3, 5 * state_count, 5)
        lower_bounds[kappa_indices] = -self.max_kappa
        upper_bounds[kappa_indices] = self.max_kappa

        result = least_squares(
            self._residuals,
            initial_guess,
            bounds=(lower_bounds, upper_bounds),
            args=(processed_path, processed_gears, is_cusp_segment, start_pose, end_pose),
            verbose=verbose,
            max_nfev=self.max_iter,
            x_scale="jac",
        )

        if return_result:
            return result

        return result.x.reshape((state_count, 5))

    def _residuals(self, variables, ref_path, gears, is_cusp, start_pose, end_pose):
        state = variables.reshape((len(ref_path), 5))

        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        kappa = state[:, 3]
        ds = state[:, 4]

        residuals = []
        residuals.extend(self._kinematic_residuals(x, y, theta, kappa, ds, gears, is_cusp))
        residuals.extend(self._smoothness_residuals(kappa, ds, is_cusp))
        residuals.extend(self._spacing_residuals(ds, is_cusp))
        residuals.extend(self._boundary_residuals(x, y, theta, ds, start_pose, end_pose))

        if self.ref_weight > 1e-5:
            dist_err = np.hypot(x - ref_path[:, 0], y - ref_path[:, 1])
            residuals.extend((self.ref_weight * dist_err).tolist())

        return np.asarray(residuals, dtype=float)

    def _kinematic_residuals(self, x, y, theta, kappa, ds, gears, is_cusp):
        residuals = []
        for index in range(len(x) - 1):
            if is_cusp[index]:
                residuals.append(self.w_fix * (x[index + 1] - x[index]))
                residuals.append(self.w_fix * (y[index + 1] - y[index]))
                residuals.append(self.w_fix * angle_diff(theta[index + 1], theta[index]))
                continue

            direction = 1.0 if gears[index] >= 0 else -1.0
            step = ds[index]
            current_kappa = kappa[index]
            next_kappa = kappa[index + 1]

            theta_pred = theta[index] + direction * step * (current_kappa + next_kappa) * 0.5
            theta_mid = theta[index] + direction * step * current_kappa * 0.5
            x_pred = x[index] + direction * step * np.cos(theta_mid)
            y_pred = y[index] + direction * step * np.sin(theta_mid)

            residuals.append(self.w_model * (x[index + 1] - x_pred))
            residuals.append(self.w_model * (y[index + 1] - y_pred))
            residuals.append(self.w_model * angle_diff(theta[index + 1], theta_pred))

        return residuals

    def _smoothness_residuals(self, kappa, ds, is_cusp):
        residuals = []
        for index in range(len(kappa) - 1):
            if is_cusp[index]:
                continue

            denom = np.sqrt(ds[index]) if ds[index] > 1e-3 else 0.03
            residuals.append(self.w_smooth * (kappa[index + 1] - kappa[index]) / denom)

        return residuals

    def _spacing_residuals(self, ds, is_cusp):
        residuals = []
        scale = max(self.target_spacing, 1e-4)
        for index in range(len(ds) - 1):
            if is_cusp[index]:
                residuals.append(self.w_s * 10.0 * ds[index])
            else:
                residuals.append(self.w_s * (ds[index] - self.target_spacing) / scale)

        return residuals

    def _boundary_residuals(self, x, y, theta, ds, start_pose, end_pose):
        return [
            self.w_fix * (x[0] - start_pose[0]),
            self.w_fix * (y[0] - start_pose[1]),
            self.w_fix * angle_diff(theta[0], start_pose[2]),
            self.w_fix * (x[-1] - end_pose[0]),
            self.w_fix * (y[-1] - end_pose[1]),
            self.w_fix * angle_diff(theta[-1], end_pose[2]),
            self.w_fix * ds[-1],
        ]
