from __future__ import annotations

import argparse
import os
import sys
import threading
import textwrap
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (REPO_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.append(path_entry)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matplotlib demo for TEB MPC path tracking")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed for the first rollout")
    parser.add_argument("--save", type=str, default=None, help="Save a PNG instead of opening an interactive window")
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.save:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from my.teb_local_controller.demo_support import describe_demo_configuration, run_random_demo


VISUAL_ALPHA = 0.5


class MatplotlibTEBDemo:
    def __init__(self, first_seed: Optional[int] = None):
        self.first_seed = first_seed
        self.run_count = 0
        self.current_payload = None
        self.current_config = describe_demo_configuration()
        self.hover_target = None
        self.is_solving = False
        self.pending_result = None
        self.pending_error = None
        self.legend_artist_map = {}

        self.figure = plt.figure(figsize=(15.2, 9.8))
        canvas_manager = getattr(self.figure.canvas, "manager", None)
        if canvas_manager is not None:
            canvas_manager.set_window_title("TEB MPC Matplotlib Demo")
        grid = GridSpec(
            3,
            2,
            figure=self.figure,
            height_ratios=[3.4, 2.0, 0.22],
            width_ratios=[3.05, 0.72],
            left=0.055,
            right=0.985,
            top=0.965,
            bottom=0.08,
            wspace=0.14,
            hspace=0.24,
        )

        self.path_ax = self.figure.add_subplot(grid[0, 0])
        chart_grid = grid[1, 0].subgridspec(2, 3, wspace=0.24, hspace=0.34)
        self.dt_ax = self.figure.add_subplot(chart_grid[0, 0])
        self.speed_ax = self.figure.add_subplot(chart_grid[0, 1])
        self.accel_ax = self.figure.add_subplot(chart_grid[0, 2])
        self.kappa_ax = self.figure.add_subplot(chart_grid[1, 0])
        self.jerk_ax = self.figure.add_subplot(chart_grid[1, 1])
        self.dkappa_ax = self.figure.add_subplot(chart_grid[1, 2])
        self.info_ax = self.figure.add_subplot(grid[0:2, 1])
        self.button_ax = self.figure.add_subplot(grid[2, :])

        self.info_ax.axis("off")
        self.button_ax.set_axis_off()
        self.button = Button(self.button_ax.inset_axes([0.35, 0.12, 0.3, 0.76]), "Random Init + Solve")
        self.button.color = "#d7ebe5"
        self.button.hovercolor = "#b8ddd1"
        self.button.on_clicked(self._on_randomize)
        self.button.label.set_fontsize(11)

        self.figure.patch.set_facecolor("#f5f1e6")
        self.path_ax.set_facecolor("#fffdf8")
        self.dt_ax.set_facecolor("#fffdf8")
        self.speed_ax.set_facecolor("#fffdf8")
        self.accel_ax.set_facecolor("#fffdf8")
        self.kappa_ax.set_facecolor("#fffdf8")
        self.jerk_ax.set_facecolor("#fffdf8")
        self.dkappa_ax.set_facecolor("#fffdf8")
        self.info_ax.set_facecolor("#fcfaf3")

        self.reference_points = None
        self.projected_reference_points = None
        self.solution_points = None
        self.initial_marker = None
        self.hover_match_marker = None
        self.hover_match_line = None
        self.hover_annotation = self._create_hover_annotation()
        self.hover_annotation.set_visible(False)
        self.poll_timer = None if ARGS.save else self.figure.canvas.new_timer(interval=80)
        if self.poll_timer is not None:
            self.poll_timer.add_callback(self._consume_pending_result)

        self._initialize_axes()
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.figure.canvas.mpl_connect("pick_event", self._on_pick)
        self.refresh(seed=self.first_seed)

    def _initialize_axes(self) -> None:
        self.path_ax.set_title("Reference vs MPC Path", fontsize=14, fontweight="bold")
        self.path_ax.set_xlabel("x [m]")
        self.path_ax.set_ylabel("y [m]")
        self.path_ax.grid(True, alpha=0.16, linestyle="--", linewidth=0.8)
        self.path_ax.axis("equal")
        self.path_ax.spines[["top", "right"]].set_visible(False)

        self._style_series_axis(self.dt_ax, "dt", "dt [s]")
        self._style_series_axis(self.speed_ax, "Speed", "v [m/s]")
        self._style_series_axis(self.accel_ax, "Acceleration", "a [m/s²]")
        self._style_series_axis(self.kappa_ax, "Curvature", "kappa [1/m]")
        self._style_series_axis(self.jerk_ax, "Jerk", "jerk [m/s³]")
        self._style_series_axis(self.dkappa_ax, "Curvature Rate", "dkappa [1/(m*s)]")

    def _style_series_axis(self, axis, title: str, ylabel: str) -> None:
        axis.set_title(title, fontsize=11.5, fontweight="bold")
        axis.set_xlabel("Horizon step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.5)

    def _mark_overlay_artist(self, artist) -> None:
        artist.set_in_layout(False)

    def _register_legend(self, legend, artist_map) -> None:
        if legend is None:
            return
        self._mark_overlay_artist(legend)
        handles = list(legend.legend_handles)
        texts = list(legend.get_texts())
        labels = [text.get_text() for text in texts]
        for label, handle, text in zip(labels, handles, texts):
            target = artist_map.get(label)
            if target is None:
                continue
            handle.set_picker(True)
            text.set_picker(True)
            self.legend_artist_map[handle] = target
            self.legend_artist_map[text] = target
            alpha = 1.0 if target.get_visible() else 0.25
            handle.set_alpha(alpha)
            text.set_alpha(alpha)

    def _set_legend_entry_alpha(self, target_artist, alpha: float) -> None:
        for legend_item, mapped_artist in self.legend_artist_map.items():
            if mapped_artist is target_artist:
                legend_item.set_alpha(alpha)

    def _on_pick(self, event) -> None:
        artist = self.legend_artist_map.get(event.artist)
        if artist is None:
            return
        is_visible = not artist.get_visible()
        artist.set_visible(is_visible)
        self._set_legend_entry_alpha(artist, 1.0 if is_visible else 0.25)
        self.figure.canvas.draw_idle()

    def _create_hover_annotation(self):
        annotation = self.path_ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffdf8", "edgecolor": "#cabfaa", "alpha": 0.3},
            arrowprops={"arrowstyle": "->", "color": "#756b58", "lw": 1.0},
            fontsize=9,
            zorder=20,
            annotation_clip=False,
        )
        self._mark_overlay_artist(annotation)
        return annotation

    def _project_point_to_reference(self, point_x: float, point_y: float, reference) -> dict:
        if reference.size == 1:
            return {
                "x": float(reference.x[0]),
                "y": float(reference.y[0]),
                "theta": float(reference.theta[0]),
                "v": float(reference.v[0]),
                "a": float(reference.a[0]),
                "kappa": float(reference.kappa[0]),
                "s": float(reference.s[0]),
                "segment_index": 0,
                "segment_ratio": 0.0,
            }

        best_projection = None
        best_distance_sq = float("inf")
        query = np.array([point_x, point_y], dtype=float)

        for index in range(reference.size - 1):
            start = np.array([reference.x[index], reference.y[index]], dtype=float)
            end = np.array([reference.x[index + 1], reference.y[index + 1]], dtype=float)
            segment = end - start
            segment_len_sq = float(np.dot(segment, segment))
            if segment_len_sq <= 1e-12:
                ratio = 0.0
                projection = start
            else:
                ratio = float(np.clip(np.dot(query - start, segment) / segment_len_sq, 0.0, 1.0))
                projection = start + ratio * segment

            distance_sq = float(np.dot(query - projection, query - projection))
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_projection = {
                    "x": float(projection[0]),
                    "y": float(projection[1]),
                    "theta": float(reference.theta[index] + ratio * (reference.theta[index + 1] - reference.theta[index])),
                    "v": float(reference.v[index] + ratio * (reference.v[index + 1] - reference.v[index])),
                    "a": float(reference.a[index] + ratio * (reference.a[index + 1] - reference.a[index])),
                    "kappa": float(reference.kappa[index] + ratio * (reference.kappa[index + 1] - reference.kappa[index])),
                    "s": float(reference.s[index] + ratio * (reference.s[index + 1] - reference.s[index])),
                    "segment_index": index,
                    "segment_ratio": ratio,
                }

        return best_projection

    def _build_hover_payload(self, initial_state, reference, solution) -> dict:
        solution_ds = np.hypot(np.diff(solution["x"]), np.diff(solution["y"]))
        solution_s = np.concatenate(([0.0], np.cumsum(solution_ds)))
        reference_items = []
        for index in range(reference.size):
            reference_items.append(
                {
                    "label": "Reference",
                    "index": index,
                    "x": float(reference.x[index]),
                    "y": float(reference.y[index]),
                    "theta": float(reference.theta[index]),
                    "v": float(reference.v[index]),
                    "a": float(reference.a[index]),
                    "kappa": float(reference.kappa[index]),
                    "s": float(reference.s[index]),
                }
            )

        solution_items = []
        for index in range(solution["x"].size):
            time_value = float(solution["time"][index])
            dt_value = float(solution["dt"][index]) if index < solution["dt"].size else float("nan")
            jerk_value = float(solution["jerk"][index]) if index < solution["jerk"].size else float("nan")
            dkappa_value = float(solution["dkappa"][index]) if index < solution["dkappa"].size else float("nan")
            matched_reference = self._project_point_to_reference(float(solution["x"][index]), float(solution["y"][index]), reference)
            solution_items.append(
                {
                    "label": "Optimized",
                    "index": index,
                    "x": float(solution["x"][index]),
                    "y": float(solution["y"][index]),
                    "theta": float(solution["theta"][index]),
                    "v": float(solution["v"][index]),
                    "a": float(solution["a"][index]),
                    "kappa": float(solution["kappa"][index]),
                    "s": float(solution_s[index]),
                    "time": time_value,
                    "dt": dt_value,
                    "jerk": jerk_value,
                    "dkappa": dkappa_value,
                    "track_error": float(
                        np.hypot(solution["x"][index] - matched_reference["x"], solution["y"][index] - matched_reference["y"])
                    ),
                    "heading_error": float(solution["theta"][index] - matched_reference["theta"]),
                    "matched_reference": matched_reference,
                }
            )

        self.current_payload = {
            "initial": {
                "label": "Initial",
                "index": 0,
                "x": float(initial_state.x),
                "y": float(initial_state.y),
                "theta": float(initial_state.theta),
                "v": float(initial_state.v),
                "a": float(initial_state.a),
                "kappa": float(initial_state.kappa),
                "track_error": float(np.hypot(initial_state.x - reference.x[0], initial_state.y - reference.y[0])),
                "heading_error": float(initial_state.theta - reference.theta[0]),
                "matched_reference": {
                    "x": float(reference.x[0]),
                    "y": float(reference.y[0]),
                    "theta": float(reference.theta[0]),
                    "v": float(reference.v[0]),
                    "a": float(reference.a[0]),
                    "kappa": float(reference.kappa[0]),
                    "s": float(reference.s[0]),
                    "segment_index": 0,
                    "segment_ratio": 0.0,
                },
            },
            "reference": reference_items,
            "solution": solution_items,
        }

    def _draw_heading_arrows(self, x_values, y_values, theta_values, color: str, step: int, alpha: float) -> None:
        indices = np.arange(0, x_values.size, max(step, 1))
        if indices[-1] != x_values.size - 1:
            indices = np.append(indices, x_values.size - 1)
        arrow_length = 0.42
        self.path_ax.quiver(
            x_values[indices],
            y_values[indices],
            arrow_length * np.cos(theta_values[indices]),
            arrow_length * np.sin(theta_values[indices]),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0045,
            headwidth=3.6,
            headlength=5.0,
            headaxislength=4.4,
            color=color,
            alpha=alpha,
            zorder=3,
        )

    def _wrap_line(self, text: str, width: int = 36, indent: str = "  ") -> str:
        return textwrap.fill(text, width=width, subsequent_indent=indent)

    def _draw_point_correspondence(self, reference, solution) -> None:
        if self.current_payload is None:
            return

        segments = [
            [
                (payload["matched_reference"]["x"], payload["matched_reference"]["y"]),
                (payload["x"], payload["y"]),
            ]
            for payload in self.current_payload["solution"]
        ]
        collection = LineCollection(
            segments,
            colors="#8d8578",
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.38,
            zorder=3,
            label="Point correspondence",
        )
        self.path_ax.add_collection(collection)

        projected_x = [payload["matched_reference"]["x"] for payload in self.current_payload["solution"]]
        projected_y = [payload["matched_reference"]["y"] for payload in self.current_payload["solution"]]
        self.projected_reference_points = self.path_ax.scatter(
            projected_x,
            projected_y,
            s=18,
            marker="+",
            color="#557a74",
            linewidths=0.9,
            alpha=0.42,
            label="Matched reference points",
            zorder=4,
        )

    def _format_hover_text(self, payload: dict) -> str:
        lines = [
            f"{payload['label']} #{payload['index']}",
            f"x = {payload['x']:.3f} m",
            f"y = {payload['y']:.3f} m",
            f"s = {payload.get('s', 0.0):.3f} m",
            f"theta = {payload['theta']:.3f} rad",
            f"v = {payload['v']:.3f} m/s",
            f"a = {payload['a']:.3f} m/s^2",
            f"kappa = {payload['kappa']:.3f} 1/m",
        ]
        if "time" in payload:
            lines.append(f"t = {payload['time']:.3f} s")
        if "dt" in payload and not np.isnan(payload["dt"]):
            lines.append(f"dt = {payload['dt']:.3f} s")
        if "jerk" in payload and not np.isnan(payload["jerk"]):
            lines.append(f"jerk = {payload['jerk']:.3f} m/s^3")
        if "dkappa" in payload and not np.isnan(payload["dkappa"]):
            lines.append(f"dkappa = {payload['dkappa']:.3f} 1/(m*s)")
        if "track_error" in payload:
            lines.append(f"track err = {payload['track_error']:.3f} m")
        if "heading_error" in payload:
            lines.append(f"heading err = {payload['heading_error']:.3f} rad")
        matched_reference = payload.get("matched_reference")
        if matched_reference is not None:
            lines.extend(
                [
                    "-- matched ref --",
                    f"ref x = {matched_reference['x']:.3f} m",
                    f"ref y = {matched_reference['y']:.3f} m",
                    f"ref s = {matched_reference['s']:.3f} m",
                    f"ref theta = {matched_reference['theta']:.3f} rad",
                ]
            )
        return "\n".join(lines)

    def _update_hover_match_visuals(self, payload) -> bool:
        matched_reference = payload.get("matched_reference")
        if matched_reference is None:
            if self.hover_match_marker is not None:
                self.hover_match_marker.set_visible(False)
            if self.hover_match_line is not None:
                self.hover_match_line.set_visible(False)
            return False

        if self.hover_match_marker is None:
            self.hover_match_marker = self.path_ax.scatter(
                [matched_reference["x"]],
                [matched_reference["y"]],
                s=140,
                marker="+",
                color="#0b4f6c",
                linewidths=2.2,
                alpha=0.95,
                zorder=11,
            )
            self._mark_overlay_artist(self.hover_match_marker)
        else:
            self.hover_match_marker.set_offsets(np.array([[matched_reference["x"], matched_reference["y"]]], dtype=float))
            self.hover_match_marker.set_visible(True)

        if self.hover_match_line is None:
            (self.hover_match_line,) = self.path_ax.plot(
                [matched_reference["x"], payload["x"]],
                [matched_reference["y"], payload["y"]],
                color="#0b4f6c",
                linestyle=(0, (3, 3)),
                linewidth=1.6,
                alpha=0.9,
                zorder=10,
            )
            self._mark_overlay_artist(self.hover_match_line)
        else:
            self.hover_match_line.set_data([matched_reference["x"], payload["x"]], [matched_reference["y"], payload["y"]])
            self.hover_match_line.set_visible(True)
        return True

    def _update_hover_annotation(self, event, scatter_artist, payload_items) -> bool:
        contains, details = scatter_artist.contains(event)
        if not contains or not details.get("ind"):
            return False
        index = int(details["ind"][0])
        payload = payload_items[index]
        self.hover_annotation.xy = (payload["x"], payload["y"])
        self.hover_annotation.set_text(self._format_hover_text(payload))
        self.hover_annotation.set_visible(True)
        return True

    def _find_hover_target(self, event):
        if self.solution_points is not None:
            contains, details = self.solution_points.contains(event)
            if contains and details.get("ind"):
                index = int(details["ind"][0])
                return ("solution", index), self.current_payload["solution"][index]
        if self.reference_points is not None:
            contains, details = self.reference_points.contains(event)
            if contains and details.get("ind"):
                index = int(details["ind"][0])
                return ("reference", index), self.current_payload["reference"][index]
        if self.initial_marker is not None:
            contains, details = self.initial_marker.contains(event)
            if contains and details.get("ind"):
                return ("initial", 0), self.current_payload["initial"]
        return None, None

    def _set_button_state(self, label: str, color: str, hovercolor: str) -> None:
        self.button.label.set_text(label)
        self.button.color = color
        self.button.hovercolor = hovercolor

    def _consume_pending_result(self) -> None:
        if self.pending_result is None and self.pending_error is None:
            return

        if self.poll_timer is not None:
            self.poll_timer.stop()

        self.is_solving = False
        self._set_button_state("Random Init + Solve", "#d7ebe5", "#b8ddd1")

        if self.pending_error is not None:
            error = self.pending_error
            self.pending_error = None
            self.info_ax.clear()
            self.info_ax.axis("off")
            self.info_ax.text(
                0.02,
                0.98,
                f"Solve failed\n\n{error}",
                va="top",
                ha="left",
                fontsize=11,
                family="DejaVu Sans",
                bbox={"boxstyle": "round,pad=0.8", "facecolor": "#fff7f2", "edgecolor": "#d17a5a"},
            )
            self.figure.canvas.draw_idle()
            return

        initial_state, reference, solution = self.pending_result
        self.pending_result = None
        self.run_count += 1
        self._build_hover_payload(initial_state, reference, solution)
        self._render_path(initial_state, reference, solution)
        self._render_dt(reference.dt_ref, solution)
        self._render_dynamics(reference, solution)
        self._render_info(initial_state, reference, solution)
        self.figure.canvas.draw_idle()

    def _solve_in_background(self, seed: Optional[int]) -> None:
        try:
            self.pending_result = run_random_demo(seed=seed)
        except Exception as exc:  # pragma: no cover - interactive path
            self.pending_error = str(exc)

    def _on_hover(self, event) -> None:
        if event.inaxes != self.path_ax or self.current_payload is None:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.hover_target = None
                if self.hover_match_marker is not None:
                    self.hover_match_marker.set_visible(False)
                if self.hover_match_line is not None:
                    self.hover_match_line.set_visible(False)
                self.figure.canvas.draw_idle()
            return

        hover_target, payload = self._find_hover_target(event)
        should_redraw = False

        if hover_target is None:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.hover_target = None
                if self.hover_match_marker is not None:
                    self.hover_match_marker.set_visible(False)
                if self.hover_match_line is not None:
                    self.hover_match_line.set_visible(False)
                should_redraw = True
        elif hover_target != self.hover_target or not self.hover_annotation.get_visible():
            self.hover_annotation.xy = (payload["x"], payload["y"])
            self.hover_annotation.set_text(self._format_hover_text(payload))
            self.hover_annotation.set_visible(True)
            self._update_hover_match_visuals(payload)
            self.hover_target = hover_target
            should_redraw = True

        if should_redraw:
            self.figure.canvas.draw_idle()

    def _on_randomize(self, _event) -> None:
        if self.is_solving:
            return
        self.is_solving = True
        self.pending_result = None
        self.pending_error = None
        self.hover_target = None
        self.hover_annotation.set_visible(False)
        if self.hover_match_marker is not None:
            self.hover_match_marker.set_visible(False)
        if self.hover_match_line is not None:
            self.hover_match_line.set_visible(False)
        self._set_button_state("Solving...", "#e7dac6", "#e7dac6")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        worker = threading.Thread(target=self._solve_in_background, args=(None,), daemon=True)
        worker.start()
        if self.poll_timer is not None:
            self.poll_timer.start()

    def refresh(self, seed: Optional[int] = None) -> None:
        initial_state, reference, solution = run_random_demo(seed=seed)
        self.run_count += 1
        self._build_hover_payload(initial_state, reference, solution)
        self._render_path(initial_state, reference, solution)
        self._render_dt(reference.dt_ref, solution)
        self._render_dynamics(reference, solution)
        self._render_info(initial_state, reference, solution)
        self.figure.canvas.draw_idle()

    def _render_path(self, initial_state, reference, solution) -> None:
        self.path_ax.clear()
        self.legend_artist_map = {}
        self.hover_annotation = self._create_hover_annotation()
        self.hover_annotation.set_visible(False)
        self.hover_target = None
        self.projected_reference_points = None
        self.hover_match_marker = None
        self.hover_match_line = None
        self.path_ax.set_title("Reference vs MPC Path", fontsize=14, fontweight="bold")
        self.path_ax.set_xlabel("x [m]")
        self.path_ax.set_ylabel("y [m]")
        self.path_ax.grid(True, alpha=0.16, linestyle="--", linewidth=0.8)
        self.path_ax.spines[["top", "right"]].set_visible(False)

        reference_line, = self.path_ax.plot(
            reference.x,
            reference.y,
            "--",
            color="#0f766e",
            linewidth=2.2,
            alpha=VISUAL_ALPHA,
            label="Reference path",
            zorder=1,
        )
        reference_line.set_path_effects([patheffects.Stroke(linewidth=3.2, foreground="#edf7f3"), patheffects.Normal()])

        solution_line, = self.path_ax.plot(
            solution["x"],
            solution["y"],
            color="#ca5a34",
            linewidth=2.8,
            alpha=VISUAL_ALPHA,
            label="Optimized path",
            zorder=2,
        )
        solution_line.set_path_effects([patheffects.Stroke(linewidth=4.0, foreground="#fff1ea"), patheffects.Normal()])

        self.reference_points = self.path_ax.scatter(
            reference.x,
            reference.y,
            s=42,
            marker="x",
            color="#0f766e",
            linewidths=1.3,
            alpha=VISUAL_ALPHA,
            label="Reference points",
            zorder=4,
        )
        self.solution_points = self.path_ax.scatter(
            solution["x"],
            solution["y"],
            s=34,
            marker="o",
            facecolors="#ca5a34",
            edgecolors="#fff7f2",
            linewidths=0.8,
            alpha=VISUAL_ALPHA,
            label="Optimized points",
            zorder=5,
        )
        self.initial_marker = self.path_ax.scatter(
            [initial_state.x],
            [initial_state.y],
            s=120,
            color="#d97706",
            edgecolors="#fff7e8",
            linewidths=1.8,
            alpha=VISUAL_ALPHA,
            zorder=7,
            label="Random initial state",
        )

        self._draw_point_correspondence(reference, solution)
        self._draw_heading_arrows(reference.x, reference.y, reference.theta, color="#0f766e", step=6, alpha=VISUAL_ALPHA)
        self._draw_heading_arrows(solution["x"], solution["y"], solution["theta"], color="#ca5a34", step=6, alpha=VISUAL_ALPHA)

        heading_length = 0.6
        self.path_ax.arrow(
            initial_state.x,
            initial_state.y,
            heading_length * np.cos(initial_state.theta),
            heading_length * np.sin(initial_state.theta),
            color="#d97706",
            width=0.04,
            head_width=0.22,
            head_length=0.22,
            length_includes_head=True,
            alpha=VISUAL_ALPHA,
            zorder=8,
        )

        path_legend = self.path_ax.legend(loc="upper left", frameon=True, facecolor="#fffdf8", edgecolor="#dad1bf")
        self._register_legend(
            path_legend,
            {
                "Reference path": reference_line,
                "Optimized path": solution_line,
                "Reference points": self.reference_points,
                "Optimized points": self.solution_points,
                "Random initial state": self.initial_marker,
                "Point correspondence": self.path_ax.collections[0],
                "Matched reference points": self.projected_reference_points,
            },
        )
        self.path_ax.axis("equal")

        all_x = np.concatenate((reference.x, solution["x"], np.array([initial_state.x])))
        all_y = np.concatenate((reference.y, solution["y"], np.array([initial_state.y])))
        margin = 0.8
        self.path_ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
        self.path_ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))

    def _render_dt(self, dt_ref: float, solution) -> None:
        dt_values = solution["dt"]
        indices = np.arange(dt_values.size)
        self.dt_ax.clear()
        self._style_series_axis(self.dt_ax, "dt", "dt [s]")
        point_colors = np.where(dt_values >= dt_ref, "#ca5a34", "#0f766e")
        dt_line, = self.dt_ax.plot(indices, dt_values, color="#7b655a", linewidth=1.8, alpha=0.78, zorder=1, label="optimized dt")
        dt_points = self.dt_ax.scatter(indices, dt_values, c=point_colors, s=34, alpha=0.92, edgecolors="#fffdf8", linewidths=0.8, zorder=2, label="dt points")
        dt_ref_line = self.dt_ax.axhline(dt_ref, color="#0b4f6c", linestyle="--", linewidth=1.8, label=f"dt_ref={dt_ref:.2f}s")
        self.dt_ax.set_xlim(-0.3, max(float(indices[-1]) if dt_values.size else 0.0, 0.0) + 0.3)
        dt_legend = self.dt_ax.legend(loc="upper right")
        self._register_legend(
            dt_legend,
            {
                "optimized dt": dt_line,
                "dt points": dt_points,
                f"dt_ref={dt_ref:.2f}s": dt_ref_line,
            },
        )

    def _render_series_plot(
        self,
        axis,
        title: str,
        ylabel: str,
        values,
        line_color: str,
        marker_face: str,
        reference_values=None,
        reference_label: str | None = None,
        baseline: float | None = None,
    ) -> None:
        axis.clear()
        self._style_series_axis(axis, title, ylabel)
        values = np.asarray(values, dtype=float)
        indices = np.arange(values.size)
        value_line, = axis.plot(indices, values, color=line_color, linewidth=1.85, alpha=0.86, zorder=1, label="optimized")
        value_points = axis.scatter(indices, values, s=28, facecolors=marker_face, edgecolors="#fffdf8", linewidths=0.8, alpha=0.95, zorder=2, label="points")
        reference_line = None
        if reference_values is not None:
            reference_values = np.asarray(reference_values, dtype=float)
            reference_line, = axis.plot(indices, reference_values, color="#0f766e", linestyle="--", linewidth=1.55, alpha=0.7, label=reference_label or "reference", zorder=0)
        baseline_line = None
        if baseline is not None:
            baseline_line = axis.axhline(baseline, color="#0b4f6c", linestyle=":", linewidth=1.4, alpha=0.8, label="baseline")
        axis.set_xlim(-0.3, max(float(indices[-1]) if values.size else 0.0, 0.0) + 0.3)
        if reference_values is not None or baseline is not None:
            legend = axis.legend(loc="upper right", fontsize=8)
            legend_map = {
                "optimized": value_line,
                "points": value_points,
            }
            if reference_line is not None:
                legend_map[reference_label or "reference"] = reference_line
            if baseline_line is not None:
                legend_map["baseline"] = baseline_line
            self._register_legend(legend, legend_map)

    def _render_dynamics(self, reference, solution) -> None:
        self._render_series_plot(
            self.speed_ax,
            "Speed",
            "v [m/s]",
            solution["v"],
            line_color="#ca5a34",
            marker_face="#ca5a34",
            reference_values=reference.v,
            reference_label="ref v",
        )
        self._render_series_plot(
            self.accel_ax,
            "Acceleration",
            "a [m/s²]",
            solution["a"],
            line_color="#d97706",
            marker_face="#d97706",
            reference_values=reference.a,
            reference_label="ref a",
            baseline=0.0,
        )
        self._render_series_plot(
            self.kappa_ax,
            "Curvature",
            "kappa [1/m]",
            solution["kappa"],
            line_color="#8b5cf6",
            marker_face="#8b5cf6",
            reference_values=reference.kappa,
            reference_label="ref kappa",
            baseline=0.0,
        )
        self._render_series_plot(
            self.jerk_ax,
            "Jerk",
            "jerk [m/s³]",
            solution["jerk"],
            line_color="#0b4f6c",
            marker_face="#0b4f6c",
            baseline=0.0,
        )
        self._render_series_plot(
            self.dkappa_ax,
            "Curvature Rate",
            "dkappa [1/(m*s)]",
            solution["dkappa"],
            line_color="#0f766e",
            marker_face="#0f766e",
            baseline=0.0,
        )

    def _render_info(self, initial_state, reference, solution) -> None:
        reference_config = self.current_config["reference"]
        limits_config = self.current_config["limits"]
        weights_config = self.current_config["weights"]
        solver_config = self.current_config["solver"]
        solution_length = float(np.sum(np.hypot(np.diff(solution["x"]), np.diff(solution["y"]))))
        reference_length = float(reference.s[-1])
        total_time = float(solution["time"][-1])
        avg_dt = float(np.mean(solution["dt"]))
        min_dt = float(np.min(solution["dt"]))
        max_dt = float(np.max(solution["dt"]))
        min_v = float(np.min(solution["v"]))
        max_v = float(np.max(solution["v"]))
        min_a = float(np.min(solution["a"]))
        max_a = float(np.max(solution["a"]))
        min_kappa = float(np.min(solution["kappa"]))
        max_kappa = float(np.max(solution["kappa"]))
        terminal_position_error = float(np.hypot(solution["x"][-1] - reference.x[-1], solution["y"][-1] - reference.y[-1]))
        terminal_heading_error = float(solution["theta"][-1] - reference.theta[-1])
        weight_summary = self._wrap_line(
            ", ".join(f"{name}={value:.2f}" for name, value in weights_config.items()),
            width=33,
        )
        segment_summary = self._wrap_line(
            f"segments ({reference_config['segment_count']}): " + " | ".join(reference_config["segment_descriptions"]),
            width=33,
        )
        info_lines = [
            "Randomized TEB-MPC rollout",
            "",
            f"Run count: {self.run_count}",
            f"Reference points: {reference.size}",
            f"Reference length: {reference_length:.2f} m",
            f"Optimized length: {solution_length:.2f} m",
            f"Configured target length: {reference_config['target_length']:.2f} m",
            f"Solve time: {solution['solve_time_ms']:.2f} ms",
            f"Total horizon time: {total_time:.2f} s",
            f"Average dt: {avg_dt:.3f} s",
            f"dt range: [{min_dt:.3f}, {max_dt:.3f}] s",
            f"v range: [{min_v:.3f}, {max_v:.3f}] m/s",
            f"a range: [{min_a:.3f}, {max_a:.3f}] m/s²",
            f"kappa range: [{min_kappa:.3f}, {max_kappa:.3f}] 1/m",
            f"Total cost: {solution['costs']['total']:.2f}",
            f"Track cost: {solution['costs']['track']:.2f}",
            f"Control cost: {solution['costs']['control']:.2f}",
            f"Terminal cost: {solution['costs']['terminal']:.2f}",
            f"Terminal pos err: {terminal_position_error:.3f} m",
            f"Terminal heading err: {terminal_heading_error:.3f} rad",
            "",
            "Reference config",
            self._wrap_line(
                f"ds = {reference_config['ds']:.2f} m, cruise = {reference_config['cruise_speed']:.2f} m/s, dt_ref = {reference_config['dt_ref']:.2f} s",
                width=33,
            ),
            segment_summary,
            "",
            "Controller limits",
            self._wrap_line(f"dt in [{limits_config['dt_min']:.2f}, {limits_config['dt_max']:.2f}] s", width=33),
            self._wrap_line(
                f"v <= {limits_config['max_speed']:.2f} m/s, |a| <= {limits_config['max_accel']:.2f} m/s²",
                width=33,
            ),
            self._wrap_line(
                f"|jerk| <= {limits_config['max_jerk']:.2f} m/s³, |kappa| <= {limits_config['max_kappa']:.2f} 1/m",
                width=33,
            ),
            self._wrap_line(f"|dkappa| <= {limits_config['max_dkappa']:.2f} 1/(m*s)", width=33),
            "",
            "Weights",
            weight_summary,
            "",
            "Solver",
            self._wrap_line(
                f"ipopt_max_iter = {solver_config['ipopt_max_iter']}, tol = {solver_config['ipopt_tol']:.1e}, print = {solver_config['ipopt_print_level']}",
                width=33,
            ),
            "",
            "Initial state",
            f"x = {initial_state.x:.3f} m",
            f"y = {initial_state.y:.3f} m",
            f"theta = {initial_state.theta:.3f} rad",
            f"v = {initial_state.v:.3f} m/s",
            f"a = {initial_state.a:.3f} m/s²",
            f"kappa = {initial_state.kappa:.3f} 1/m",
            "",
            "Hover points for detailed state, timing, and tracking error.",
            "Click the button below to sample a new start state.",
        ]

        self.info_ax.clear()
        self.info_ax.axis("off")
        info_text = self.info_ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            va="top",
            ha="left",
            fontsize=8.8,
            family="DejaVu Sans",
            linespacing=1.08,
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "#fffdf8", "edgecolor": "#d9d2c1"},
        )
        info_text.set_path_effects([patheffects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0.86, 0.83, 0.76), alpha=0.35)])


def main() -> None:
    demo = MatplotlibTEBDemo(first_seed=ARGS.seed)
    if ARGS.save:
        demo.figure.savefig(ARGS.save, dpi=150)
        print(f"Saved matplotlib demo snapshot to {ARGS.save}")
        return
    plt.show()


if __name__ == "__main__":
    main()