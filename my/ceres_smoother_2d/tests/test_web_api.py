"""Integration tests for the Flask web service.

These tests hit a *running* server (default: http://127.0.0.1:5000).
If no server is reachable, the tests are skipped — run `./run_web.sh` first
or set `WEB_BASE_URL` to the right address.

Run:
    WEB_BASE_URL=http://127.0.0.1:5000 python -m pytest tests/test_web_api.py -v
"""
import json
import os
import time
from urllib.parse import urljoin

import pytest
import requests

WEB_BASE_URL = os.environ.get("WEB_BASE_URL", "http://127.0.0.1:5000")


def _server_up(base_url: str, timeout: float = 1.0) -> bool:
    try:
        r = requests.get(urljoin(base_url, "/api/costmap"), timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(WEB_BASE_URL),
    reason=f"Web server not reachable at {WEB_BASE_URL}. Start it with ./run_web.sh",
)


@pytest.fixture(scope="module")
def base_url():
    return WEB_BASE_URL


@pytest.fixture(scope="module")
def costmap(base_url):
    """Load the costmap meta once per module — saves a round-trip per test."""
    r = requests.get(urljoin(base_url, "/api/costmap"), timeout=5)
    r.raise_for_status()
    return r.json()


def _post_plan(base_url, sx, sy, gx, gy, **params):
    body = {"start": [sx, sy], "goal": [gx, gy]}
    body.update(params)
    return requests.post(
        urljoin(base_url, "/api/plan"),
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )


# ---------- /api/costmap ----------

class TestCostmapEndpoint:
    def test_returns_200(self, base_url):
        r = requests.get(urljoin(base_url, "/api/costmap"), timeout=5)
        assert r.status_code == 200

    def test_returns_json(self, base_url):
        r = requests.get(urljoin(base_url, "/api/costmap"), timeout=5)
        assert r.headers.get("Content-Type", "").startswith("application/json")

    def test_required_fields(self, costmap):
        for k in ("full_width", "full_height", "resolution",
                  "extent_x", "extent_y", "png", "esdf_png"):
            assert k in costmap, f"missing field: {k}"

    def test_dimensions_consistent(self, costmap):
        assert costmap["full_width"] * costmap["resolution"] == pytest.approx(
            costmap["extent_x"], abs=1e-6)
        assert costmap["full_height"] * costmap["resolution"] == pytest.approx(
            costmap["extent_y"], abs=1e-6)

    def test_pngs_are_valid_base64(self, costmap):
        import base64
        for k in ("png", "esdf_png"):
            raw = base64.b64decode(costmap[k])
            # PNG magic bytes
            assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{k} is not a PNG"


# ---------- /api/plan: input validation ----------

class TestPlanInputValidation:
    def test_missing_body(self, base_url):
        r = requests.post(urljoin(base_url, "/api/plan"), timeout=5)
        assert r.status_code in (400, 500)

    def test_missing_start_or_goal(self, base_url):
        r = _post_plan(base_url, 1, 1, 50, 50)
        # Bypass: only one of start/goal
        r2 = requests.post(
            urljoin(base_url, "/api/plan"),
            data=json.dumps({"start": [1, 1]}),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert r2.status_code == 400

    def test_non_numeric_coords(self, base_url):
        r = _post_plan(base_url, "x", "y", 50, 50)
        assert r.status_code == 400

    def test_malformed_json(self, base_url):
        r = requests.post(
            urljoin(base_url, "/api/plan"),
            data="{not-json",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert r.status_code == 400


# ---------- /api/plan: free paths ----------

class TestPlanFreePath:
    def test_straight_line_in_open_corridor(self, base_url):
        # y=28.65m is the widest free corridor in occupancy_map.png (x=6..71)
        r = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["found"] is True, d.get("reason")
        assert d["start_ok"] and d["goal_ok"]
        assert d["raw_points"] >= 2
        assert d["smooth_points"] >= 2
        assert d["plan_ms"] >= 0
        assert d["smooth_ms"] >= 0
        assert d["success"] is True
        assert d["raw_x"][0] == pytest.approx(d["smooth_x"][0], abs=1e-6)
        assert d["raw_y"][0] == pytest.approx(d["smooth_y"][0], abs=1e-6)
        assert d["raw_x"][-1] == pytest.approx(d["smooth_x"][-1], abs=1e-6)
        assert d["raw_y"][-1] == pytest.approx(d["smooth_y"][-1], abs=1e-6)

    def test_vertical_path(self, base_url):
        # (35.2, 14.5) -> (35.2, 38.75) along x=35.2 (tallest free column, 24.8m)
        # robot_radius=0: this narrow column doesn't allow robot inflation.
        r = _post_plan(base_url, 35.2, 14.5, 35.2, 38.75, downsample=3, robot_radius=0)
        d = r.json()
        assert d["found"] is True, d.get("reason")
        assert d["smooth_points"] >= 2

    def test_smooth_length_near_euclidean_for_straight(self, base_url):
        # A* adds zig-zag so smoothed length is slightly longer than Euclidean
        sx, sy, gx, gy = 8, 28.65, 65, 28.65
        eucl = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
        r = _post_plan(base_url, sx, sy, gx, gy, downsample=3, robot_radius=0)
        d = r.json()
        assert d["found"] is True
        # Allow generous slack for A* zig-zag, but the path can't be more than ~2x Euclidean
        assert d["smooth_length"] < 2.0 * eucl
        assert d["smooth_length"] >= eucl * 0.99

    def test_min_clearance_reported(self, base_url):
        r = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3, robot_radius=0)
        d = r.json()
        assert "min_clearance" in d
        assert d["min_clearance"] >= 0.0


# ---------- /api/plan: collision / infeasible cases ----------

class TestPlanCollision:
    def test_start_in_obstacle(self, base_url):
        # (60,10) is inside an obstacle region of the map (after the orientation fix)
        r = _post_plan(base_url, 60, 10, 8, 28.65, downsample=3)
        d = r.json()
        assert d["found"] is False
        assert d["start_ok"] is False
        assert "起点" in d["reason"] or "start" in d["reason"].lower()

    def test_goal_in_obstacle(self, base_url):
        r = _post_plan(base_url, 8, 28.65, 60, 10, downsample=3)
        d = r.json()
        assert d["found"] is False
        assert d["goal_ok"] is False
        assert "终点" in d["reason"] or "goal" in d["reason"].lower()

    def test_both_in_obstacles(self, base_url):
        r = _post_plan(base_url, 60, 10, 65, 10, downsample=3)
        d = r.json()
        assert d["found"] is False
        assert d["start_ok"] is False
        assert d["goal_ok"] is False

    def test_clearance_array_length_matches_smooth(self, base_url):
        r = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3)
        d = r.json()
        if d["found"]:
            assert len(d["clearances"]) == d["smooth_points"]


# ---------- /api/plan: parameter overrides ----------

class TestPlanParameters:
    def test_downsample_changes_ds_points(self, base_url):
        d_small = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=10).json()
        d_large = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=2).json()
        assert d_small["found"] and d_large["found"]
        assert d_small["ds_points"] < d_large["ds_points"]

    def test_higher_iterations_can_reduce_cost(self, base_url):
        d1 = _post_plan(base_url, 8, 28.65, 65, 28.65,
                        downsample=3, max_iterations=5).json()
        d2 = _post_plan(base_url, 8, 28.65, 65, 28.65,
                        downsample=3, max_iterations=200).json()
        assert d1["found"] and d2["found"]
        # More iterations should not make cost worse
        assert d2["final_cost"] <= d1["final_cost"] * 1.01

    def test_invalid_downsample_clamped(self, base_url):
        # Server treats downsample < 1 as 1; should not crash
        r = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=0)
        assert r.status_code == 200


# ---------- Smoke / performance ----------

class TestSmoke:
    def test_response_time_reasonable(self, base_url):
        t0 = time.perf_counter()
        r = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3)
        wall = time.perf_counter() - t0
        assert wall < 10.0, f"plan took {wall:.2f}s wall time"
        assert r.status_code == 200

    def test_repeated_calls_idempotent(self, base_url):
        a = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3).json()
        b = _post_plan(base_url, 8, 28.65, 65, 28.65, downsample=3).json()
        # Paths should be the same length and same endpoints
        assert a["found"] and b["found"]
        assert a["smooth_length"] == pytest.approx(b["smooth_length"], abs=0.05)
        assert a["raw_x"][0] == pytest.approx(b["raw_x"][0], abs=1e-6)
        assert a["raw_x"][-1] == pytest.approx(b["raw_x"][-1], abs=1e-6)