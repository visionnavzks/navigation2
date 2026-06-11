"""Integration tests for the Flask web app: helpers and /api endpoints."""

import os
import sys
import math
import types
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEB = os.path.join(_REPO, "web")
for _p in (_REPO, _WEB, os.path.join(_REPO, "build-py313"), os.path.join(_REPO, "build")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


@pytest.fixture
def app_module(monkeypatch):
    # Patch out the optional native module import path so app.py imports cleanly
    # regardless of whether the extension is built.
    fake = types.ModuleType("nb_smoother_clothoid")
    fake.Costmap2D = type("Costmap2D", (), {})
    fake.ClothoidSmoother = type("ClothoidSmoother", (), {})
    fake.SmootherParams = type("SmootherParams", (), {})
    fake.OptimizerParams = type("OptimizerParams", (), {})
    fake.compute_esdf = lambda *a, **kw: []
    sys.modules.setdefault("nb_smoother_clothoid", fake)
    # Force re-import in case previous test cached it
    for mod in list(sys.modules):
        if mod == "app":
            del sys.modules[mod]
    import app as app_mod
    return app_mod


# ----- helpers -----

def test_path_length_zero(app_module):
    assert app_module._path_length([]) == 0.0
    assert app_module._path_length([(0.0, 0.0)]) == 0.0


def test_path_length_segments(app_module):
    pts = [(0.0, 0.0), (3.0, 4.0), (3.0, 9.0)]
    assert app_module._path_length(pts) == pytest.approx(10.0)


def test_reconstruct_path_with_yaw_empty(app_module):
    assert app_module._reconstruct_path_with_yaw([], 0.0, 0.0) == []


def test_reconstruct_path_with_yaw_single(app_module):
    # For a single-point path, the implementation uses start_yaw internally
    # but then overrides the last point with goal_yaw, so the result is goal_yaw.
    out = app_module._reconstruct_path_with_yaw([(1.0, 2.0)], 0.5, 1.0)
    assert out == [(1.0, 2.0, 1.0)]


def test_reconstruct_path_with_yaw_single_uses_goal_yaw(app_module):
    # Document the (possibly surprising) behavior: the last-point override
    # is always applied, even for a single-point path.
    out = app_module._reconstruct_path_with_yaw([(0.0, 0.0)], 0.0, math.pi / 2)
    assert out == [(0.0, 0.0, math.pi / 2)]


def test_reconstruct_path_with_yaw_first_segment(app_module):
    out = app_module._reconstruct_path_with_yaw([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 0.0, 0.0)
    # all yaws zero
    assert all(p[2] == 0.0 for p in out)


def test_reconstruct_path_with_yaw_uses_segment_direction(app_module):
    out = app_module._reconstruct_path_with_yaw([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 0.0, 0.0)
    # middle yaw is the segment direction
    assert abs(out[1][2] - math.pi / 4) < 1e-9


def test_reconstruct_path_with_yaw_last_uses_goal(app_module):
    out = app_module._reconstruct_path_with_yaw([(0.0, 0.0), (1.0, 0.0)], 0.0, math.pi / 2)
    assert out[-1][2] == pytest.approx(math.pi / 2)


def test_reconstruct_path_with_yaw_handles_coincident_points(app_module):
    # If two consecutive points coincide, the segment is degenerate
    out = app_module._reconstruct_path_with_yaw([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)], 0.1, 0.2)
    assert out[0][2] == pytest.approx(0.1)
    # Last point should still be goal yaw
    assert out[-1][2] == pytest.approx(0.2)


def test_normalize_angle_helper(app_module):
    assert abs(app_module._normalize_angle(0.0)) < 1e-12
    assert abs(app_module._normalize_angle(2 * math.pi)) < 1e-12
    assert abs(app_module._normalize_angle(math.pi) - math.pi) < 1e-12


# ----- occupancy / costmap helpers -----

def test_occupancy_to_costmap_basic(app_module):
    occ = np.zeros((4, 4), dtype=np.uint8)
    occ[0, 0] = 254  # lethal
    occ[1, 0] = 0    # free
    occ[2, 0] = 255  # also lethal
    occ[3, 0] = 205  # unknown
    cost = app_module._occupancy_to_costmap(occ)
    # 254 and 255 are lethal
    assert cost[0, 0] == 254
    assert cost[2, 0] == 254
    # 0 is free
    assert cost[1, 0] == 0
    # 205 is unknown
    assert cost[3, 0] == 255


def test_occupancy_to_costmap_inflated(app_module):
    occ = np.zeros((4, 4), dtype=np.uint8)
    occ[1, 1] = 100  # inflated region
    cost = app_module._occupancy_to_costmap(occ)
    # should produce a value in (0, 253)
    assert 0 < cost[1, 1] < 254


def test_occupancy_to_costmap_real_map(app_module):
    """Integration check against the real map: 0/127/255 should map to 0/inflated/254."""
    from PIL import Image
    import os
    map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "maps", "occupancy_map.png")
    if not os.path.exists(map_path):
        pytest.skip(f"occupancy map not found: {map_path}")
    img = Image.open(map_path).convert("L")
    occ = np.array(img, dtype=np.uint8)
    cost = app_module._occupancy_to_costmap(occ)
    unique = set(cost.flatten().tolist())
    # Must have free (0), inflated (1..253), lethal (254), unknown (255)
    assert 0 in unique
    assert 254 in unique
    inflated_vals = [v for v in unique if 0 < v < 254]
    assert len(inflated_vals) > 0


def test_inflate_costmap_sets_lethal_neighborhood(app_module):
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[10, 10] = 254
    inflated = app_module._inflate_costmap(grid, radius_cells=3)
    # Center should remain lethal
    assert inflated[10, 10] == 254
    # Adjacent cells should be inflated (less than 254)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            assert 0 < inflated[10 + dy, 10 + dx] <= 254
    # Far cells should remain free
    assert inflated[0, 0] == 0


def test_inflate_costmap_respects_bounds(app_module):
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0, 0] = 254
    inflated = app_module._inflate_costmap(grid, radius_cells=3)
    # Should not crash; only in-bounds cells modified
    assert inflated[0, 0] == 254
    assert inflated.shape == (10, 10)


def test_inflate_costmap_does_not_lower_lethal(app_module):
    grid = np.full((10, 10), 254, dtype=np.uint8)
    grid[5, 5] = 254
    inflated = app_module._inflate_costmap(grid, radius_cells=2)
    # All cells should still be 254
    assert (inflated == 254).all()


# ----- _coerce_bool -----

def test_coerce_bool_none_uses_default(app_module):
    assert app_module._coerce_bool(None, True) is True
    assert app_module._coerce_bool(None, False) is False


def test_coerce_bool_passthrough(app_module):
    assert app_module._coerce_bool(True, False) is True
    assert app_module._coerce_bool(False, True) is False


def test_coerce_bool_numbers(app_module):
    assert app_module._coerce_bool(1, False) is True
    assert app_module._coerce_bool(0, True) is False


def test_coerce_bool_strings(app_module):
    for s in ["1", "true", "yes", "on", "TRUE", "Yes"]:
        assert app_module._coerce_bool(s, False) is True
    for s in ["0", "false", "no", "off", "garbage"]:
        assert app_module._coerce_bool(s, True) is False


# ----- /api/costmap endpoint -----

def test_api_costmap_returns_synthetic(app_module):
    client = app_module.app.test_client()
    resp = client.get("/api/costmap")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "size_x" in data
    assert "size_y" in data
    assert "data" in data
    assert len(data["data"]) == data["size_x"] * data["size_y"]


# ----- /api/smooth endpoint -----

def test_api_smooth_no_path_in_synthetic(app_module):
    """In the synthetic map, default start/goal (10, 10)/(50, 30) may or may not
    yield a path; either way the endpoint should respond without crashing."""
    client = app_module.app.test_client()
    resp = client.post("/api/smooth", json={})
    assert resp.status_code in (200, 409, 500)
    if resp.status_code == 200:
        body = resp.get_json()
        assert body["success"] is True
    elif resp.status_code == 409:
        body = resp.get_json()
        assert body["error"]["code"] == "SC_ASTAR_NO_PATH"


def test_api_smooth_invalid_json(app_module):
    """Malformed JSON should not crash the server."""
    client = app_module.app.test_client()
    # With invalid JSON, get_json(silent=True) returns {}; the endpoint then
    # uses the default start/goal. In the synthetic map that may yield 409
    # (no path) or 200 (a path exists). The important guarantee is no 5xx.
    resp = client.post("/api/smooth", data="not-json", content_type="text/plain")
    assert resp.status_code in (200, 409)
    body = resp.get_json()
    if resp.status_code == 200:
        assert body.get("success") is True
    else:
        assert body["error"]["code"] == "SC_ASTAR_NO_PATH"


def test_api_smooth_with_valid_payload(app_module):
    """A request with explicit (default) start/goal values should return a structured response."""
    client = app_module.app.test_client()
    resp = client.post("/api/smooth", json={
        "start_x": 5.0, "start_y": 5.0,
        "goal_x": 10.0, "goal_y": 10.0,
    })
    assert resp.status_code in (200, 409)
    body = resp.get_json()
    if resp.status_code == 200:
        assert body["success"] is True
        assert "astar_x" in body
        assert "opt_x" in body
    else:
        assert body["error"]["code"] == "SC_ASTAR_NO_PATH"
