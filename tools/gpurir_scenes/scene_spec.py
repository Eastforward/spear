"""Deterministic per-scene composition: seed -> SceneSpec.

Fixed conventions:
  - Room 5.2 x 4.4 x 2.8 m (constant across scenes).
  - Mic at room center, height 1.2 m. Mic-forward = +Y = window direction.
  - Each scene has 1 or 2 animals; distribution 50/50.
  - Animated animals get a smooth trajectory (10 anchor pts, cubic interp);
    static animals get a random position with wall + mic margins.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP  # noqa: E402
from gpurir_scenes.furniture_map import any_bbox_hits_series  # noqa: E402


def _forward_yaw_offset_for_tag(tag):
    """Look up per-tag rig-specific walking yaw offset.

    Kept as a helper (not inlined) so unit tests can monkey-patch
    ANIMATED_RIG_MAP entries to prove _generate_trajectory really reads
    the per-tag value, not a hard-coded constant.
    """
    return ANIMATED_RIG_MAP[tag]["walking_forward_yaw_offset_deg"]


ANIMATED_TAGS = list(ANIMATED_RIG_MAP.keys())
STATIC_TAGS = list(STATIC_MESH_MAP.keys())
ALL_TAGS = ANIMATED_TAGS + STATIC_TAGS

ROOM_SIZE_M = (5.2, 4.4, 2.8)
T60_S = 0.45
MIC_POS_M = (ROOM_SIZE_M[0] / 2.0, ROOM_SIZE_M[1] / 2.0, 1.2)
N_FRAMES = 75
FPS = 15
WALL_MARGIN_M = 0.5
MIC_MARGIN_M = 1.0
ANIMAL_MIN_SEP_M = 1.0  # center separation; mild animal overlap is acceptable
WALL_CLEARANCE_M = 0.20
ANIMAL_CLEARANCE_M = 0.15
SOURCE_HEIGHT_M = 0.45  # dog-mouth-ish; audio source height
STATIC_ACTOR_Z_M = 0.0  # actor on floor (visual only; audio still uses source height)

# Legacy alias for the Quaternius rig family (Dog, Cat). Both random and
# hand-written scenes now read the offset per-tag from
# species_rig_map.ANIMATED_RIG_MAP[tag]["walking_forward_yaw_offset_deg"],
# so this constant is only kept for:
#   (a) backward compat with scene_two_dogs (which aliases this name), and
#   (b) documentation.
# Do NOT reference this constant when computing yaw for a rig-based animal --
# use _forward_yaw_offset_for_tag(tag) instead, or the corresponding
# ANIMATED_RIG_MAP field.
ANIM_FORWARD_YAW_OFFSET_DEG = 0.0   # 2026-07-08: was 180.0; see species_rig_map.QUATERNIUS_FORWARD_YAW_OFFSET_DEG

TRAJ_ANCHORS = 10
PLACEMENT_TRIES = 200
LOCAL_TRAJ_TRIES = 500
EPS_M = 1e-6

ANIMAL_FOOTPRINT_RADIUS_M = {
    "cat_persian": 0.30,
    "cat_tabby": 0.30,
    "cat_british_shorthair_v2": 0.30,
    "chipmunk": 0.22,
    "dog_golden": 0.45,
    "dog_beagle_v2": 0.38,
    "goat": 0.45,
    "sheep": 0.50,
    "pig": 0.45,
    "horse": 0.65,
    "cattle_bovinae": 0.60,
    "yak": 0.65,
    "donkey_ass": 0.55,
}
DEFAULT_FOOTPRINT_RADIUS_M = 0.45


@dataclass
class AnimalPlacement:
    tag: str
    is_animated: bool
    trajectory_m: Optional[np.ndarray] = None
    yaw_deg: Optional[np.ndarray] = None
    static_pos_m: Optional[tuple] = None
    static_yaw_deg: Optional[float] = None
    # For rig-based animals (dogs/cats/chipmunk): which anim to play on the
    # spawned actor. Default "Walking"; use "Idle" for a stationary dog.
    wanted_anim: str = "Walking"
    # Runtime render hints for non-Quaternius animated rigs such as Mixamo
    # humans. Leave None for legacy dog/cat defaults.
    actor_scale: Optional[float] = None
    actor_z_lift_cm: Optional[float] = None
    walking_forward_yaw_offset_deg: Optional[float] = None
    animation_play_rate: Optional[float] = None
    # Optional UE runtime grounding for human dataset renders.  The actor is
    # still placed at scale 1 on the apartment floor; after animation pose
    # evaluation, only its world Z is corrected so the current lowest mesh
    # bound touches the measured floor.
    ground_snap_to_floor: bool = False
    ground_snap_max_abs_correction_cm: float = 15.0


@dataclass
class SceneSpec:
    seed: int
    room_size_m: tuple = ROOM_SIZE_M
    t60_s: float = T60_S
    mic_pos_m: tuple = MIC_POS_M
    animals: list = field(default_factory=list)


def animal_footprint_radius_m(tag):
    return ANIMAL_FOOTPRINT_RADIUS_M.get(tag, DEFAULT_FOOTPRINT_RADIUS_M)


def _required_wall_slack_m(tag, wall_margin_m=WALL_MARGIN_M):
    return max(wall_margin_m, animal_footprint_radius_m(tag) + WALL_CLEARANCE_M)


def _required_center_sep_m(tag_a, tag_b, min_sep_m=ANIMAL_MIN_SEP_M):
    radius_sum = animal_footprint_radius_m(tag_a) + animal_footprint_radius_m(tag_b)
    footprint_sep = radius_sum + ANIMAL_CLEARANCE_M
    if tag_a.startswith("dog_") and tag_b.startswith("dog_"):
        return max(min_sep_m, footprint_sep)
    return footprint_sep


def _sample_static_pos(rng, room_size_m, mic_pos_m, tag):
    rx, ry, _ = room_size_m
    wall_slack_m = _required_wall_slack_m(tag)
    for _ in range(200):
        x = rng.uniform(wall_slack_m, rx - wall_slack_m)
        y = rng.uniform(wall_slack_m, ry - wall_slack_m)
        if ((x - mic_pos_m[0]) ** 2 + (y - mic_pos_m[1]) ** 2) ** 0.5 >= MIC_MARGIN_M:
            # Audio side uses SOURCE_HEIGHT_M for the sample position (mouth height);
            # visual actor Z is separately set to floor.
            return (float(x), float(y), SOURCE_HEIGHT_M)
    raise RuntimeError("could not sample a static pose within margins after 200 tries")


def _sample_static_pos_grid(room_size_m, mic_pos_m, tag, existing_animals):
    rx, ry, _ = room_size_m
    wall_slack_m = _required_wall_slack_m(tag)
    best_clearance = -float("inf")
    best_xy = None
    for x in np.linspace(wall_slack_m, rx - wall_slack_m, 61):
        for y in np.linspace(wall_slack_m, ry - wall_slack_m, 61):
            if ((x - mic_pos_m[0]) ** 2 + (y - mic_pos_m[1]) ** 2) ** 0.5 < MIC_MARGIN_M:
                continue
            series = np.array([[x, y]])
            clearance = _min_pairwise_clearance_series(tag, series, existing_animals)
            if clearance > best_clearance:
                best_clearance = clearance
                best_xy = (float(x), float(y))
    if best_xy is not None and best_clearance + EPS_M >= 0.0:
        return (best_xy[0], best_xy[1], SOURCE_HEIGHT_M)
    raise RuntimeError(
        f"could not place static fallback position for {tag}: best clearance {best_clearance:.3f} m"
    )


def _generate_trajectory(rng, room_size_m, tag):
    """10 random anchors, cubic-spline to 75 frames."""
    rx, ry, _ = room_size_m
    wall_slack_m = _required_wall_slack_m(tag)
    anchors = np.empty((TRAJ_ANCHORS, 2))
    for i in range(TRAJ_ANCHORS):
        anchors[i, 0] = rng.uniform(wall_slack_m, rx - wall_slack_m)
        anchors[i, 1] = rng.uniform(wall_slack_m, ry - wall_slack_m)
    ts = np.linspace(0.0, 1.0, TRAJ_ANCHORS)
    tf = np.linspace(0.0, 1.0, N_FRAMES)
    cs_x = CubicSpline(ts, anchors[:, 0])
    cs_y = CubicSpline(ts, anchors[:, 1])
    xs = np.clip(cs_x(tf), wall_slack_m, rx - wall_slack_m)
    ys = np.clip(cs_y(tf), wall_slack_m, ry - wall_slack_m)
    zs = np.full(N_FRAMES, SOURCE_HEIGHT_M)
    traj = np.stack([xs, ys, zs], axis=1)
    # Yaw: tangent direction of the smooth spline + per-tag rig offset. This
    # per-tag lookup replaces the global constant so a future non-Quaternius
    # rig (Mixamo, custom) can declare its own offset without silently
    # inheriting Quaternius's 180.
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    motion_deg = np.degrees(np.arctan2(dy, dx))
    offset = _forward_yaw_offset_for_tag(tag)
    yaw = (motion_deg + offset) % 360.0
    return traj, yaw


def _generate_local_trajectory(rng, room_size_m, tag, existing_animals):
    """Small fallback walk contained in a safe patch of floor.

    Full-room random splines can be hard to place once another animal already
    occupies a long trajectory. This keeps the animal animated while giving the
    collision validator a compact candidate to approve.
    """
    rx, ry, _ = room_size_m
    half_len_m = 0.25
    wall_slack_m = _required_wall_slack_m(tag) + half_len_m
    tf = np.linspace(0.0, 1.0, N_FRAMES)
    for _ in range(LOCAL_TRAJ_TRIES):
        cx = rng.uniform(wall_slack_m, rx - wall_slack_m)
        cy = rng.uniform(wall_slack_m, ry - wall_slack_m)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        phase = np.sin(2.0 * np.pi * tf) * half_len_m
        xs = cx + phase * np.cos(heading)
        ys = cy + phase * np.sin(heading)
        zs = np.full(N_FRAMES, SOURCE_HEIGHT_M)
        traj = np.stack([xs, ys, zs], axis=1)
        if _min_pairwise_clearance_series(tag, traj[:, :2], existing_animals) < 0.0:
            continue
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        motion_deg = np.degrees(np.arctan2(dy, dx))
        offset = _forward_yaw_offset_for_tag(tag)
        yaw = (motion_deg + offset) % 360.0
        return traj, yaw
    raise RuntimeError(
        f"could not place local fallback trajectory for {tag} after {LOCAL_TRAJ_TRIES} tries"
    )


def _placement_xy_series(placement):
    """Return an (N, 2) xy trajectory. Static animals become (1, 2)."""
    if placement.is_animated:
        return placement.trajectory_m[:, :2]
    return np.array([[placement.static_pos_m[0], placement.static_pos_m[1]]])


def _min_xy_distance_series(new_series, existing_animals):
    """Return min (xy) distance between new_series (K,2) and every frame of
    every already-placed animal. Static-vs-animated compares against all
    animated frames. Animated-vs-animated compares same frame only, which is
    the visible collision condition for videos.
    """
    if not existing_animals:
        return float("inf")
    m = float("inf")
    for a in existing_animals:
        other = _placement_xy_series(a)
        d = _collision_distances(new_series, other)
        m = min(m, float(d.min()))
    return m


def _collision_distances(series_a, series_b):
    if len(series_a) > 1 and len(series_a) == len(series_b):
        return np.linalg.norm(series_a - series_b, axis=1)
    return np.linalg.norm(series_a[:, None, :] - series_b[None, :, :], axis=2).reshape(-1)


def _min_pairwise_clearance_series(new_tag, new_series, existing_animals, min_sep_m=ANIMAL_MIN_SEP_M):
    if not existing_animals:
        return float("inf")
    m = float("inf")
    for a in existing_animals:
        other = _placement_xy_series(a)
        d = _collision_distances(new_series, other)
        required = _required_center_sep_m(new_tag, a.tag, min_sep_m=min_sep_m)
        m = min(m, float(d.min()) - required)
    return m


def _min_wall_margin_series(new_series, room_size_m):
    """Signed slack from each xy pt to the nearest wall. Positive = safe."""
    rx, ry, _ = room_size_m
    left = new_series[:, 0].min()
    right = rx - new_series[:, 0].max()
    front = new_series[:, 1].min()
    back = ry - new_series[:, 1].max()
    return float(min(left, right, front, back))


def check_no_clipping(spec, wall_margin_m=WALL_MARGIN_M, min_sep_m=ANIMAL_MIN_SEP_M,
                      furniture_bboxes=None):
    """Raise AssertionError if any animal in spec clips a wall or another
    animal at ANY frame. Meant to be called AFTER placement (e.g. by
    hand-authored scenes) and also used internally by compose_scene() during
    per-candidate placement.

    If furniture_bboxes is provided (list of FurnitureBBox from
    furniture_map.load_apartment_furniture), also assert that no animal
    trajectory point enters any furniture AABB. Passing None (default)
    preserves the pre-furniture behaviour (wall + pairwise only).
    """
    animals = list(spec.animals)
    for i, a in enumerate(animals):
        s = _placement_xy_series(a)
        slack = _min_wall_margin_series(s, spec.room_size_m)
        required_wall_slack_m = _required_wall_slack_m(a.tag, wall_margin_m=wall_margin_m)
        assert slack + EPS_M >= required_wall_slack_m, (
            f"{a.tag} clips wall: min wall slack = {slack:.2f} m "
            f"(need >= {required_wall_slack_m:.2f})"
        )
        others = animals[:i]
        if others:
            min_clearance = _min_pairwise_clearance_series(a.tag, s, others, min_sep_m=min_sep_m)
            assert min_clearance + EPS_M >= 0.0, (
                f"{a.tag} too close to another animal: clearance = {min_clearance:.2f} m "
                f"(need >= 0.00)"
            )
        if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, s):
            raise AssertionError(
                f"{a.tag} clips furniture at some frame "
                f"(bbox_count={len(furniture_bboxes)})"
            )


def compose_scene(seed: int, furniture_bboxes=None) -> SceneSpec:
    rng = np.random.default_rng(seed)
    n_animals = int(rng.choice([1, 2]))
    tag_choices = rng.choice(ALL_TAGS, size=n_animals, replace=False)
    animals = []
    for tag in tag_choices:
        tag = str(tag)
        is_animated = tag in ANIMATED_TAGS
        placed = False
        # Anti-overlap: retry if placement is too close to
        # an already-placed animal or (if provided) any furniture bbox.
        for _ in range(PLACEMENT_TRIES):
            if is_animated:
                traj, yaw = _generate_trajectory(rng, ROOM_SIZE_M, tag)
                cand_series = traj[:, :2]
                if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, cand_series):
                    continue
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
                    animals.append(AnimalPlacement(
                        tag=tag, is_animated=True,
                        trajectory_m=traj, yaw_deg=yaw,
                    ))
                    placed = True
                    break
            else:
                pos = _sample_static_pos(rng, ROOM_SIZE_M, MIC_POS_M, tag)
                cand_series = np.array([[pos[0], pos[1]]])
                if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, cand_series):
                    continue
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
                    yaw_val = float(rng.uniform(0.0, 360.0))
                    animals.append(AnimalPlacement(
                        tag=tag, is_animated=False,
                        static_pos_m=pos, static_yaw_deg=yaw_val,
                    ))
                    placed = True
                    break
        if not placed and is_animated:
            traj, yaw = _generate_local_trajectory(rng, ROOM_SIZE_M, tag, animals)
            # Local trajectory fallback must ALSO honour furniture — otherwise
            # a scene that failed the main loop 200 times because of furniture
            # would sneak the animal in via the fallback and clip anyway.
            if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, traj[:, :2]):
                pass  # skip fallback; leave placed=False; downstream handles.
            else:
                animals.append(AnimalPlacement(
                    tag=tag, is_animated=True,
                    trajectory_m=traj, yaw_deg=yaw,
                ))
                placed = True
        if not placed and not is_animated:
            try:
                pos = _sample_static_pos_grid(ROOM_SIZE_M, MIC_POS_M, tag, animals)
            except RuntimeError:
                pos = None
            if pos is not None:
                if furniture_bboxes and any_bbox_hits_series(
                    furniture_bboxes, np.array([[pos[0], pos[1]]])
                ):
                    pos = None  # grid pick hits furniture, drop and fall through
            if pos is not None:
                animals.append(AnimalPlacement(
                    tag=tag,
                    is_animated=False,
                    static_pos_m=pos,
                    static_yaw_deg=float(rng.uniform(0.0, 360.0)),
                ))
                placed = True
        if not placed and animals:
            # Prefer a valid one-animal scene over forcing a clipped second
            # placement in a room that has no safe slot left.
            continue
        if not placed:
            raise RuntimeError(
                f"could not place {tag} without clipping after {PLACEMENT_TRIES} tries"
            )
    spec = SceneSpec(seed=seed, animals=animals)
    check_no_clipping(spec, furniture_bboxes=furniture_bboxes)
    return spec


if __name__ == "__main__":
    import json
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    spec = compose_scene(seed=seed)
    out = {
        "seed": spec.seed,
        "room_size_m": list(spec.room_size_m),
        "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {"tag": a.tag, "is_animated": a.is_animated,
             "static_pos_m": list(a.static_pos_m) if a.static_pos_m else None,
             "static_yaw_deg": a.static_yaw_deg,
             "trajectory_shape": list(a.trajectory_m.shape) if a.trajectory_m is not None else None}
            for a in spec.animals
        ],
    }
    print(json.dumps(out, indent=2))
