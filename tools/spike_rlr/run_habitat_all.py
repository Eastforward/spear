"""C group: full-stack Habitat + RLR.

Everything happens inside a single Habitat simulator:
  - shoebox_v2 GLB loaded as stage mesh
  - 2 Quaternius Dog GLBs loaded as rigid objects, moved each frame by
    set_translation()/set_rotation() (T-pose; no skeletal animation --
    the "ice-skating" limitation is a known trade-off documented in the
    spike plan)
  - 4 RGB sensors on the agent, one per view yaw (0/90/180/270 deg)
  - 1 AudioSensor (Ambisonic FOA, same config as B group)

Per-frame we snapshot RGB from each of the 4 view sensors, then advance
the audio source positions and grab the FOA IR. At the end we convolve
+ mix like the other backends and ffmpeg-mux 4 stereo videos.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DOG_GLB   = REPO_ROOT.parents[1] / "assets" / "mesh_library" / "quaternius_animalpack" / "Dog.glb"


# --- reuse the RLR audio code -----------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_audio_pass_rlr import (  # noqa: E402
    _load_dry_source, _habitat_from_scene, _make_rlr_materials_json,
    _TAG_AUDIO_OVERRIDES,
)
from scene_two_dogs_v2 import compose_two_dog_scene_v2  # noqa: E402

# habitat-sim  (requires ss2 env + LD_PRELOAD system libEGL/libGLdispatch)
import habitat_sim  # noqa: E402
from habitat_sim.sensor import RLRAudioPropagationChannelLayoutType  # noqa: E402
import quaternion  # noqa: E402


# --- coordinate helpers -----------------------------------------------------
def _yaw_quat(yaw_deg):
    """Yaw about world Y (up) in Habitat coords, degrees. Returns a
    numpy quaternion (for agent.state.rotation)."""
    yaw_rad = np.radians(yaw_deg)
    return np.quaternion(np.cos(yaw_rad / 2), 0.0, np.sin(yaw_rad / 2), 0.0)


def _yaw_magnum(yaw_deg):
    """Same as _yaw_quat but returns a magnum.Quaternion for
    RigidObject.rotation (which requires the magnum type, not numpy)."""
    import magnum as mn
    yaw_rad = np.radians(yaw_deg)
    return mn.Quaternion.rotation(mn.Rad(float(yaw_rad)), mn.Vector3.y_axis())


def _make_render_view_specs(width, height, fov_deg):
    """4 RGB sensors, one per view yaw.

    The agent is already yawed 180 deg about Y (see _place_mic) so its
    forward (-Z_local) points at +Z_habitat = +Y_scene = SSOT view0
    (window direction). Sensor orientation is applied ON TOP of the agent
    frame, so view0 gets orientation=(0,0,0) to inherit the agent's
    forward; the other views rotate +90/180/270 about the agent's local Y.
    """
    view_yaws = {"view0": 180, "view1": 270, "view2": 0, "view3": 90}
    specs = []
    for name, yaw in view_yaws.items():
        s = habitat_sim.CameraSensorSpec()
        s.uuid = name
        s.sensor_type = habitat_sim.SensorType.COLOR
        s.resolution = [height, width]
        s.hfov = fov_deg
        s.orientation = [0.0, np.radians(yaw), 0.0]
        s.position = [0.0, 0.0, 0.0]
        specs.append(s)
    return specs


def _make_audio_spec(sample_rate, indirect_ray_count=100):
    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.enableMaterials = False
    audio_spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
    audio_spec.channelLayout.channelCount = 4
    audio_spec.acousticsConfig.sampleRate = sample_rate
    audio_spec.acousticsConfig.threadCount = 4
    audio_spec.acousticsConfig.direct = True
    audio_spec.acousticsConfig.indirect = True
    audio_spec.acousticsConfig.diffraction = True
    audio_spec.acousticsConfig.transmission = True
    audio_spec.acousticsConfig.temporalCoherence = True
    audio_spec.acousticsConfig.indirectRayCount = indirect_ray_count
    audio_spec.acousticsConfig.sourceRayCount = 200
    audio_spec.acousticsConfig.indirectRayDepth = 50
    audio_spec.acousticsConfig.frequencyBands = 4
    audio_spec.acousticsConfig.unitScale = 1.0
    return audio_spec


def _build_sim(spec, glb_path, materials_json_path, indirect_ray_count=100):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb_path)
    sim_cfg.enable_physics = True  # need physics enabled to spawn rigid objects
    # Use the built-in default lighting instead of NO_LIGHTS (default). This
    # gives Habitat's headlight + ambient so the scene isn't pitch black.
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim_cfg.override_scene_light_defaults = True

    agent_cfg = habitat_sim.AgentConfiguration()
    render_specs = _make_render_view_specs(
        spec["render_config"]["width"],
        spec["render_config"]["height"],
        spec["camera_configs"][0]["fov_deg"],
    )
    audio_spec = _make_audio_spec(spec["audio_config"]["sample_rate_hz"],
                                    indirect_ray_count=indirect_ray_count)
    agent_cfg.sensor_specifications = list(render_specs) + [audio_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON(str(materials_json_path))
    return sim


def _place_mic(sim, mic_pos_scene):
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = _habitat_from_scene(mic_pos_scene)
    state.rotation = np.quaternion(0.0, 0.0, 1.0, 0.0)  # face +Y_scene = +Z_hab
    agent.set_state(state)


def _spawn_dog(sim, glb_path, init_pos_scene, scale=0.10):
    """Load Dog.glb as a rigid object and return the handle.

    Quaternius Dog raw mesh is ~8.4m long -- game asset scale. Real dog is
    ~0.9m long, so we scale by ~0.1 to get realistic size.
    """
    obj_mgr = sim.get_object_template_manager()
    template = obj_mgr.create_new_template(str(glb_path))
    template.render_asset_handle = str(glb_path)
    template.scale = np.array([scale, scale, scale], dtype=np.float32)
    template_id = obj_mgr.register_template(template, f"dog_{init_pos_scene[0]:.2f}_{init_pos_scene[1]:.2f}")

    rigid_obj_mgr = sim.get_rigid_object_manager()
    obj = rigid_obj_mgr.add_object_by_template_id(template_id)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = _habitat_from_scene(init_pos_scene)
    return obj


def _set_dog_pose(dog_obj, pos_scene, yaw_deg=0.0):
    dog_obj.translation = _habitat_from_scene(pos_scene)
    dog_obj.rotation = _yaw_magnum(yaw_deg)


def run_c_group(quality_mode="low", verbose=True):
    spec_path = REPO_ROOT / "data" / "shoebox_v2_spec.json"
    glb_path = REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2_mesh.glb"
    materials_sidecar = REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2_materials.json"
    rlr_materials_json = materials_sidecar.with_name(
        materials_sidecar.stem + "_rlr.json"
    )

    with open(spec_path) as f:
        spec = json.load(f)
    with open(materials_sidecar) as f:
        materials_sidecar_data = json.load(f)
    _make_rlr_materials_json(materials_sidecar_data, rlr_materials_json)

    sample_rate = spec["audio_config"]["sample_rate_hz"]
    duration_s = spec["audio_config"]["duration_s"]
    n_frames = spec["render_config"]["n_frames"]
    fps = spec["render_config"]["fps"]
    n_samples_total = int(round(sample_rate * duration_s))
    samples_per_frame = int(round(sample_rate / fps))

    ray_counts = {"low": 100, "high": 500, "max": 5000}
    ir_ray = ray_counts.get(quality_mode, 100)

    sim = _build_sim(spec, glb_path, rlr_materials_json, indirect_ray_count=ir_ray)
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    _place_mic(sim, spec["mic"]["pos_m"])

    # Compose scene / trajectories
    scene = compose_two_dog_scene_v2(spec_path)

    # Spawn dog rigid objects
    dogs = {}
    for a in scene.animals:
        obj = _spawn_dog(sim, DOG_GLB, a.trajectory_m[0])
        dogs[a.tag] = obj
        if verbose:
            print(f"[c-group] spawned {a.tag} at {a.trajectory_m[0]}")

    # Load dry sources for each dog
    dry = {a.tag: _load_dry_source(a.tag, sample_rate, duration_s)
           for a in scene.animals}

    # Frame loop: RGB + audio IR per frame
    frames_by_view = {f"view{i}": [] for i in range(4)}
    per_source_wet = {a.tag: np.zeros((4, n_samples_total), dtype=np.float32)
                       for a in scene.animals}

    t0 = time.time()
    for f in range(n_frames):
        # Update dog positions
        for a in scene.animals:
            pos = a.trajectory_m[f]
            yaw = float(a.yaw_deg[f])
            _set_dog_pose(dogs[a.tag], pos, yaw_deg=yaw)

        # Render one audio IR per source per frame. The AudioSensor lives on
        # the agent (fixed mic), so we sweep the source position for each dog.
        source_irs = {}
        for a in scene.animals:
            src_hab = _habitat_from_scene(a.trajectory_m[f])
            audio_sensor.setAudioSourceTransform(src_hab)
            obs = sim.get_sensor_observations()

            # Also collect RGB frames (only once per frame, from the first
            # source we process)
            if a.tag == scene.animals[0].tag:
                for i in range(4):
                    view_key = f"view{i}"
                    rgb = obs[view_key]  # HxWx4 (RGBA)
                    frames_by_view[view_key].append(rgb[..., :3].astype(np.uint8))

            ir_raw = obs["audio_sensor"]
            ir = np.asarray(ir_raw, dtype=np.float32)
            if ir.ndim == 2 and ir.shape[0] > ir.shape[1]:
                ir = ir.T
            elif ir.ndim == 1:
                ir = ir[None, :]
            source_irs[a.tag] = ir

        # Convolve this frame's dry chunk against each source's IR
        for a in scene.animals:
            ir = source_irs[a.tag]
            frame_start = f * samples_per_frame
            frame_end = min(frame_start + samples_per_frame, n_samples_total)
            dry_chunk = dry[a.tag][frame_start:frame_end]
            if len(dry_chunk) == 0:
                continue
            for c in range(4):
                wet_chunk = np.convolve(dry_chunk, ir[c], mode="full")
                w_end = min(frame_start + len(wet_chunk), n_samples_total)
                per_source_wet[a.tag][c, frame_start:w_end] += \
                    wet_chunk[:w_end - frame_start]

        if verbose and f % 15 == 0:
            print(f"[c-group] frame {f}/{n_frames}  elapsed={time.time()-t0:.1f}s")

    # ---- write audio ----
    total = sum(per_source_wet.values())
    peak = float(np.abs(total).max()) or 1.0
    total = (total * (0.9 / peak)).astype(np.float32)

    audio_out_dir = REPO_ROOT / "tmp" / "spike_output" / "raw_audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    foa_path = audio_out_dir / "audio_C_habitat_FOA.wav"
    sf.write(str(foa_path), total.T, sample_rate, subtype="PCM_16")

    # stereo downmix using ch3 (see B group notes for channel ordering)
    W = total[0]
    lr = total[3]
    L = W - 0.707 * lr
    R = W + 0.707 * lr
    stereo = np.stack([L, R], axis=1)
    peak = np.abs(stereo).max() or 1.0
    stereo = stereo * (0.9 / peak)
    stereo_path = audio_out_dir / "audio_C_habitat_stereo.wav"
    sf.write(str(stereo_path), stereo, sample_rate, subtype="PCM_16")
    print(f"[c-group] wrote {foa_path}, {stereo_path}")

    # per-source stereo tracks (for A/B/C spectrogram parity)
    for tag, buf in per_source_wet.items():
        solo_peak = float(np.abs(buf).max()) or 1.0
        buf_n = buf * (0.9 / solo_peak)
        sf.write(str(audio_out_dir / f"audio_C_habitat_FOA_{tag}_FOA.wav"),
                 buf_n.T, sample_rate, subtype="PCM_16")
        W = buf_n[0]; lr = buf_n[3]
        L = W - 0.707 * lr; R = W + 0.707 * lr
        st = np.stack([L, R], axis=1)
        p = np.abs(st).max() or 1.0
        st = st * (0.9 / p)
        sf.write(str(audio_out_dir / f"audio_C_habitat_FOA_{tag}_stereo.wav"),
                 st, sample_rate, subtype="PCM_16")
        print(f"[c-group] wrote SOLO {tag}")

    # ---- write video: 4 views ----
    import imageio.v2 as imageio
    tmp_frames = REPO_ROOT / "tmp" / "spike_rlr" / "habitat_frames"
    tmp_frames.mkdir(parents=True, exist_ok=True)
    videos_out = REPO_ROOT / "tmp" / "spike_output" / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)

    for view_key, frames in frames_by_view.items():
        view_dir = tmp_frames / view_key
        view_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(frames):
            imageio.imwrite(str(view_dir / f"frame_{i:03d}.png"), im)
        silent_mp4 = videos_out / f"C_habitat_{view_key}_silent.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(view_dir / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(silent_mp4),
        ], check=True)
        # mux with mixed stereo
        muxed = videos_out / f"C_habitat_{view_key}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_mp4),
            "-i", str(stereo_path),
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v", "-map", "1:a", "-shortest",
            str(muxed),
        ], check=True)
        print(f"[c-group] wrote {muxed}")

    elapsed = time.time() - t0
    print(f"[c-group] TOTAL wall time: {elapsed:.1f}s")
    return {"wall_time_s": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="low", choices=["low", "high", "max"])
    args = ap.parse_args()
    run_c_group(quality_mode=args.quality)


if __name__ == "__main__":
    main()
