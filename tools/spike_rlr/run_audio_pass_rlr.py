"""B group: RLR audio backend via habitat-sim AudioSensor (档 ① swap-in).

We import habitat_sim only to reach the RLR audio bindings. We do NOT
render anything from Habitat here — the video comes from A group's UE
render pass. Per-frame we:

  1. Move the audio source (dog_husky or dog_golden) to its current xyz
  2. Call sim.get_sensor_observations()["audio"] to get the RIR (FOA 4ch)
  3. Convolve dry source audio with the RIR to get the wet contribution
     for that source at that frame's timestamp
  4. Overlap-add wet frames into a per-source output buffer

Then mix the two sources and write a 4ch WAV.

Coordinate frame note: Habitat uses right-handed Y-up. Our SSOT is also
right-handed Y-up. Habitat's own convention for scene loading may treat
+Y as up (gravity down) — we swap axes so scene axis-Z (height in SSOT)
becomes Habitat axis-Y. See `_habitat_from_scene()` below.

CLI:
    python run_audio_pass_rlr.py \
        --spec ../../data/shoebox_v2_spec.json \
        --mesh ../../tmp/spike_rlr/shoebox_v2_mesh.glb \
        --materials ../../tmp/spike_rlr/shoebox_v2_materials.json \
        --out ../../tmp/spike_output/raw_audio/audio_B_rlr_FOA.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# --- import habitat-sim from ss2 env ---------------------------------------
# This module MUST be run with ss2 env's python; see run_all.sh.
import habitat_sim  # noqa: E402
from habitat_sim.sensor import RLRAudioPropagationChannelLayoutType  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


# --- coordinate transform ---------------------------------------------------
# SSOT coord frame: right-handed Y-up meters, +Y = mic forward = "window"
# Habitat coord frame: right-handed Y-up meters, but by convention many
# scene assets have Y as height. We keep it simple: for a shoebox v2 GLB
# authored with our SSOT convention where Z=height, we swap Y<->Z when
# feeding Habitat so height goes to Habitat's Y axis. (Test empirically
# after first RIR — if RIR looks wrong, revisit.)

def _habitat_from_scene(pos_scene):
    """Map SSOT (x, y_forward, z_up) -> Habitat (x, z_up, y_forward).

    Habitat treats +Y as up. Our SSOT is right-handed Y-up in the "gaming"
    sense where +Y = camera-forward (window direction) and +Z = height.
    Swapping y and z gives Habitat the "up axis is Y" convention while
    preserving right-handedness.
    """
    x, y, z = pos_scene
    return np.array([x, z, y], dtype=np.float32)


def _make_rlr_materials_json(materials_sidecar, out_path):
    """Convert our per-triangle material JSON to a SoundSpaces-style JSON
    that setAudioMaterialsJSON can read.

    Our sidecar format:
      {"materials": [{"name": tag, "alpha": [b1..b4], "scattering": s, "transmission": [t1..t4]}, ...]}

    Sound-Spaces format:
      {"materials": [{"name": ..., "absorption": [freq1, val1, freq2, val2, ...], "scattering": [...], "transmission": [...], "labels": [...]}, ...]}
    """
    # 4 bands center frequencies (SoundSpaces high-quality preset default)
    bands_hz = [125.0, 500.0, 2000.0, 8000.0]

    def interleave(vals):
        return [v for pair in zip(bands_hz, vals) for v in pair]

    out = {"materials": []}
    for m in materials_sidecar["materials"]:
        out["materials"].append({
            "name": m["name"],
            "absorption": interleave(m["alpha"]),
            "scattering": interleave([m["scattering"]] * 4),
            "transmission": interleave(m["transmission"]),
            "labels": [m["name"]],
        })

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def _load_scene_and_scene_two_dogs(spec_path=None):
    """Local import guard + spec-version dispatch.

    - spec_version == "v2" (shoebox_v2): compose_two_dog_scene_v2
    - spec_version == "apartment_v1":    compose_two_dog_scene_apartment

    If spec_path is None (legacy callers), assume shoebox v2.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    if spec_path is None:
        from scene_two_dogs_v2 import compose_two_dog_scene_v2
        return compose_two_dog_scene_v2
    with open(spec_path) as f:
        version = json.load(f).get("spec_version", "v2")
    if version == "apartment_v1":
        from scene_two_dogs_apartment import compose_two_dog_scene_apartment
        return compose_two_dog_scene_apartment
    # default: shoebox_v2 (spec_version=="v2")
    from scene_two_dogs_v2 import compose_two_dog_scene_v2
    return compose_two_dog_scene_v2


# Override map: for the spike we override some tags with hand-picked wavs
# from OmniAudio so listeners can tell golden vs husky apart.
# Special value "__pink_noise__" triggers synthesis of steady-state pink noise
# (used for husky so the perceived amplitude/spectrum change during
# occlusion is 100% due to RLR, not to the source content itself).
_TAG_AUDIO_OVERRIDES = {
    # Golden: real dog bark (restored per user request)
    "dog_golden": "/data/datasets/omniaudio/train-data-az-360-large/Barking Aldi Dog_358.wav",
    # Husky: synthesized C-major do-re-mi piano scale (1 note per ~0.6s).
    # Rich HF harmonics so HRTF ILD is strong; discrete notes make it easy
    # to hear the same note move around the head as husky detours.
    "dog_husky":  "__piano_scale__",
}


def _synth_pink_noise(sample_rate, duration_s, seed=0):
    """Synthesize pink (1/f) noise. Flat perceptual spectrum, fully stationary."""
    n = int(round(sample_rate * duration_s))
    rng = np.random.default_rng(seed)
    # Voss-McCartney approximation: sum of N halved-rate white noise generators
    n_octaves = 16
    y = np.zeros(n, dtype=np.float32)
    for k in range(n_octaves):
        step = 1 << k
        if step >= n:
            break
        white = rng.standard_normal(n // step + 1).astype(np.float32)
        # Repeat each sample `step` times, then crop to n
        expanded = np.repeat(white, step)[:n]
        y += expanded
    y = y / max(np.abs(y).max(), 1e-9)
    y = y * 0.5  # peak = 0.5 so headroom stays sane after conv
    return y.astype(np.float32)


def _synth_click_train(sample_rate, duration_s, click_hz=4.0, click_ms=10.0):
    """Repeating short pulse. Between clicks the RIR tail is audible in
    isolation, so any per-frame IR change (occlusion, moving source) shows
    up in the perceived reverb between clicks -- much clearer to hear than
    a continuous source where late-reverb accumulates and masks direct-sound
    variation.

    click_hz: pulses per second.
    click_ms: width of each pulse (Hanning-windowed).
    """
    n = int(round(sample_rate * duration_s))
    y = np.zeros(n, dtype=np.float32)
    click_len = max(int(round(sample_rate * click_ms / 1000)), 2)
    click = np.hanning(click_len).astype(np.float32) * 0.9
    period_samples = int(round(sample_rate / click_hz))
    for start in range(0, n, period_samples):
        end = min(start + click_len, n)
        y[start:end] += click[:end - start]
    return y


def _synth_piano_scale(sample_rate, duration_s, base_hz=261.63,
                        note_period_s=0.6, harmonics=8, amp=0.35,
                        sustain_s=1.8):
    """Synthesize a legato C-major piano scale (do-re-mi-fa-sol-la-si-do…).

    Each note's tail (~1.8s exponential decay, sustain-pedal-style) continues
    past the next attack, so successive notes overlap — no silent gaps.
    Sequence wraps back to lower octaves after C5 so we always fill the
    full clip regardless of duration.

    Signal characteristics:
      * Piano-like additive harmonics with mild inharmonicity
      * Fast (5ms) attack, slow (~2s) exponential decay -> legato/sustain
      * Rich content up to ~2-4 kHz (harmonics of 260-520 Hz fundamentals)
        -> strong HRTF ILD cues for spatial demo
    """
    # C major scale semitones from C4 (repeats up an octave for longer clips)
    scale = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24]
    n_total = int(round(sample_rate * duration_s))
    n_notes = int(np.ceil(duration_s / note_period_s))
    period_n = int(round(note_period_s * sample_rate))
    sustain_n = int(round(sustain_s * sample_rate))
    y = np.zeros(n_total + sustain_n, dtype=np.float32)  # extra room for tails

    for i in range(n_notes):
        step = scale[i % len(scale)]
        f0 = base_hz * (2.0 ** (step / 12.0))
        start = i * period_n
        t = np.arange(sustain_n, dtype=np.float32) / sample_rate
        note = np.zeros(sustain_n, dtype=np.float32)
        for h in range(1, harmonics + 1):
            f_h = f0 * h * (1.0 + 0.0004 * h * h)
            a_h = (1.0 / (h ** 1.2)) * np.exp(-0.05 * h)
            note += a_h * np.sin(2 * np.pi * f_h * t)
        # Slow exponential decay (sustain pedal style)
        env = np.exp(-1.8 * t / sustain_s)
        # 5ms attack ramp
        attack_n = int(0.005 * sample_rate)
        env[:attack_n] *= np.linspace(0, 1, attack_n)
        note *= env
        end = start + sustain_n
        y[start:end] += note.astype(np.float32)

    y = y[:n_total]
    y = y / (np.max(np.abs(y)) + 1e-9) * amp
    return y.astype(np.float32)


def _synth_hf_tone(sample_rate, duration_s, base_hz=2000.0, vibrato_hz=6.0,
                    vibrato_cents=40.0, amp=0.5):
    """Steady high-frequency tone (2kHz) with mild vibrato.

    Why high frequency for the spike:
      * Sofa fabric α_8kHz = 0.60 vs α_125Hz = 0.15 -> 4x stronger occlusion
        drop on high freqs (drop is fully spatial, not source-timevariant)
      * Human ITD/ILD azimuth resolution is best in the 1.5-5 kHz band,
        so L/R movement is easier to hear
      * Vibrato (~40 cent, 6 Hz) keeps the ear "hooked" on the tone without
        adding intrinsic amplitude/spectrum motion
    Purpose: any amplitude/timbre change you hear is caused by RLR
    spatial propagation, not the source waveform.
    """
    n = int(round(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Vibrato: cents -> ratio -> instantaneous frequency
    vib_ratio = np.exp(np.log(2) / 1200.0 * vibrato_cents *
                        np.sin(2 * np.pi * vibrato_hz * t))
    inst_hz = base_hz * vib_ratio
    # Integrate instantaneous frequency to get phase
    phase = 2 * np.pi * np.cumsum(inst_hz) / sample_rate
    y = amp * np.sin(phase).astype(np.float32)
    return y


def _load_dry_source(tag, sample_rate, duration_s, seed=42):
    """Load a dry source wav for a given tag.

    First checks _TAG_AUDIO_OVERRIDES (hand-picked spike sources), then
    falls back to audio_registry, then to a synthesized placeholder.
    """
    n_samples = int(round(sample_rate * duration_s))
    try:
        override = _TAG_AUDIO_OVERRIDES.get(tag)
        if override == "__pink_noise__":
            print(f"[audio] {tag}: using SYNTHETIC pink noise (steady-state)")
            return _synth_pink_noise(sample_rate, duration_s, seed=seed)
        if override == "__click_train__":
            print(f"[audio] {tag}: using SYNTHETIC click train (4Hz, 10ms pulses)")
            return _synth_click_train(sample_rate, duration_s,
                                       click_hz=4.0, click_ms=10.0)
        if override == "__hf_tone__":
            print(f"[audio] {tag}: using SYNTHETIC 2kHz vibrato tone (steady HF)")
            return _synth_hf_tone(sample_rate, duration_s,
                                   base_hz=2000.0, vibrato_hz=6.0,
                                   vibrato_cents=40.0, amp=0.5)
        if override == "__piano_scale__":
            print(f"[audio] {tag}: using SYNTHETIC piano C-major scale (do-re-mi)")
            return _synth_piano_scale(sample_rate, duration_s)
        if override and os.path.exists(override):
            wav_path = override
            print(f"[audio] {tag}: using SPIKE OVERRIDE {os.path.basename(wav_path)}")
        else:
            sys.path.insert(0, str(REPO_ROOT / "tools"))
            from gpurir_scenes.audio_registry import pick_audio
            picked = pick_audio(tag, np.random.default_rng(seed))
            # pick_audio returns (path, source, keyword) or just path (varies by
            # registry version). Support both.
            wav_path = picked[0] if isinstance(picked, tuple) else picked
            print(f"[audio] {tag}: using registry pick {os.path.basename(wav_path)}")
        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sample_rate:
            # simple linear resample (crude but good enough for spike)
            new_len = int(round(len(y) * sample_rate / sr))
            y = np.interp(np.linspace(0, len(y), new_len, endpoint=False),
                          np.arange(len(y)), y).astype(np.float32)
        # Loop / trim to n_samples
        if len(y) < n_samples:
            reps = int(np.ceil(n_samples / max(len(y), 1)))
            y = np.tile(y, reps)
        y = y[:n_samples]
        # Peak normalize to prevent clipping in convolution
        peak = np.abs(y).max()
        if peak > 1e-9:
            y = y * (0.8 / peak)
        return y
    except Exception as e:
        # Fallback: use a bark-like AM sinusoid (obvious placeholder)
        print(f"[audio] WARNING: audio_registry failed for {tag}: {e}. Using placeholder tone.")
        t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
        # Different F0 per tag so B/C tests can distinguish; envelope repeats
        base_hz = 300.0 if "husky" in tag else 800.0
        env = np.abs(np.sin(2 * np.pi * 3 * t))  # 3 barks/sec
        y = 0.5 * env * np.sin(2 * np.pi * base_hz * t).astype(np.float32)
        return y


def build_rlr_sim(glb_path, materials_json_path, sample_rate=16000,
                  channel_layout="ambisonics", indirect_ray_count=500):
    """Create a Habitat Simulator with a single AudioSensor.

    No visual sensors, no physics -- audio only. The scene mesh comes from
    our SSOT-generated shoebox_v2_mesh.glb.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb_path)
    sim_cfg.enable_physics = False
    # Audio-only: skip the OpenGL renderer entirely (avoids the "cannot
    # retrieve OpenGL version" error when running headless without X). RLR
    # doesn't need a GL context — it operates on the loaded mesh directly.
    sim_cfg.create_renderer = False
    sim_cfg.load_semantic_mesh = False
    sim_cfg.requires_textures = False

    # Minimal agent (required by habitat, but we don't use it for locomotion)
    agent_cfg = habitat_sim.AgentConfiguration()

    audio_spec = habitat_sim.AudioSensorSpec()
    # Habitat's Simulator.get_sensor_observations hardcodes the uuid
    # 'audio_sensor' when dispatching to _get_audio_observation(); if we
    # rename we lose the routing. Stick to what the sim expects.
    audio_spec.uuid = "audio_sensor"
    audio_spec.enableMaterials = False  # we'll set via JSON to override GLB PBR
    if channel_layout == "ambisonics":
        audio_spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
        audio_spec.channelLayout.channelCount = 4  # 1st-order FOA
    elif channel_layout == "binaural":
        audio_spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Binaural
        audio_spec.channelLayout.channelCount = 2
    else:
        raise ValueError(f"unknown channel layout {channel_layout}")

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
    audio_spec.acousticsConfig.unitScale = 1.0  # meters

    # sensor mounted on the agent
    agent_cfg.sensor_specifications = [audio_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Load material overrides
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON(str(materials_json_path))

    return sim, audio_sensor


def _set_agent_pose(sim, mic_pos_scene, mic_forward=(0.0, 1.0, 0.0)):
    """Place the Habitat agent (mic/listener) at mic_pos and face +Y_scene.

    Habitat convention: agent's "forward" is -Z_habitat (like a camera looking
    down its -Z axis). Our SSOT: +Y_scene = mic_forward = window direction.
    After the Y<->Z axis swap in _habitat_from_scene, +Y_scene becomes
    +Z_habitat. So we need to yaw the agent 180 deg about Y (up) so its
    forward (-Z) becomes +Z, i.e. it now faces +Y_scene.

    This orientation matters because RLR's ambisonic output is expressed in
    the LISTENER's local frame (not world frame). Wrong orientation was why
    all sources ended up on one side in the stereo downmix.
    """
    import quaternion  # noqa: E402  (bundled with habitat-sim)
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = _habitat_from_scene(mic_pos_scene)
    # 180-degree rotation about the Y_habitat (up) axis.
    # Quaternion: w=0, x=0, y=1, z=0
    state.rotation = np.quaternion(0.0, 0.0, 1.0, 0.0)
    agent.set_state(state)


def compute_rir_and_render(spec_path, glb_path, materials_sidecar_path,
                           out_wav_path, downmix_stereo_path=None,
                           quality_mode="high", verbose=True):
    """End-to-end: build sim, iterate trajectory, convolve, mix, write."""
    with open(spec_path) as f:
        spec = json.load(f)
    with open(materials_sidecar_path) as f:
        materials_sidecar = json.load(f)

    sample_rate = spec["audio_config"]["sample_rate_hz"]
    duration_s = spec["audio_config"]["duration_s"]
    n_frames = spec["render_config"]["n_frames"]
    fps = spec["render_config"]["fps"]
    n_samples_total = int(round(sample_rate * duration_s))
    samples_per_frame = int(round(sample_rate / fps))

    # Build the SoundSpaces-formatted materials json in a sibling temp file
    rlr_materials_json = Path(materials_sidecar_path).with_name(
        Path(materials_sidecar_path).stem + "_rlr.json"
    )
    _make_rlr_materials_json(materials_sidecar, rlr_materials_json)
    if verbose:
        print(f"[rlr] materials JSON -> {rlr_materials_json}")

    # Quality mode -> ray count (must be set on spec, not on live sensor)
    ray_count_by_quality = {"low": 100, "high": 500, "max": 5000}
    indirect_ray_count = ray_count_by_quality.get(quality_mode, 500)

    sim, audio_sensor = build_rlr_sim(
        glb_path, rlr_materials_json,
        sample_rate=sample_rate,
        indirect_ray_count=indirect_ray_count,
    )

    # Compose scene to get trajectories (dispatch by spec_version)
    load_scene = _load_scene_and_scene_two_dogs(spec_path)
    scene = load_scene(spec_path)

    # Place mic (Habitat's agent = the listener)
    _set_agent_pose(sim, spec["mic"]["pos_m"])

    # Wet output buffer (n_channels x n_samples_total). We know from
    # build_rlr_sim we set 4-ch ambisonic; assert to be defensive.
    n_channels = 4  # 1st-order FOA
    wet = np.zeros((n_channels, n_samples_total), dtype=np.float32)
    per_source_wet_map = {}  # tag -> (n_channels, n_samples) FOA buffer

    # For each source: precompute dry, then per-frame get IR + convolve
    for a in scene.animals:
        tag = a.tag
        traj_scene = a.trajectory_m  # (n_frames, 3)
        dry = _load_dry_source(tag, sample_rate, duration_s)
        if verbose:
            print(f"[rlr] source {tag}: dry rms = {np.sqrt(np.mean(dry**2)):.4f}, "
                  f"peak = {np.max(np.abs(dry)):.4f}")

        per_source_wet = np.zeros_like(wet)
        t0 = time.time()

        for f in range(n_frames):
            src_pos = traj_scene[f]
            src_hab = _habitat_from_scene(src_pos)
            audio_sensor.setAudioSourceTransform(src_hab)

            obs = sim.get_sensor_observations()
            ir_raw = obs["audio_sensor"]  # list-of-arrays (per channel) or ndarray
            # Normalize to (n_channels, n_samples_ir) ndarray
            if isinstance(ir_raw, list):
                ir = np.asarray(ir_raw, dtype=np.float32)
            else:
                ir = np.asarray(ir_raw, dtype=np.float32)
            if ir.ndim == 2 and ir.shape[0] > ir.shape[1]:
                ir = ir.T  # ensure (n_channels, n_samples_ir)
            elif ir.ndim == 1:
                ir = ir[None, :]  # (1, n_samples)

            # convolve the audio around this frame with the IR, add into per_source_wet
            frame_start = f * samples_per_frame
            frame_end = min(frame_start + samples_per_frame, n_samples_total)
            dry_chunk = dry[frame_start:frame_end]
            if len(dry_chunk) == 0:
                continue
            # naive conv per channel; result length = len(dry_chunk) + ir.shape[1] - 1
            for c in range(n_channels):
                wet_chunk = np.convolve(dry_chunk, ir[c], mode="full")
                w_end = min(frame_start + len(wet_chunk), n_samples_total)
                per_source_wet[c, frame_start:w_end] += wet_chunk[:w_end - frame_start]

            if verbose and f % 15 == 0:
                print(f"[rlr] {tag}: frame {f}/{n_frames} "
                      f"src={src_pos} ir_shape={ir.shape} "
                      f"elapsed={time.time() - t0:.1f}s")

        wet += per_source_wet
        per_source_wet_map[tag] = per_source_wet.copy()

    # Write per-source (solo) FOA + stereo BEFORE the shared peak-norm so
    # each solo track has its own natural loudness. Using a common global
    # normalizer would drown out husky in the mixed track once occluded.
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_stereo(foa_buf, path):
        """FOA(4, N) -> stereo(N, 2) via ch3 L/R decode. See below for details."""
        W_ch = foa_buf[0]
        lr_axis = foa_buf[3]
        L = W_ch - 0.707 * lr_axis
        R = W_ch + 0.707 * lr_axis
        st = np.stack([L, R], axis=1)
        p = np.abs(st).max()
        if p > 1e-9:
            st = st * (0.9 / p)
        sf.write(str(path), st, sample_rate, subtype="PCM_16")

    for tag, buf in per_source_wet_map.items():
        solo_peak = np.abs(buf).max()
        buf_norm = buf * (0.9 / solo_peak) if solo_peak > 1e-9 else buf
        solo_foa = out_wav_path.parent / f"{out_wav_path.stem}_{tag}_FOA.wav"
        solo_stereo = out_wav_path.parent / f"{out_wav_path.stem}_{tag}_stereo.wav"
        sf.write(str(solo_foa), buf_norm.T, sample_rate, subtype="PCM_16")
        _write_stereo(buf_norm, solo_stereo)
        print(f"[rlr] wrote SOLO {solo_foa.name} + {solo_stereo.name}")

    # Peak normalize the mixed buffer
    peak = np.abs(wet).max()
    if peak > 1e-9:
        wet = wet * (0.9 / peak)

    # Write 4-channel FOA wav (mixed)
    sf.write(str(out_wav_path), wet.T, sample_rate, subtype="PCM_16")
    print(f"[rlr] wrote {out_wav_path}  shape={wet.shape}  sr={sample_rate}")

    # FOA -> stereo downmix.
    # RLR outputs 1st-order ambisonics but empirically the channel that
    # correlates with source azimuth (left/right of listener) is channel 3
    # for our shoebox scene, not channel 1 as ACN would predict. This is
    # likely FuMa-style ordering [W, X, Y, Z] where the sensor's local
    # X-axis maps to left/right when the agent is yawed 180 about world Y.
    #
    # Sign was also inverted vs my initial guess: agent right ↔ world -X,
    # so wet[3] > 0 when source is at world +X (mic's L side). We flip so
    # world +X hits R channel and world -X hits L (matches "画面右 = 世界
    # +X" convention documented in pipeline_zh.md appendix A).
    if downmix_stereo_path and n_channels == 4:
        W = wet[0]
        # LR axis (empirically ch3; see comment above)
        lr_axis = wet[3]
        # Positive lr_axis correlates with world +X (right in the SPEAR
        # picture-right = world+X = right ear convention).
        L = W - 0.707 * lr_axis   # world -X (left) louder in L
        R = W + 0.707 * lr_axis   # world +X (right) louder in R
        stereo = np.stack([L, R], axis=1)
        peak = np.abs(stereo).max()
        if peak > 1e-9:
            stereo = stereo * (0.9 / peak)
        sf.write(str(downmix_stereo_path), stereo, sample_rate, subtype="PCM_16")
        print(f"[rlr] wrote stereo downmix {downmix_stereo_path}")

    return {"wall_time_s": time.time() - t0}


def compute_binaural(spec_path, glb_path, materials_sidecar_path,
                      out_wav_path, quality_mode="high", verbose=True):
    """RLR native binaural (2ch) rendering. Uses the AudioSensor's built-in
    HRTF decoder. Same trajectory + per-source overlap-add as FOA path.

    Empirical listener-frame calibration: the agent is yawed 180° about Y
    in `_set_agent_pose()`, which makes RLR's binaural L and R channels
    come out swapped relative to the SSOT "world +X = camera right" convention.
    We swap L↔R at the end to restore convention.
    """
    with open(spec_path) as f:
        spec = json.load(f)
    with open(materials_sidecar_path) as f:
        materials_sidecar = json.load(f)

    sample_rate = spec["audio_config"]["sample_rate_hz"]
    duration_s = spec["audio_config"]["duration_s"]
    n_frames = spec["render_config"]["n_frames"]
    fps = spec["render_config"]["fps"]
    n_samples_total = int(round(sample_rate * duration_s))
    samples_per_frame = int(round(sample_rate / fps))

    rlr_materials_json = Path(materials_sidecar_path).with_name(
        Path(materials_sidecar_path).stem + "_rlr.json")
    _make_rlr_materials_json(materials_sidecar, rlr_materials_json)

    ray_count_by_quality = {"low": 100, "high": 500, "max": 5000}
    indirect_ray_count = ray_count_by_quality.get(quality_mode, 500)

    sim, audio_sensor = build_rlr_sim(
        glb_path, rlr_materials_json,
        sample_rate=sample_rate,
        channel_layout="binaural",
        indirect_ray_count=indirect_ray_count,
    )

    load_scene = _load_scene_and_scene_two_dogs(spec_path)
    scene = load_scene(spec_path)
    _set_agent_pose(sim, spec["mic"]["pos_m"])

    wet = np.zeros((2, n_samples_total), dtype=np.float32)
    per_source_wet_map = {}

    for a in scene.animals:
        tag = a.tag
        traj_scene = a.trajectory_m
        dry = _load_dry_source(tag, sample_rate, duration_s)
        if verbose:
            print(f"[rlr-bin] source {tag}: dry rms={np.sqrt(np.mean(dry**2)):.4f}")
        per_source_wet = np.zeros_like(wet)
        t0 = time.time()

        for f in range(n_frames):
            src_pos = traj_scene[f]
            src_hab = _habitat_from_scene(src_pos)
            audio_sensor.setAudioSourceTransform(src_hab)
            obs = sim.get_sensor_observations()
            ir_raw = obs["audio_sensor"]
            ir = np.asarray(ir_raw, dtype=np.float32)
            if ir.ndim == 2 and ir.shape[0] > ir.shape[1]:
                ir = ir.T
            elif ir.ndim == 1:
                ir = ir[None, :]

            frame_start = f * samples_per_frame
            frame_end = min(frame_start + samples_per_frame, n_samples_total)
            dry_chunk = dry[frame_start:frame_end]
            if len(dry_chunk) == 0:
                continue
            for c in range(2):
                wet_chunk = np.convolve(dry_chunk, ir[c], mode="full")
                w_end = min(frame_start + len(wet_chunk), n_samples_total)
                per_source_wet[c, frame_start:w_end] += wet_chunk[:w_end - frame_start]

            if verbose and f % 25 == 0:
                print(f"[rlr-bin] {tag}: frame {f}/{n_frames} elapsed={time.time()-t0:.1f}s")

        wet += per_source_wet
        per_source_wet_map[tag] = per_source_wet.copy()

    # Swap L↔R to match SSOT convention (see docstring)
    wet = wet[[1, 0], :]

    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    for tag, buf in per_source_wet_map.items():
        buf_sw = buf[[1, 0], :]
        p = np.abs(buf_sw).max()
        if p > 1e-9:
            buf_sw = buf_sw * (0.9 / p)
        solo_path = out_wav_path.parent / f"{out_wav_path.stem}_{tag}_binaural.wav"
        sf.write(str(solo_path), buf_sw.T, sample_rate, subtype="PCM_16")
        print(f"[rlr-bin] wrote SOLO {solo_path.name}")

    peak = np.abs(wet).max()
    if peak > 1e-9:
        wet = wet * (0.9 / peak)
    sf.write(str(out_wav_path), wet.T, sample_rate, subtype="PCM_16")
    print(f"[rlr-bin] wrote {out_wav_path} shape={wet.shape} sr={sample_rate}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "data" / "shoebox_v2_spec.json"))
    ap.add_argument("--mesh", default=str(REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2_mesh.glb"))
    ap.add_argument("--materials", default=str(REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2_materials.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" / "spike_output" / "raw_audio" / "audio_B_rlr_FOA.wav"))
    ap.add_argument("--stereo-out", default=str(REPO_ROOT / "tmp" / "spike_output" / "raw_audio" / "audio_B_rlr_stereo.wav"))
    ap.add_argument("--quality", default="high", choices=["low", "high", "max"])
    ap.add_argument("--channel-layout", default="ambisonics",
                    choices=["ambisonics", "binaural"],
                    help="ambisonics=4ch FOA output (default); binaural=2ch native binaural output")
    args = ap.parse_args()

    t_start = time.time()

    # Level-1/Level-2 profiling: wrap the RLR audio pass in a StageTimer.
    # clip_id is derived from spec basename (e.g. apartment_v1_000). CSV
    # goes to <out-dir>/profile_per_clip.csv where out-dir is inferred from
    # the --out argument's parent-parent (…/tmp/spike_output_apartment/raw_audio_hq/foo.wav
    # -> …/tmp/spike_output_apartment/).
    spec_stem = Path(args.spec).stem  # e.g. "apartment_v1_spec"
    clip_id = spec_stem.replace("_spec", "") + "_000"
    csv_path = Path(args.out).resolve().parent.parent / "profile_per_clip.csv"
    stage_name = "rlr_audio_binaural" if args.channel_layout == "binaural" else "rlr_audio_foa"

    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    from profiling import StageTimer

    with StageTimer(stage_name, clip_id=clip_id, csv_path=csv_path):
        if args.channel_layout == "binaural":
            # Reuse the same --out path (caller controls filename) but call the
            # binaural code path. --stereo-out is unused in this mode.
            compute_binaural(
                spec_path=args.spec,
                glb_path=args.mesh,
                materials_sidecar_path=args.materials,
                out_wav_path=Path(args.out),
                quality_mode=args.quality,
            )
        else:
            compute_rir_and_render(
                spec_path=args.spec,
                glb_path=args.mesh,
                materials_sidecar_path=args.materials,
                out_wav_path=args.out,
                downmix_stereo_path=args.stereo_out,
                quality_mode=args.quality,
            )
    print(f"[rlr] TOTAL wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
