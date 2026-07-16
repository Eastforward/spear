"""FOA (4ch Ambisonic) -> HRTF binaural (2ch) decoder.

Uses spaudiopy's magLS decoder + MIT KEMAR HRTFs. Also computes IPD/ILD
features from the resulting binaural signal for illustration.

Channel convention: RLR-Audio-Propagation writes 1st-order Ambisonics in
ACN/SN3D-ish ordering (W=0, Y=1, Z=2, X=3). spaudiopy expects ACN/N3D:
same ordering, different normalisation. For a 1st-order SH set, N3D vs
SN3D differs only by a global scaling per l-degree (l=0: same; l=1: multiply
Y/Z/X by sqrt(3)). Since we output normalised binaural at the end, that
scaling is absorbed and either convention yields perceptually equivalent
listening results. We apply the SN3D->N3D scaling explicitly to keep
IPD/ILD physically meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from spaudiopy import decoder, io


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HRTF = Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa")


def _rlr_to_acn_n3d(sig_rlr: np.ndarray) -> np.ndarray:
    """Convert RLR's 4-ch FOA output to ACN/N3D ordering (what spaudiopy wants).

    Empirical mapping (see comment block in run_audio_pass_rlr.py:468-486):
      RLR ch0 -> ACN Y (horizontal, sin(azi)*cos(elev))   -> ACN index 1
      RLR ch1 -> ACN W (omni)                             -> ACN index 0  (WRONG?)
      RLR ch2 -> ACN Z (vertical, sin(elev))              -> ACN index 2
      RLR ch3 -> ACN X (horizontal, cos(azi)*cos(elev))   -> ACN index 3
    ...actually the diagnostic showed RLR ch3 is the L/R axis with positive-
    sign convention matching world +X = listener right. The listener is
    yawed 180 deg about Y (agent looks at +Y_scene), so agent-frame X and
    world-frame X have opposite signs; ACN wants "agent-frame Y = left+".

    Diagnostic result on golden-only clip:
      EARLY (source at LEFT of listener):   ch1 energy=52 (biggest non-W)
      LATE  (source at RIGHT of listener):  ch3 energy=0.44 (biggest non-W)
      → ch1 encodes source position along ONE axis; ch3 along another.
      → since golden trajectory moves along scene-X (across camera view),
        ch1 must be the front/back axis (source moves from front-left to
        front-right stays in ~front), ch3 is left/right.

    So the working guess is RLR = [W_or_X, ?, ?, Y-ish]. Simpler experiment:
    treat RLR as FuMa (W, X, Y, Z) with W=ch0, and permute after
    that until L/R lines up. FuMa→ACN mapping:
       ACN[0]=W=FuMa[0],  ACN[1]=Y=FuMa[2],  ACN[2]=Z=FuMa[3],  ACN[3]=X=FuMa[1]
    And FuMa uses sqrt(2) W scaling that N3D doesn't.

    We test this empirically by rendering + listening.
    """
    W = sig_rlr[0]
    X_fuma = sig_rlr[1]
    Y_fuma = sig_rlr[2]
    Z_fuma = sig_rlr[3]

    # FuMa -> ACN reordering + normalisation to N3D
    #   FuMa W has 1/sqrt(2) scale relative to N3D W
    W_n3d = W * np.sqrt(2.0)
    Y_n3d = Y_fuma * np.sqrt(3.0)  # ACN index 1
    Z_n3d = Z_fuma * np.sqrt(3.0)  # ACN index 2
    X_n3d = X_fuma * np.sqrt(3.0)  # ACN index 3

    return np.stack([W_n3d, Y_n3d, Z_n3d, X_n3d], axis=0)


def _sn3d_to_n3d_foa(sig_wxyz_sn3d: np.ndarray) -> np.ndarray:
    """Legacy: assume input already in ACN/SN3D order. Kept for A/B testing."""
    out = sig_wxyz_sn3d.copy()
    out[1:4] *= np.sqrt(3.0)
    return out


def load_foa(path: Path) -> tuple[np.ndarray, int]:
    """Read a 4-ch FOA WAV. Returns (foa[4, N], sr)."""
    x, sr = sf.read(str(path), always_2d=True)
    assert x.shape[1] == 4, f"expected 4-ch FOA, got shape {x.shape}"
    return x.T.astype(np.float32), sr  # (4, N)


def foa_to_binaural(foa: np.ndarray, foa_sr: int, hrtf_path: Path = DEFAULT_HRTF,
                    normalize: bool = True) -> tuple[np.ndarray, int]:
    """Decode FOA -> binaural (2, N) via magLS + KEMAR HRTF.

    KEMAR SOFA is 44.1kHz; we resample FOA to 44.1kHz for decoding, then
    return at 44.1kHz (better for listening). Downstream can resample.
    """
    hrirs = io.load_sofa_hrirs(str(hrtf_path))
    kemar_sr = int(hrirs.fs)  # 44100

    if foa_sr != kemar_sr:
        n_out = int(round(foa.shape[1] * kemar_sr / foa_sr))
        foa_rs = signal.resample_poly(foa, kemar_sr, foa_sr, axis=1).astype(np.float32)
        # resample_poly changes length slightly vs exact ratio; renormalize:
        foa_rs = signal.resample_poly(foa, up=kemar_sr, down=foa_sr, axis=1).astype(np.float32)
        _ = n_out
    else:
        foa_rs = foa

    # RLR uses FuMa-like ordering; remap to ACN/N3D that spaudiopy expects
    foa_n3d = _rlr_to_acn_n3d(foa_rs)

    # magLS decoder for 1st-order (N_sph=1 -> 4 SH channels)
    hrirs_nm = decoder.magls_bin(hrirs, N_sph=1)   # shape (2, 4, L)
    binaural = decoder.sh2bin(foa_n3d, hrirs_nm)    # (2, N + L - 1)

    if normalize:
        peak = np.max(np.abs(binaural))
        if peak > 0:
            binaural = binaural / peak * 0.95

    return binaural.astype(np.float32), kemar_sr


def compute_ipd_ild(binaural: np.ndarray, sr: int, n_fft: int = 512,
                    hop: int = 160) -> dict:
    """Compute IPD/ILD from binaural stereo. Returns dict with tensors + summary."""
    L, R = binaural[0], binaural[1]

    STFT_L = signal.stft(L, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          return_onesided=True)[2]
    STFT_R = signal.stft(R, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          return_onesided=True)[2]

    # IPD: phase difference L vs R, wrapped to [-pi, pi]
    ipd = np.angle(STFT_L * np.conj(STFT_R))  # (F, T)

    # ILD: log-magnitude ratio in dB (clip to avoid log(0))
    eps = 1e-10
    ild = 20.0 * np.log10((np.abs(STFT_L) + eps) / (np.abs(STFT_R) + eps))  # (F, T)

    freqs = np.linspace(0, sr / 2, ipd.shape[0])
    times = np.arange(ipd.shape[1]) * hop / sr

    # Rough summaries: energy-weighted mean ILD per time frame, and
    # ITD estimate via broadband cross-correlation of L,R
    energy_lr = np.abs(STFT_L) + np.abs(STFT_R)
    weighted_ild_t = np.sum(ild * energy_lr, axis=0) / (np.sum(energy_lr, axis=0) + eps)

    # broadband ITD via GCC-PHAT on 100ms windows
    win = int(0.1 * sr)
    itd_us = []
    for start in range(0, len(L) - win, win):
        l_seg, r_seg = L[start:start+win], R[start:start+win]
        # PHAT
        X = np.fft.rfft(l_seg)
        Y = np.fft.rfft(r_seg)
        cc = X * np.conj(Y)
        cc /= (np.abs(cc) + eps)
        c = np.fft.irfft(cc, n=2*win)
        lag = np.argmax(np.abs(np.fft.fftshift(c))) - win
        itd_us.append(lag / sr * 1e6)
    itd_us = np.asarray(itd_us)

    return {
        "ipd": ipd, "ild": ild,
        "freqs": freqs, "times": times,
        "weighted_ild_t": weighted_ild_t,
        "itd_us_100ms": itd_us,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foa", required=True, help="input 4-ch FOA WAV")
    ap.add_argument("--out", required=True, help="output binaural stereo WAV")
    ap.add_argument("--hrtf", default=str(DEFAULT_HRTF))
    ap.add_argument("--target-sr", type=int, default=0,
                    help="resample output to this sr (0 = keep 44100)")
    args = ap.parse_args()

    foa, foa_sr = load_foa(Path(args.foa))
    print(f"[foa2bin] loaded FOA shape={foa.shape} sr={foa_sr}")

    bin_sig, out_sr = foa_to_binaural(foa, foa_sr, Path(args.hrtf))
    print(f"[foa2bin] binaural shape={bin_sig.shape} sr={out_sr}")

    if args.target_sr and args.target_sr != out_sr:
        n_out = int(round(bin_sig.shape[1] * args.target_sr / out_sr))
        bin_sig = signal.resample_poly(bin_sig, args.target_sr, out_sr, axis=1).astype(np.float32)
        out_sr = args.target_sr

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, bin_sig.T, out_sr)
    print(f"[foa2bin] wrote {args.out}")

    # Also dump quick IPD/ILD summary next to it
    feats = compute_ipd_ild(bin_sig, out_sr)
    summary = {
        "input_foa": args.foa,
        "output_bin": args.out,
        "sr": int(out_sr),
        "n_samples": int(bin_sig.shape[1]),
        "ipd_shape": list(feats["ipd"].shape),
        "ild_shape": list(feats["ild"].shape),
        "ild_broadband_range_db": [
            float(feats["weighted_ild_t"].min()),
            float(feats["weighted_ild_t"].max()),
        ],
        "itd_us_100ms_range": [
            float(feats["itd_us_100ms"].min()),
            float(feats["itd_us_100ms"].max()),
        ],
    }
    summary_path = Path(args.out).with_suffix(".ipd_ild.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[foa2bin] wrote {summary_path}")


if __name__ == "__main__":
    main()
