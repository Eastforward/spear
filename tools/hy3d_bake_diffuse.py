"""Stage 1 of the animated-dog-hunyuan-paint pipeline.

Thin wrapper around Hunyuan3DPaintPipeline that fixes config, takes
(input glb, reference image, workdir), and drops the three baked PBR
maps + the remeshed obj into workdir. Prints HY3D_BAKE_OK on success.

Runs inside the hunyuan3d conda env — do NOT try to run it via any
other Python. Environment variables HY3DGEN_MODELS + LD_LIBRARY_PATH +
CUDA_VISIBLE_DEVICES must be set; verify_hy3d_env.py in the same shell
first is the recommended pattern.
"""
import argparse
import os
import shutil
import sys
import time

SPEAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY3D_ROOT = os.environ.get(
    "HY3D_ROOT",
    os.path.join(os.path.dirname(SPEAR_ROOT), "Hunyuan3D-2.1"),
)
HY3D_CUSTOM_RASTERIZER_ROOT = os.path.join(
    HY3D_ROOT, "hy3dpaint", "custom_rasterizer"
)
sys.path.insert(0, HY3D_CUSTOM_RASTERIZER_ROOT)
sys.path.insert(0, HY3D_ROOT)
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dpaint"))
os.chdir(HY3D_ROOT)   # MANDATORY — pipeline uses cwd-relative paths


def _resolve_realesrgan_ckpt_path():
    """Return the RealESRGAN checkpoint path for this monorepo checkout."""
    candidates = [
        os.path.join(HY3D_ROOT, "ckpt", "RealESRGAN_x4plus.pth"),
        os.path.join(HY3D_ROOT, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "RealESRGAN_x4plus.pth not found. Expected one of: "
        + ", ".join(candidates)
    )


def _resolve_multiview_pretrained_path():
    """Return the Hunyuan paint repo id/path token for local-first loading."""
    hy3d_models = os.environ.get("HY3DGEN_MODELS", "")
    local_repo = "hunyuan3d-2.1"
    if hy3d_models:
        local_paint = os.path.join(
            os.path.expanduser(hy3d_models),
            local_repo,
            "hunyuan3d-paintpbr-v2-1",
        )
        if os.path.exists(local_paint):
            return local_repo
    return "tencent/Hunyuan3D-2.1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glb", required=True,
                   help="Absolute path to Dog_textured.glb (or a subdivided variant).")
    p.add_argument("--reference-image", required=True,
                   help="Absolute path to reference PNG/JPG (should be rembg'd + standing side-view for best result).")
    p.add_argument("--workdir", required=True,
                   help="Absolute path to a directory that will contain all Hunyuan intermediates.")
    p.add_argument("--max-num-view", type=int, default=6, choices=[6, 7, 8, 9],
                   help="Number of Hunyuan multiview cameras. Default 6; bump to 9 if bake coverage insufficient (F4).")
    p.add_argument("--resolution", type=int, default=512,
                   help="Hunyuan per-view diffusion resolution. Default 512.")
    p.add_argument("--texture-size", type=int, default=4096,
                   help="Output baked texture size in pixels. Default 4096.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    work_glb = os.path.join(args.workdir, "input.glb")
    shutil.copy2(args.input_glb, work_glb)
    output_glb = os.path.join(args.workdir, "hy3d_output_mesh.glb")

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    conf = Hunyuan3DPaintConfig(max_num_view=args.max_num_view, resolution=args.resolution)
    conf.realesrgan_ckpt_path = _resolve_realesrgan_ckpt_path()
    conf.multiview_pretrained_path = _resolve_multiview_pretrained_path()
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    conf.texture_size = args.texture_size

    t0 = time.time()
    print(f"[hy3d_bake] loading pipeline ({args.max_num_view} views, {args.resolution} res)...", flush=True)
    pipeline = Hunyuan3DPaintPipeline(conf)
    print(f"[hy3d_bake] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    # IMPORTANT: pass output as .obj, not .glb. Hunyuan writes the OBJ with
    # proper UV coordinates via save_obj_mesh(); then it ALSO writes a .glb
    # (via trimesh) which usually comes out broken with "incorrect header"
    # and no UVs. Downstream tools need the OBJ.
    output_obj = os.path.splitext(output_glb)[0] + ".obj"
    t1 = time.time()
    pipeline(
        mesh_path=work_glb,
        image_path=args.reference_image,
        output_mesh_path=output_obj,     # <-- .obj (writes UVs)
        use_remesh=True,
        save_glb=True,                    # still write the (usually broken) .glb sidecar
    )
    dt = time.time() - t1
    print(f"[hy3d_bake] paint pipeline finished in {dt:.1f}s", flush=True)

    # Rename baked sidecars to canonical names so downstream tools have stable paths
    stem = os.path.splitext(os.path.basename(output_obj))[0]
    for src_suffix, dst_name in [(".jpg", "hy3d_diffuse.jpg"),
                                 ("_metallic.jpg", "hy3d_metallic.jpg"),
                                 ("_roughness.jpg", "hy3d_roughness.jpg")]:
        src = os.path.join(args.workdir, stem + src_suffix)
        dst = os.path.join(args.workdir, dst_name)
        if os.path.exists(src):
            shutil.move(src, dst)
    # Canonical name for the textured obj (has UVs — downstream weight-transfer needs it)
    textured_obj = os.path.join(args.workdir, "hy3d_textured.obj")
    if os.path.exists(output_obj):
        shutil.move(output_obj, textured_obj)

    # Sanity check
    required = ["white_mesh_remesh.obj", "hy3d_textured.obj", "hy3d_diffuse.jpg"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.workdir, f))]
    if missing:
        print(f"HY3D_BAKE_FAIL missing {missing} in {args.workdir}")
        sys.exit(1)

    # Assert hy3d_textured.obj actually has UVs (this is the whole point of the
    # tool ordering fix — we discovered 2026-07-05 that if downstream tools use
    # white_mesh_remesh.obj, they get zero UVs and produce untextured meshes)
    with open(textured_obj) as f:
        content = f.read()
    if "\nvt " not in content:
        print(f"HY3D_BAKE_FAIL {textured_obj} contains no vt (UV) lines")
        sys.exit(1)

    print(f"HY3D_BAKE_OK workdir={args.workdir} elapsed={dt:.1f}s "
          f"textured_obj={textured_obj}")


if __name__ == "__main__":
    main()
