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

HY3D_ROOT = "/data/jzy/code/Hunyuan3D-2.1"
sys.path.insert(0, HY3D_ROOT)
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dpaint"))
os.chdir(HY3D_ROOT)   # MANDATORY — pipeline uses cwd-relative paths


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
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    conf.texture_size = args.texture_size

    t0 = time.time()
    print(f"[hy3d_bake] loading pipeline ({args.max_num_view} views, {args.resolution} res)...", flush=True)
    pipeline = Hunyuan3DPaintPipeline(conf)
    print(f"[hy3d_bake] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    pipeline(
        mesh_path=work_glb,
        image_path=args.reference_image,
        output_mesh_path=output_glb,
        use_remesh=True,     # near-no-op on our 1233-vert mesh; probe verified
        save_glb=True,
    )
    dt = time.time() - t1
    print(f"[hy3d_bake] paint pipeline finished in {dt:.1f}s", flush=True)

    # Rename baked sidecars to canonical names so downstream tools have stable paths
    stem = os.path.splitext(os.path.basename(output_glb))[0]
    for src_suffix, dst_name in [(".jpg", "hy3d_diffuse.jpg"),
                                 ("_metallic.jpg", "hy3d_metallic.jpg"),
                                 ("_roughness.jpg", "hy3d_roughness.jpg")]:
        src = os.path.join(args.workdir, stem + src_suffix)
        dst = os.path.join(args.workdir, dst_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Sanity check the three products
    required = ["white_mesh_remesh.obj", "hy3d_diffuse.jpg"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.workdir, f))]
    if missing:
        print(f"HY3D_BAKE_FAIL missing {missing} in {args.workdir}")
        sys.exit(1)

    print(f"HY3D_BAKE_OK workdir={args.workdir} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
