"""One-shot probe: feed Dog_textured.glb + a border-collie reference image
into Hunyuan3D-Paint and observe what comes out. NOT the production
pipeline — this is the 30-60 min investigation asked for by the spec-C
choice of the animated-dog-hunyuan-paint spec.

Outputs everything into /tmp/probe_hy3dpaint/ so we can inspect:
  - white_mesh_remesh.obj  <- Hunyuan's remeshed copy (if use_remesh=True)
  - textured_mesh.obj/.glb <- Hunyuan's output
  - textured_mesh.jpg      <- the baked albedo (2K or 4K)
  - textured_mesh_metallic.jpg
  - textured_mesh_roughness.jpg

Two runs:
  RUN 1: use_remesh=True (default) — see what happens if we let it remesh
  RUN 2: use_remesh=False           — see if pipeline works without remesh

Then we compare vertex counts, UV layouts, and whether the animation
data would survive.
"""
import os
import shutil
import subprocess
import sys
import time

HY3D = "/data/jzy/code/Hunyuan3D-2.1"
sys.path.insert(0, HY3D)
sys.path.insert(0, os.path.join(HY3D, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D, "hy3dpaint"))
os.chdir(HY3D)  # required — pipeline uses relative paths for cfgs and ckpts

INPUT_GLB = "/data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb"
REFERENCE_IMG = "/data/jzy/code/Hunyuan3D-2.1/demos/collie_clean.png"
OUT_ROOT = "/tmp/probe_hy3dpaint"


def dump_glb_info(label, glb_path):
    """Print vertex/anim summary of a glb using pygltflib (which handles binary)."""
    if not os.path.exists(glb_path):
        print(f"  [{label}] MISSING: {glb_path}", flush=True)
        return
    size = os.path.getsize(glb_path)
    print(f"  [{label}] {glb_path} ({size/1024:.1f} KB)", flush=True)
    try:
        from pygltflib import GLTF2
        g = GLTF2().load_binary(glb_path)
        nverts = 0
        for mesh in g.meshes:
            for prim in mesh.primitives:
                if prim.attributes.POSITION is not None:
                    nverts += g.accessors[prim.attributes.POSITION].count
        print(f"    verts={nverts}  meshes={len(g.meshes)}  nodes={len(g.nodes)}  "
              f"skins={len(g.skins)}  anims={[a.name for a in g.animations]}  "
              f"materials={len(g.materials)}  images={len(g.images)}", flush=True)
    except Exception as e:
        print(f"    (glb-info error: {e})", flush=True)


def run_probe(use_remesh, output_subdir):
    """Run Hunyuan3D-Paint once with the given use_remesh setting."""
    out_dir = os.path.join(OUT_ROOT, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Copy input into a working dir — Hunyuan writes sibling files (white_mesh_remesh.obj etc.)
    work_glb = os.path.join(out_dir, "input.glb")
    shutil.copy2(INPUT_GLB, work_glb)

    output_glb = os.path.join(out_dir, "output_textured.glb")

    print(f"\n==== RUN use_remesh={use_remesh} out={out_dir} ====", flush=True)
    print(f"  input : {work_glb}", flush=True)
    print(f"  ref   : {REFERENCE_IMG}", flush=True)
    dump_glb_info("BEFORE", work_glb)

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    t0 = time.time()
    print("  loading pipeline (may take 30-60s first time)...", flush=True)
    pipeline = Hunyuan3DPaintPipeline(conf)
    print(f"  pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    result_path = pipeline(
        mesh_path=work_glb,
        image_path=REFERENCE_IMG,
        output_mesh_path=output_glb,
        use_remesh=use_remesh,
        save_glb=True,
    )
    print(f"  pipeline() returned {result_path!r} after {time.time()-t1:.1f}s", flush=True)

    # Enumerate everything Hunyuan wrote
    print(f"  outputs in {out_dir}:", flush=True)
    for fn in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, fn)
        print(f"    {fn}  ({os.path.getsize(fp)/1024:.1f} KB)", flush=True)

    dump_glb_info("AFTER (output_textured.glb)", output_glb.replace(".obj", ".glb")
                  if not output_glb.endswith(".glb") else output_glb)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    # RUN 1: default remesh path
    run_probe(use_remesh=True, output_subdir="run1_remesh_true")
    # (RUN 2 deferred — pipeline load is expensive; try no-remesh only if run 1 succeeds)


if __name__ == "__main__":
    main()
