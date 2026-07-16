import argparse, sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
from gpurir_scenes.run_render_pass import run_render_pass
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene
p = argparse.ArgumentParser()
p.add_argument("--room", required=True)
p.add_argument("--out-dir", required=True)
args = p.parse_args()
spec = compose_two_dog_scene()
run_render_pass(spec, args.room, args.out_dir)
print("RENDER_DONE", args.out_dir)
