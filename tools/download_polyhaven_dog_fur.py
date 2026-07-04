"""Download a single fur diffuse map for the animated dog.

Originally planned against PolyHaven, but PolyHaven's Textures→Fur category
returned 0 results as of 2026-07 (verified by fetching
https://polyhaven.com/textures/fur). Switched to ambientCG, which is also
CC0 (https://docs.ambientcg.com/license/) and no login required.

Chosen asset: ambientCG Carpet013_1K-JPG — dense reddish-brown short-fibre
shag carpet whose macro structure reads convincingly as short animal fur
when tiled on a small mesh. Initial choice Fabric047 was rejected (it's
a quilted green tarpaulin, not fur-like). If Carpet013 ever 404s, the
earlier Fabric015 / Fabric063 / Carpet010 candidates were also verified
downloadable but look less like fur (basketweave, plain twill, striped
turf respectively).
The URL pattern is https://ambientcg.com/get?file=<AssetID>_<Res>-<Fmt>.zip .

Script keeps the filename dog_fur_diffuse.jpg so the Blender step below
doesn't care where the source came from.
"""

import io
import os
import sys
import urllib.request
import zipfile

ASSET_ID = "Carpet013"
ZIP_URL = f"https://ambientcg.com/get?file={ASSET_ID}_1K-JPG.zip"
OUTPUT_PATH = "/data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg"

# Inside the ambientCG zip: Fabric047_1K-JPG/Fabric047_1K-JPG_Color.jpg (color map).
COLOR_MEMBER_SUFFIXES = ("_Color.jpg", "_Color.JPG", "_color.jpg")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 100_000:
        print(f"ALREADY_EXISTS {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)", flush=True)
        return 0

    req = urllib.request.Request(
        ZIP_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"DOWNLOAD_FAILED {ZIP_URL}: {e}", flush=True)
        return 1

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        print(f"NOT_A_ZIP {ZIP_URL}: {e}", flush=True)
        return 1

    color_member = None
    for name in zf.namelist():
        if any(name.endswith(sfx) for sfx in COLOR_MEMBER_SUFFIXES):
            color_member = name
            break
    if color_member is None:
        print(f"NO_COLOR_MAP_IN_ZIP members={zf.namelist()}", flush=True)
        return 1

    data = zf.read(color_member)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(data)
    print(
        f"DOWNLOADED {OUTPUT_PATH} ({len(data)} bytes) "
        f"from {ZIP_URL} member={color_member}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
