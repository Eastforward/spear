"""Background downloader for CC0 wall textures from PolyHaven.

Queries the PolyHaven public API for indoor-wall-friendly textures
(walls / plaster / wallpaper), downloads Diffuse + Normal + Roughness at 2K,
and saves organized under /data/jzy/code/SPEAR/tmp/polyhaven_walls/<slug>/.

No auth required. All PolyHaven assets are CC0.

Runs unattended; safe to kill any time.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error


OUT_ROOT = "/data/jzy/code/SPEAR/tmp/polyhaven_walls"
API_ROOT = "https://api.polyhaven.com"
RES = "2k"                 # 2K is plenty for game walls, keeps sizes ~2 MB
FMT = "jpg"                # jpg preferred: smaller, UE can convert to BC7
MAPS_WANTED = ("Diffuse", "nor_gl", "Rough")
MAX_ASSETS = 20            # download this many wall-appropriate textures
CATEGORIES = ("wall", "plaster", "wallpaper", "brick")


_UA = ("Mozilla/5.0 (X11; Linux x86_64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36 "
       "spear-textures-downloader/0.1 (research)")


def _req(url):
    return urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})


def _get_json(url):
    with urllib.request.urlopen(_req(url), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "cached"
    tmp = dst + ".part"
    with urllib.request.urlopen(_req(url), timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    return "downloaded"


def collect_asset_ids():
    """Union of wall/plaster/wallpaper/brick categories, dedup, keep textures only."""
    seen = {}
    for cat in CATEGORIES:
        try:
            data = _get_json(f"{API_ROOT}/assets?t=textures&c={cat}")
        except urllib.error.URLError as e:
            print(f"[polyhaven] fetch {cat!r} failed: {e}", flush=True)
            continue
        for aid, meta in data.items():
            seen.setdefault(aid, meta)
    return list(seen.items())


def rank_indoor_appropriate(assets):
    """Prefer plaster/wallpaper/paint indoor looks; deprioritize brick/rock heavy assets."""
    def score(pair):
        aid, meta = pair
        cats = set(meta.get("categories", []))
        s = 0
        if "wallpaper" in cats: s += 4
        if "plaster" in cats:   s += 3
        if "wall" in cats:      s += 1
        if "brick" in cats:     s += 0  # neutral
        if "rock" in cats:      s -= 1
        # Names hinting at damage/rust/dirt: mild deprioritize
        low = aid.lower()
        for bad in ("dirty", "rust", "moss", "damaged", "old"):
            if bad in low:
                s -= 1
        return -s  # ascending = best first when negated
    return sorted(assets, key=score)


def download_asset(aid, meta):
    files = _get_json(f"{API_ROOT}/files/{aid}")
    if RES not in files.get("Diffuse", {}):
        return None
    dst_dir = os.path.join(OUT_ROOT, aid)
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"asset_id": aid, "categories": meta.get("categories", []),
                   "authors": meta.get("authors"), "download_count": meta.get("download_count")},
                  f, indent=2)
    results = {}
    for map_kind in MAPS_WANTED:
        entry = files.get(map_kind, {}).get(RES, {}).get(FMT)
        if entry is None:
            results[map_kind] = "missing"
            continue
        url = entry["url"]
        fname = f"{map_kind}.{FMT}"
        dst = os.path.join(dst_dir, fname)
        try:
            status = _download(url, dst)
            results[map_kind] = status
        except (urllib.error.URLError, TimeoutError) as e:
            results[map_kind] = f"err:{e}"
    return results


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    log_path = os.path.join(OUT_ROOT, "download_log.txt")
    log = open(log_path, "a", encoding="utf-8")

    def emit(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    emit("Starting PolyHaven wall texture download.")
    assets = collect_asset_ids()
    emit(f"Collected {len(assets)} candidate asset ids across categories={CATEGORIES}")
    ranked = rank_indoor_appropriate(assets)
    chosen = ranked[:MAX_ASSETS]
    emit(f"Chose top {len(chosen)}: {[a for a, _ in chosen]}")

    successes = 0
    for i, (aid, meta) in enumerate(chosen, 1):
        emit(f"[{i}/{len(chosen)}] {aid}")
        try:
            res = download_asset(aid, meta)
            emit(f"    {aid} -> {res}")
            if res:
                successes += 1
        except Exception as e:
            emit(f"    {aid} FAILED: {e}")

    emit(f"DONE. successes={successes} out_dir={OUT_ROOT}")
    log.close()


if __name__ == "__main__":
    main()
