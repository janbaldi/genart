"""Self-contained HTML contact sheet."""

import base64
import html
import json
from pathlib import Path

import imageio.v3 as iio
from skimage import transform

THUMB = 360  # thumbnail long side in px


def build(collection: Path, technique: str | None = None) -> Path:
    """Write ``collection/sheet.html`` with every render (or one technique's), sorted by t."""
    items = []
    for side in sorted((collection / "renders").glob("*.json")):
        meta = json.loads(side.read_text())
        if technique and technique not in meta["genome"]["technique_chain"]:
            continue
        img = iio.imread(side.with_suffix(".png"))
        s = THUMB / max(img.shape[:2])
        thumb = transform.resize(
            img, (round(img.shape[0] * s), round(img.shape[1] * s)), anti_aliasing=True
        )
        png = iio.imwrite("<bytes>", (thumb * 255).round().astype("uint8"), extension=".png")
        items.append((meta["genome"]["t"], meta, base64.b64encode(png).decode()))
    items.sort(key=lambda x: x[0])

    cards = "\n".join(
        f'<figure data-id="{html.escape(m["id"])}"><img src="data:image/png;base64,{b64}">'
        f"<figcaption><b>{html.escape(m['id'])}</b> t={t:.2f}<br>"
        f"{html.escape(' → '.join(m['genome']['technique_chain']))} {html.escape(_choices(m))}"
        "</figcaption></figure>"
        for t, m, b64 in items
    )
    page = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(collection.name)} contact sheet</title>
<style>
body {{ background:#2b2a28; color:#ddd8cf; font:13px/1.4 system-ui, sans-serif; margin:24px; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:18px; }}
figure {{ margin:0; cursor:pointer; }}
figure img {{ width:100%; display:block; }}
figure.copied figcaption b {{ color:#9fd39b; }}
</style>
<h1>{html.escape(collection.name)}</h1>
<p>{len(items)} renders · click a thumbnail to copy its genome id</p>
<main>
{cards}
</main>
<script>
document.querySelectorAll('figure').forEach(f => f.onclick = () => {{
  navigator.clipboard.writeText(f.dataset.id);
  f.classList.add('copied'); setTimeout(() => f.classList.remove('copied'), 800);
}});
</script>
"""
    out = collection / (f"sheet-{technique}.html" if technique else "sheet.html")
    out.write_text(page)
    return out


def _choices(meta: dict) -> str:
    """Categorical genome params, e.g. ``contour · radial``."""
    return " · ".join(v for v in meta["genome"]["params"].values() if isinstance(v, str))
