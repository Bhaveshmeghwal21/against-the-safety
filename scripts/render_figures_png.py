"""Render docs/figures/*.svg to docs/figures/png/*.png with headless Microsoft Edge (for places that cannot show SVG, such as a Hugging Face Article).

  python scripts/render_figures_png.py [headline tradeoff ...]      # default: the five README figures

Needs Edge. The browser profile is kept inside the repository's .cache folder.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
EDGE = [Path(p) for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]
DEFAULT = ["pipeline", "headline", "tradeoff", "trace_examples", "rounds"]


def main() -> None:
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit("Microsoft Edge was not found; install it or render the SVG files another way.")
    profile = ROOT / ".cache" / "edge_profile"
    profile.mkdir(parents=True, exist_ok=True)
    out = FIG / "png"
    out.mkdir(exist_ok=True)
    for name in sys.argv[1:] or DEFAULT:
        svg = FIG / f"{name}.svg"
        head = svg.read_text(encoding="utf-8")[:400]
        w, h = (int(re.search(rf'{k}="(\d+)"', head).group(1)) for k in ("width", "height"))
        target = out / f"{name}.png"
        target.unlink(missing_ok=True)
        subprocess.run([str(edge), "--headless", "--disable-gpu", f"--user-data-dir={profile}", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--screenshot={target}", f"--window-size={w},{h}", svg.as_uri()], check=False, capture_output=True, timeout=120)
        print("wrote" if target.exists() else "FAILED", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
