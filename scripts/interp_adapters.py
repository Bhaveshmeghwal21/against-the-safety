"""Linear interpolation between two LoRA adapters that share one architecture (weight-space interpolation, as in WiSE-FT).

  python scripts/interp_adapters.py --a runs/.../main/adapter --b runs/.../candD/adapter --alpha 0.25 --out out/interp/candD_a25/adapter

result = (1 - alpha) * a + alpha * b for every tensor (both LoRA factors separately). alpha 0 is `a`, alpha 1 is `b`. The candidates of
round 3 continue from the main adapter, so their weights lie close to it and the blend is a one-parameter path between the two.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def interpolate(a_dir: Path, b_dir: Path, alpha: float, out: Path) -> dict:
    a, b = load_file(str(a_dir / "adapter_model.safetensors")), load_file(str(b_dir / "adapter_model.safetensors"))
    if a.keys() != b.keys():
        raise SystemExit("the two adapters do not have the same tensors")
    mixed = {}
    for k in a:
        if a[k].shape != b[k].shape:
            raise SystemExit(f"shape mismatch for {k}: {tuple(a[k].shape)} vs {tuple(b[k].shape)}")
        mixed[k] = ((1 - alpha) * a[k].float() + alpha * b[k].float()).to(b[k].dtype).contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(mixed, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    for f in b_dir.iterdir():  # config, tokenizer and chat template: the evaluation loads the tokenizer from the adapter folder
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copyfile(f, out / f.name)
    num = sum(float(((b[k].float() - a[k].float()) ** 2).sum()) for k in a) ** 0.5
    den = sum(float((a[k].float() ** 2).sum()) for k in a) ** 0.5
    info = {"alpha": alpha, "a": str(a_dir), "b": str(b_dir), "tensors": len(a), "relative_distance_b_from_a": round(num / den, 4)}
    (out.parent / "interp.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(interpolate(Path(args.a), Path(args.b), args.alpha, Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
