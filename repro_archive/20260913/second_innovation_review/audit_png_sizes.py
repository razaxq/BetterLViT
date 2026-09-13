"""Read only PNG headers in a supplied local image directory; no model evaluation."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image_dir", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
files = sorted(args.image_dir.glob("*.png"), key=lambda p: p.name)
if not files:
    raise SystemExit("No PNG files found")
counts = Counter()
headers = hashlib.sha256()
for path in files:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path.name}")
    width, height = struct.unpack(">II", header[16:24])
    counts[f"{width}x{height}"] += 1
    headers.update(path.name.encode("utf-8") + b"\0" + header)
result = {
    "kind": "local_migration_copy_png_header_audit_not_remote_or_pixel_hash_verification",
    "image_dir": str(args.image_dir.resolve()),
    "samples": len(files),
    "sizes": dict(sorted(counts.items())),
    "filename_and_header_sha256": headers.hexdigest(),
}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
