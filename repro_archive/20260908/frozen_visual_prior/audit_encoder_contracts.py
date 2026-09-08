"""Read public, revision-pinned metadata; never download or execute model weights/code.

Python standard library only. Run from any directory:
    python audit_encoder_contracts.py --output encoder_contracts.json
The public DINOv3 files may return HTTP 401/403; that is recorded, not bypassed.
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


MODELS = (
    ("m42-health/CXformer-small", "21777c302a43197b704c6c92591d9897efaaac3b"),
    ("facebook/dinov2-with-registers-small", "0d9846e56b43a21fa46d7f3f5070f0506a5795a9"),
    ("microsoft/rad-dino", "110cbc18d5133582e320b43d53bf5c44e410c936"),
    ("facebook/dinov3-vits16-pretrain-lvd1689m", "114c1379950215c8b35dfcd4e90a5c251dde0d32"),
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "BetterLViT-metadata-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Metadata exceeds the 2 MB per-request bound")
    return data


def file_record(model_id, revision, filename, parse_json=True):
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    try:
        raw = fetch(url)
    except urllib.error.HTTPError as exc:
        if "dinov3" in model_id and exc.code in (401, 403):
            return {"url": url, "http_status": exc.code, "access": "unauthenticated request denied"}
        raise
    record = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if parse_json:
        record["content"] = json.loads(raw)
    else:
        record["handling"] = "Read as bytes for SHA256 only; not executed or vendored"
    return record


def adapter_budget(channels, grid, batch=16, rank=64, output_channels=256, target=56):
    # Proposal: resize frozen features first, then 1x1 -> GELU -> 1x1 at target size.
    # Counts include both convolution biases; MACs exclude bias/GELU/resize/backbone.
    weights = channels * rank + rank * output_channels
    return {
        "input_channels": channels,
        "patch_grid_at_224": [grid, grid],
        "rank": rank,
        "output_channels": output_channels,
        "target_grid": [target, target],
        "trainable_parameters": weights + rank + output_channels,
        "projection_macs_per_image": weights * target * target,
        "batch_size": batch,
        "fp32_raw_patch_feature_bytes": batch * channels * grid * grid * 4,
        "fp32_resized_frozen_feature_bytes": batch * channels * target * target * 4,
        "fp32_injected_feature_bytes": batch * output_channels * target * target * 4,
        "scope": "Analytical individual tensors, not total GPU memory or measured speed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("encoder_contracts.json"))
    args = parser.parse_args()
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Public metadata only; no authentication, weights, training, or dataset access",
        "models": [],
    }
    for model_id, revision in MODELS:
        api_url = f"https://huggingface.co/api/models/{model_id}/revision/{revision}"
        metadata = json.loads(fetch(api_url))
        if metadata["sha"] != revision:
            raise ValueError(f"Revision mismatch: {model_id}")
        item = {
            "model_id": model_id,
            "revision": revision,
            "api_url": api_url,
            "gated": metadata.get("gated"),
            "config": file_record(model_id, revision, "config.json"),
            "processor": file_record(model_id, revision, "preprocessor_config.json"),
        }
        if model_id == "m42-health/CXformer-small":
            item["custom_processor_source"] = file_record(model_id, revision, "custom_processor.py", False)
        config = item["config"].get("content")
        if config:
            item["proposed_adapter_budget"] = adapter_budget(config["hidden_size"], 224 // config["patch_size"])
        else:
            item["proposed_adapter_budget"] = adapter_budget(384, 14)
            item["budget_dimensions_source"] = "Official public DINOv3 model card; gated config not verified"
        report["models"].append(item)
    small = report["models"][0]["proposed_adapter_budget"]
    assert small["trainable_parameters"] == 41280
    assert small["projection_macs_per_image"] == 128450560
    assert small["fp32_injected_feature_bytes"] == 51380224
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "models": len(report["models"]),
                      "small_adapter_parameters": small["trainable_parameters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
