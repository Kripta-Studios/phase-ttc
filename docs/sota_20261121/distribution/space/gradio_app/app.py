"""Interactive CPU inference for the two pinned PHASE-TTC releases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / "models.json").read_text())


@lru_cache(maxsize=2)
def model_path(name: str) -> Path:
    """Load only the fixed release selected by the maintainer."""
    if name not in MODELS:
        raise ValueError("Select a released model")
    local = os.environ.get("PHASE_TTC_LOCAL_MODELS")
    if local:
        return Path(local) / name
    return Path(
        snapshot_download(
            repo_id=f"AlvaroSchwiedop/{name}",
            revision=MODELS[name],
            local_dir=Path(tempfile.gettempdir()) / "phase-ttc-models" / f"{name}-{MODELS[name]}",
        )
    )


def preview(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Show the final prepared RGB crop and accumulated positive event planes."""
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) > 40 or sum(i.file_size for i in z.infolist()) > 128 * 1024**2:
            raise ValueError("Input exceeds the 128 MB expanded input limit")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        images = {}
        for modality in ["rgb", "event"]:
            records = [r for r in meta["observations"] if r["modality"] == modality]
            if not records:
                images[modality] = None
                continue
            record = max(records, key=lambda r: r["anchor_us"])
            pixels = data[record["pixels"]][-1]
            if modality == "rgb":
                images[modality] = (pixels.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            else:
                plane = pixels[:5].sum(0)
                plane = np.nan_to_num(plane, nan=0, posinf=0, neginf=0)
                images[modality] = (
                    (plane / max(float(plane.max()), 1e-6) * 255).clip(0, 255).astype(np.uint8)
                )
    return images["rgb"], images["event"]


def run_model(name: str, uploaded: str | None) -> tuple:
    """Execute inference in an isolated process and return the measured prediction."""
    root = model_path(name)
    path = Path(uploaded) if uploaded else root / "examples/synthetic_looming.npz"
    if path.stat().st_size > 20 * 1024**2:
        raise gr.Error("Use an NPZ smaller than 20 MB, containing one prepared query.")
    try:
        rgb, events = preview(path)
        dest = Path(tempfile.mkdtemp(prefix="phase_ttc_")) / "prediction.json"
        command = [
            sys.executable,
            str(root / "predict.py"),
            "--input",
            str(path),
            "--device",
            "cpu",
            "--output",
            str(dest),
        ]
        if not uploaded:
            command += ["--verify", str(root / "examples/synthetic_expected.json")]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            error = (
                result.stderr.strip().splitlines()[-1]
                if result.stderr.strip()
                else "Inference failed"
            )
            raise ValueError(error)
        output = json.loads(dest.read_text())
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise gr.Error(f"Input could not be processed: {exc}. See the input format below.") from exc
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("Inference exceeded 120 seconds. Try a smaller prepared query.") from exc
    output["model_revision"] = MODELS[name]
    output["input_kind"] = "uploaded observations" if uploaded else "synthetic execution fixture"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    value = output["ttc_seconds"]
    summary = (
        f"### Predicted TTC: {value:.4f} s"
        if value is not None
        else "### No estimate available for this query"
    )
    summary += "\n\n" + (
        "Uploaded prepared observations."
        if uploaded
        else "Synthetic example · reference check passed · no accuracy target."
    )
    return summary, rgb, events, output, str(dest)


with gr.Blocks(title="PHASE-TTC | Try the released models", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# From motion to time\nRun PHASE-TTC on causal event and RGB observations.")
    gr.Markdown(
        "The original nine component model measures motion in each sensor and fuses it in phase. "
        "The separate EvTTC adaptation adds a geometric quality router."
    )
    with gr.Row():
        with gr.Column(scale=1):
            selector = gr.Radio(list(MODELS), value="PHASE-TTC", label="Released model")
            upload = gr.File(
                label="Optional: upload a prepared query", file_types=[".npz"], type="filepath"
            )
            gr.Markdown(
                "Leave the file empty to use synthetic expanding shapes. "
                "This example checks execution and has no ground truth accuracy score."
            )
            run = gr.Button("Run inference", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown("### Ready to estimate TTC\nChoose a model and run the example.")
            with gr.Row():
                rgb = gr.Image(
                    label="Final prepared RGB crop",
                    interactive=False,
                    height=240,
                    alt_text="Final RGB crop from the selected input",
                )
                events = gr.Image(
                    label="Positive event planes, summed",
                    interactive=False,
                    height=240,
                    alt_text="Sum of the positive event planes from the selected input",
                )
    with gr.Accordion("Prediction details and download", open=False):
        details = gr.JSON(label="Inference output")
        download = gr.File(label="Download prediction JSON", interactive=False)
    run.click(
        run_model,
        [selector, upload],
        [summary, rgb, events, details, download],
        api_name="predict",
        concurrency_limit=1,
    )
    gr.Markdown(
        "## Use your own observations\nDownload a model and run its `extract_example.py` "
        "on your own EvTTC benchmark copy. The resulting NPZ contains prepared sensor crops, "
        "timestamps and supplied object regions. Ordinary images and videos need an adapter. "
        "[Input format](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC/blob/main/INPUT_FORMAT.md)."
    )
    gr.Markdown(
        "## Results and scope\nThe original model was the first of three entries "
        "in the archived GarlTTC Test leaderboard on 28 October 2026: 35.80 Overall MiD, "
        "20.49% below Garl-TTC. A different RGB + geometry head without event inputs "
        "scored 23.47238 on 18 November. A third submission on 21 November, box-only "
        "comparator 1063261, scored 27.97563: a 4.50325-point gap. Its crucial / small / "
        "large / negative bands were 26.38540 / 28.00007 / 32.93790 / 30.89121, versus "
        "24.36932 / 23.95223 / 22.89423 / 18.12626 for RGB + geometry. Both use supplied "
        "region history; only the better head adds RGB estimates. This is a within-account "
        "protocol comparison, not a company or matched end-to-end sensor comparison. The "
        "public ranking display after the third submission has not been archived. Endpoint "
        "replay reproduces the RGB + geometry ZIP byte for byte, while media replay "
        "covers 12/5,805 physical queries, one per sequence, but retains the frozen public event "
        "tensor; it passes the endpoint gate on 0/12 and the geometry-TTC gate on 5/12. "
        "TRAIN40 projection reproduces 88,744/88,744 integer boxes. On 597 "
        "development-exposed FCWD queries, robust geometry plus event content scale records "
        "16.1% relative TTC error versus 26.8% for the public Garl-TTC event model run "
        "locally; protocols are not matched, and this FCWD model is distinct from "
        "submissions 1058935 and 1063261. Isolated p95 compute is 36.22 / 0.98 / "
        "official RGB + geometry model on GPU / box-only model on CPU / robust geometry "
        "+ event content scale on CPU, with prior history cached and excluding reading, "
        "queueing and input production. "
        "The adaptation has separate local EvTTC results and "
        "no verified official leaderboard score. This demo runs inference on CPU; its "
        "response time is "
        "not the paper's latency benchmark. Signed estimates and unavailable outputs retain "
        "their model meaning."
    )
    gr.Markdown(
        "[Download original](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) · "
        "[Download adaptation](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) · "
        "[Read the paper](https://schwiedop.com/papers/phase-ttc/) · "
        "[Versioned evidence](https://github.com/Kripta-Studios/phase-ttc/tree/"
        "8fa8dda64bcdf846331191eb4ad4c8171b979699/docs/sota_20261121)\n\n"
        "Uploaded files are processed on this Space. Use the local commands for private data. "
        "[Hugging Face privacy](https://huggingface.co/privacy) · "
        "[Terms](https://huggingface.co/terms-of-service)"
    )


if __name__ == "__main__":
    theme = gr.themes.Base(primary_hue="blue", font=[gr.themes.GoogleFont("Manrope"), "sans-serif"])
    demo.queue(max_size=12).launch(
        server_name="0.0.0.0",
        max_file_size="20mb",
        theme=theme,
        css=".gradio-container{max-width:1120px!important}h1,h2,h3{text-wrap:balance}p{text-wrap:pretty}"
        "h1{max-width:680px!important}button{font-weight:600!important}",
    )
