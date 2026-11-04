"""Render a bounded real EvTTC excerpt and exact model-input visualizations.

Run with --repo pointing at the research checkout. It reads the frozen query
manifest and published per-query results; it never executes or fits a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ATTRIBUTION = 'EvTTC: An Event Camera Dataset for Time-to-Collision Estimation\nKaizhen Sun, Jinghang Li, Kuan Dai, Bangyan Liao, Wei Xiong and Yi Zhou.\nProject and license declaration: https://nail-hnu.github.io/EvTTC/\nPublication: https://arxiv.org/abs/2412.05053\nDerived research illustration: CCRm-low-0-overlap-0, frames 287-409.\nNo original HDF5 dataset is redistributed.\n\nThe Clear BSD License\n\nCopyright (c) 2024 Neuromorphic Automation and Intelligence Lab (NAIL), China\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:\n\n     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.\n\n     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.\n\n     * Neither the name of Neuromorphic Automation and Intelligence Lab (NAIL), China nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.\n\nNO EXPRESS OR IMPLIED LICENSES TO ANY PARTY\'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\n'

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_window(handle: h5py.File, start: int, end: int) -> dict[str, np.ndarray]:
    group = handle['prophesee/event_cam_left']
    ms = group['ms_map_idx']
    lo = int(ms[max(0, start // 1000)])
    hi = int(ms[min(len(ms) - 1, (end + 999) // 1000 + 1)])
    times = np.asarray(group['t'][lo:hi], dtype=np.int64)
    a, b = np.searchsorted(times, [start, end], side='left')
    return {k: np.asarray(group[k][lo + int(a):lo + int(b)]) for k in ['x', 'y', 't', 'p']}


def polarity_picture(raw: dict[str, np.ndarray], width: int, height: int) -> Image.Image:
    position = raw['y'].astype(np.int64) * width + raw['x'].astype(np.int64)
    positive = np.log1p(np.bincount(position[raw['p'] > 0], minlength=width * height)).reshape(height, width)
    negative = np.log1p(np.bincount(position[raw['p'] <= 0], minlength=width * height)).reshape(height, width)
    occupied = np.concatenate([positive[positive > 0], negative[negative > 0]])
    scale = max(float(np.quantile(occupied, .95)), 1) if len(occupied) else 1
    p, n = np.clip(positive / scale, 0, 1), np.clip(negative / scale, 0, 1)
    base = np.full((height, width, 3), [11, 26, 39], dtype=np.float64)
    base += p[..., None] * np.array([31, 180, 149])
    base += n[..., None] * np.array([220, 111, 85])
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)))


def hash_arrays(hasher, raw):
    for key in ['x', 'y', 't', 'p']:
        hasher.update(key.encode())
        hasher.update(str(raw[key].dtype).encode())
        hasher.update(raw[key].tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo / 'src'), str(repo)]
    from operational.train40_runtime.encoding import map_roi_lut, encode_mapped_lut, crop_uint8_roi_first
    from operational.rgb_port_revision.transfer_inputs import sensor_triplets
    from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes

    site = Path(__file__).resolve().parents[1]
    output = site / 'public/assets/research/replay-ccrm'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ATTRIBUTION.txt').write_text(ATTRIBUTION, encoding='utf-8')
    url = '/assets/research/replay-ccrm/'
    manifest_file = repo / 'artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json'
    results_file = repo / 'paper/data/transfer_queries.csv'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    sequence = 'CCRm-low-0-overlap-0'
    queries = sorted([r for r in manifest['rows'] if r['sequence_id'] == sequence], key=lambda r: r['anchor_us'])
    with results_file.open(encoding='utf-8', newline='') as handle:
        results = {r['query_id']: r for r in csv.DictReader(handle)}
    assert len(queries) == 16
    raw_path = Path(queries[0]['raw_path'])
    assert raw_path.stat().st_size == queries[0]['raw_stat']['size_bytes']
    start_frame, end_frame = 287, 409
    frame_hash, event_hash = hashlib.sha256(), hashlib.sha256()
    frames, published_queries = [], []
    channel_names = [f'positive_bin_{i}' for i in range(1, 6)] + [f'negative_bin_{i}' for i in range(1, 6)] + ['log_event_count', 'log_event_rate']
    with h5py.File(raw_path, 'r') as handle:
        clock = np.asarray(handle['blackflys/left/ts'], dtype=np.int64)
        start_us = int(clock[start_frame])
        rgb_data = handle['blackflys/left/data']
        ffmpeg = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x320', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'replay.mp4')], stdin=subprocess.PIPE)
        for number, index in enumerate(range(start_frame, end_frame + 1)):
            rgb = np.asarray(rgb_data[index], dtype=np.uint8)
            timestamp = int(clock[index])
            raw = event_window(handle, timestamp - 100_000, timestamp)
            frame_hash.update(index.to_bytes(4, 'little')); frame_hash.update(rgb.tobytes())
            hash_arrays(event_hash, raw)
            composite = Image.new('RGB', (1024, 320), (11, 26, 39))
            # Swap channels 0 and 2 only for camera display to match the
            # publisher visualization. HDF5 metadata does not declare channel
            # encoding; model crops preserve the frozen runtime channel order.
            composite.paste(Image.fromarray(rgb[..., ::-1]).resize((512, 320), Image.Resampling.LANCZOS), (0, 0))
            events = polarity_picture(raw, 1280, 720).resize((512, 288), Image.Resampling.LANCZOS)
            composite.paste(events, (512, 16))
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(composite.tobytes())
            if number == 0:
                composite.save(output / 'poster.jpg', quality=92)
            frames.append({'index': number, 'source_frame_index': index, 'time_s': number / 20, 'sensor_elapsed_s': (timestamp - start_us) / 1e6, 'timestamp_us': timestamp, 'event_count': len(raw['t']), 'event_window_us': [timestamp - 100_000, timestamp]})
            if number % 20 == 0:
                print(f'Video frame {number + 1}/123', flush=True)
        ffmpeg.stdin.close()
        assert ffmpeg.wait() == 0

        for number, row in enumerate(queries):
            anchor = int(row['anchor_us'])
            rgb_square = common_square_from_boxes(row['rgb_boxes'], [1, 2], margin_fraction=.25)
            event_square = tuple(row['square_xyxy'])
            rgb_sheet, event_sheet = Image.new('RGB', (384, 128)), Image.new('RGB', (384, 128))
            latest_tensor = None
            for endpoint, (index, window) in enumerate(zip(row['rgb_frame_indices'], row['windows_us'])):
                rgb = np.asarray(rgb_data[int(index)], dtype=np.uint8)
                crop = crop_uint8_roi_first(rgb, rgb_square, size=128).transpose(1, 2, 0)
                rgb_sheet.paste(Image.fromarray(crop), (128 * endpoint, 0))
                raw = event_window(handle, int(window[0]), int(window[1]))
                mapped = map_roi_lut(raw, event_square, 128)
                event_sheet.paste(polarity_picture(mapped, 128, 128), (128 * endpoint, 0))
                latest_tensor = encode_mapped_lut(mapped, int(window[0]), int(window[1]), 128, sequence).numpy()
            rgb_sheet.save(output / f'query-{number:02d}-rgb.jpg', quality=96)
            event_sheet.save(output / f'query-{number:02d}-events.png')
            combined = Image.new('RGB', (384, 256))
            combined.paste(rgb_sheet, (0, 0)); combined.paste(event_sheet, (0, 128))
            combined.save(output / f'query-{number:02d}-crops.jpg', quality=96)
            channels = Image.new('RGB', (768, 256))
            assert latest_tensor is not None
            for plane in range(12):
                normalized = np.clip(latest_tensor[plane] / (2 if plane < 10 else 20), 0, 1)
                color = np.array([49, 211, 178]) if plane < 5 else np.array([244, 143, 116]) if plane < 10 else np.array([203, 220, 235])
                picture = 11 + normalized[..., None] * (color - 11)
                channels.paste(Image.fromarray(np.uint8(picture)), ((plane % 6) * 128, (plane // 6) * 128))
            channels.save(output / f'query-{number:02d}-channels.png')
            values = results[row['query_id']]
            index = int(row['rgb_frame_indices'][-1])
            assert int(clock[index]) == anchor
            published_queries.append({'id': row['query_id'], 'index': number, 'time_s': (index - start_frame) / 20, 'sensor_elapsed_s': (anchor - start_us) / 1e6, 'timestamp_us': anchor, 'frame_index': index, 'gt_ttc': float(values['target_ttc']), 'rgb_ttc': float(values['R_CTX_TRAIN40']), 'event_ttc': float(values['E_CTX_TRAIN40']), 'fused_ttc': float(values['PHASE_ONLY_TRAIN40_S7']), 'crops_url': url + f'query-{number:02d}-crops.jpg', 'channels_url': url + f'query-{number:02d}-channels.png', 'rgb_triplet_url': url + f'query-{number:02d}-rgb.jpg', 'event_triplet_url': url + f'query-{number:02d}-events.png', 'rgb_timestamps_us': row['rgb_timestamps_us'], 'event_windows_us': row['windows_us'], 'rgb_square_xyxy': list(rgb_square), 'event_square_xyxy': list(event_square), 'history_rgb_observations': len(sensor_triplets(clock, anchor)), 'history_event_observations': 4, 'channel_stats': [{'name': name, 'min': float(latest_tensor[i].min()), 'max': float(latest_tensor[i].max()), 'mean': float(latest_tensor[i].mean())} for i, name in enumerate(channel_names)]})
            print(f'Exact input query {number + 1}/16', flush=True)

    data = {'sequence_id': sequence, 'dataset': 'EvTTC', 'fps': 20, 'duration_s': 123 / 20, 'sensor_duration_s': (int(clock[end_frame]) - start_us) / 1e6, 'start_timestamp_us': start_us, 'video_url': url + 'replay.mp4', 'poster_url': url + 'poster.jpg', 'provenance_url': url + 'provenance.json', 'channel_names': channel_names, 'frames': frames, 'queries': published_queries, 'prediction_policy': 'Sixteen frozen recorded queries; no prediction interpolation or new inference.', 'evaluation_scope': 'Development-exposed EvTTC transfer study. Scalar scenario TTC, not per-object ground truth.', 'display_policy': 'Camera playback swaps channels 0 and 2 to match the publisher visualization; the stored channel encoding is not declared in the HDF5 metadata. Model input crops preserve the frozen runtime channel order. RGB and events remain in separate camera coordinate systems. Events show preceding half-open 100 ms windows. Movie uses 20 Hz CFR; sidecar preserves each exact sensor timestamp. Input channels are preprocessing measurements, not learned activations.'}
    (output / 'replay-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (site / 'public/assets/phase-real-data.js').write_text('window.PHASE_REAL_DATA = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    sources = ['operational/train40_runtime/encoding.py', 'operational/rgb_port_revision/transfer_inputs.py', 'src/e_jepa_ttc/data/event_v4_geometry.py', 'src/e_jepa_ttc/representations/voxel_grid.py', 'src/e_jepa_ttc/rgb_port/data.py']
    provenance = {'schema': 'phase_ttc_web_real_excerpt_v1', 'dataset': 'EvTTC', 'project_url': 'https://nail-hnu.github.io/EvTTC/', 'license': 'BSD-3-Clause-Clear', 'license_url': 'https://spdx.org/licenses/BSD-3-Clause-Clear.html', 'attribution_file': 'ATTRIBUTION.txt', 'sequence_id': sequence, 'source_file': raw_path.name, 'source_bytes': raw_path.stat().st_size, 'source_frame_range_inclusive': [start_frame, end_frame], 'rgb_dataset': 'blackflys/left/data', 'rgb_timestamp_dataset': 'blackflys/left/ts', 'event_group': 'prophesee/event_cam_left', 'rgb_used_bytes_sha256': frame_hash.hexdigest(), 'event_window_used_bytes_sha256': event_hash.hexdigest(), 'query_manifest_sha256': digest(manifest_file), 'query_results_sha256': digest(results_file), 'query_manifest_repository_path': str(manifest_file.relative_to(repo)).replace('\\', '/'), 'query_results_repository_path': str(results_file.relative_to(repo)).replace('\\', '/'), 'extraction_code_sha256': digest(Path(__file__)), 'runtime_source_sha256': {name: digest(repo / name) for name in sources}, 'transformations': ['Camera display swaps channels 0 and 2 to match the publisher visualization and downsamples with Lanczos to 512x320; no frame interpolation. The stored channel encoding is not declared in the HDF5 metadata. Original stored bytes are hashed. Publisher visualization used for display comparison: https://raw.githubusercontent.com/NAIL-HNU/EvTTC/main/assets/gif/CCRm.gif', 'Native event coordinates accumulated by polarity over [timestamp-100000,timestamp); log1p counts scaled by occupied95th percentile for display, then downsampled512x288 and letterboxed.', '128x128 RGB crops call crop_uint8_roi_first with common_square_from_boxes over the last two manifest boxes and preserve the raw runtime channel order. JPEG contact sheets are display encodings, not lossless tensors.', 'Exact12channel event tensor calls map_roi_lut and encode_mapped_lut using manifest square and native100ms window.', 'Channel visualizations clip values to[0,2] for ten normalized voxel planes and[0,20] for the two logcount/lograte planes; actual min/max/mean values retained in sidecar.'], 'caveats': ['Selected research illustration is not representative benchmark evidence.', 'Recorded predictions are frozen TRAIN40 transfer outputs on previously exposed data.', 'Spatial RGB/event registration uses separate sensor frames; no overlay claims.', 'Raw HDF5 and annotations are not redistributed.'], 'assets': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in output.iterdir() if p.is_file() and p.name != 'provenance.json'}}
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(frames)} real frames and {len(published_queries)} recorded queries to {output}', flush=True)


if __name__ == '__main__':
    main()
