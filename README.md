# PhyCam-Eval

PhyCam-Eval is a Python reference implementation for testing object
detectors under physically scaled optical defocus. It models a complex pupil,
Fourier-optical PSFs and OTFs, finite source-cell and photosite integration,
and MTF-matched comparator blur.

The model is a controlled synthetic benchmark, not a calibrated commercial
camera. The public implementation makes its optical assumptions, numerical
validation, detector protocol, and claim boundaries inspectable.

## Install

```bash
python -m pip install -e .
```

Install optional detector dependencies with:

```bash
python -m pip install -e ".[eval]"
```

Detector evaluation is pinned to PyTorch 2.11.0 and TorchVision 0.26.0 to
match the validated runtime contract.

## Example

```python
import numpy as np

from phycam_eval import LDRCaptureSeverity, render_ldr
from phycam_eval.reference_profiles import synthetic_coco_ldr_native_profile

image = np.full((64, 96, 3), 0.25, dtype=np.float64)
result = render_ldr(
    image,
    synthetic_coco_ldr_native_profile(),
    LDRCaptureSeverity(edge_waves_ref=0.75),
    image_id="example",
)

rendered = result.output_frame.array
```

The input is decoded to linear-light sRGB before optical formation and encoded
back to display sRGB afterward. Untouched input and the modeled-neutral output
remain separate references.

The separate `render_forward` interface accepts declared scene-linear inputs
and carries formation through photosites, electrons, RAW ADC values, and the
ISP. The reported detector experiment uses the decoded-LDR path above.

## Detector evaluation

The reference-study path is separate from the lower-level benchmark helpers. Its
defaults are inspectable without COCO or model files:

```bash
phycam-coco-study protocol
```

### Reference study protocol

| Item | Frozen setting |
| --- | --- |
| Dataset | All 5,000 COCO val2017 images, ordered by image ID |
| Optical path | Decoded-LDR, representative-wavelength pupil to PSF/OTF, exact source-cell/photosite integration |
| Disabled stages | Sensor noise, CFA/demosaicing, gain/ADC, motion/rolling readout, and any separate tone curve |
| Defocus grid | 0.5, 0.75, 1.0, 1.5, 2.0, and 3.0 edge-to-center waves at 550 nm |
| Primary pair | YOLOv8n, primary profile, physical defocus minus MTF50-matched finite Gaussian |
| Other detectors | Faster R-CNN, RetinaNet, and DETR at physical anchors 0.5, 1.5, and 3.0 |
| Uncertainty | 2,000 paired image-cluster bootstrap replicates, seed 20260715 |
| Primary estimand | Trapezoidal signed COCO AP curve AUC over the declared wave grid |

The command plans the experiment, runs resumable detector shards, and analyzes
the completed 67-cell layout. A reproduction of the same frozen scientific
protocol is:

```bash
phycam-coco-study plan \
  --coco-root /path/to/coco \
  --output-plan outputs/publication/study-plan.json

phycam-coco-study run \
  --plan outputs/publication/study-plan.json \
  --coco-root /path/to/coco \
  --detector yolov8n \
  --artifact /path/to/yolov8n.pt \
  --output-root outputs/publication/runs

phycam-coco-study run \
  --plan outputs/publication/study-plan.json \
  --coco-root /path/to/coco \
  --detector fasterrcnn_r50_fpn \
  --artifact /path/to/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
  --output-root outputs/publication/runs

phycam-coco-study run \
  --plan outputs/publication/study-plan.json \
  --coco-root /path/to/coco \
  --detector retinanet_r50_fpn_v2 \
  --artifact /path/to/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth \
  --output-root outputs/publication/runs

phycam-coco-study run \
  --plan outputs/publication/study-plan.json \
  --coco-root /path/to/coco \
  --detector detr_r50 \
  --artifact /path/to/facebook-detr-resnet-50 \
  --output-root outputs/publication/runs

phycam-coco-study analyze \
  --plan outputs/publication/study-plan.json \
  --coco-root /path/to/coco \
  --output-root outputs/publication/runs
```

The plan records exact profile, dataset, checkpoint, adapter, preprocessing,
and runtime identities. The runner rejects a checkpoint whose bytes do not
match the frozen allocation. Supplying `--max-images`, changing the profile or
wave grid, or overriding the bootstrap settings creates an exploratory
protocol variant. Exact protocol matching is recorded separately from
inferential status; new local reruns are reproductions, not new confirmatory
studies.

COCO data and pretrained checkpoints are not included. Optional detector
frameworks, datasets, and model weights remain subject to their upstream
licenses and terms; in particular, the pinned Ultralytics package is
AGPL-3.0-licensed. The study implementation verifies the exact checkpoint
identities used for the reported experiments.

The reference design keeps exact protocol matching separate from inferential
status. Historical manuscript results and private publication artifacts are
not part of this public software checkout.

## Tests

The public suite is intentionally compact and covers the optical model,
sampling, comparator matching, capture paths, detector adapters, COCO metrics,
and validation evidence.

```bash
python -m pip install -e ".[dev,eval]"
python -m pytest
```

From a GitHub checkout, the numerical optics checks can be regenerated for a
local COCO copy with:

```bash
python scripts/generate_validation_evidence.py --help
```

## Repository layout

- `phycam_eval/` — camera model and evaluation implementation.
- `tests/` — compact scientific and integration test suite.
- `scripts/` — numerical-validation entry point.

## Citation and license

Citation metadata is in the repository's
[`CITATION.cff`](https://github.com/parthiv-nair/phycam-eval/blob/master/CITATION.cff).
The public software and supporting validation script are released under the
[`MIT License`](LICENSE). Private manuscript sources and figure assets are not
included. Third-party datasets and model weights are not redistributed and
retain their original terms.
