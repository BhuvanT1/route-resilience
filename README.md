# Route Resilience AI

[![CI](https://github.com/BhuvanT1/route-resilience/actions/workflows/ci.yml/badge.svg)](https://github.com/BhuvanT1/route-resilience/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)

**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**

Satellite imagery → road mask → routable graph → criticality → failure simulation → decision support.

Aligned to ISRO/NNRMS downstream utilisation of Sentinel-2, Resourcesat LISS-IV and Cartosat-3.

## Results at a glance

| | |
|---|---|
| **Road segmentation** (U-Net R34, DeepGlobe held-out test) | IoU **0.619** · clDice **0.843** · recall 0.860 |
| vs classical Frangi baseline | IoU 0.065 (**+0.554**) |
| Overfitting check | val–test IoU gap −0.015; rerun reproduced within 0.002 IoU |
| **Occlusion** (498 tiles, tree canopy) | ΔIoU −0.025, occluded-region recall 0.86 |
| **Topology healing** (MST + loop pass vs MST only) | redundancy α 0.075 vs 0.018 (ground truth 0.132) |
| Tests | 49 core + 44 API + 89 contract + 27 metric checks, all run in CI |

Only the canopy occlusion result is claimed; two other occluder types still
score above the unoccluded control and are treated as an open bug. Details in
[Results so far](#results-so-far) and [Limitations](#limitations).

---

## Run it

Python 3.11–3.13, Node 20.

```bash
# 1. backend (CPU only; no CUDA, no GDAL required)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" pydantic python-multipart
uvicorn backend.app.main:app --reload --port 8000

# 2. frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Open the app, click **Launch Urban Digital Twin**, then **Run** on the guided
demonstration. That executes the whole pipeline against the real backend in
about two minutes.

Docker:

```bash
docker compose up --build                     # UI on :8080, API on :8000
WITH_TORCH=true docker compose up --build     # include PyTorch for inference
```

Interactive API docs: `http://localhost:8000/docs`.

---

## Why this is not just a segmentation project

Road segmentation optimises IoU, a pixel-counting metric. A prediction with a
three-pixel break scores almost identically to an unbroken one — but the broken
one yields two disconnected graph components and is useless for routing.

| Stage | Usual approach | Here |
|---|---|---|
| Augmentation | flips, colour jitter | **synthetic occluders** on the image but *not* the label, forcing contextual completion |
| Loss | BCE + Dice | adds **soft-clDice**, comparing skeletons so a break is punished heavily |
| Post-processing | morphological closing | **union–find + MST** bridging with directional and evidence-based cost |
| Output | a mask | a **routable graph** + criticality + resilience + rerouting |

---

## Scientific honesty

This is the constraint the codebase is built around, not a disclaimer bolted on.

**Every road mask carries provenance.** The API cannot return one without it, and
the UI cannot render one without displaying it:

| Mode | Meaning |
|---|---|
| `TRAINED_MODEL` | real inference through a trained checkpoint |
| `CLASSICAL_BASELINE` | Frangi ridge filter — genuine computation, **not** deep learning |
| `DEMO_DATA` | deterministic synthetic city, labelled DEMO throughout |
| `OSM_REFERENCE` | OpenStreetMap rasterisation — ground truth, not a prediction |

**Metrics are never invented.** With no ground truth the API returns
`metrics.available = false` plus the reason, and the UI renders em dashes and
instructions for producing real ground truth. There is no mock data layer in the
frontend: if the backend is down the UI shows an error, not something that looks
like a result.

**Healed road is marked as inferred.** Every bridge is tagged `kind="healed"`,
drawn distinctly, and counted — `healed_length_fraction` reports how much of the
network was inferred rather than observed. Above 20% the UI warns that downstream
criticality is low-confidence.

**Flood scenarios do not predict floods.** They simulate the *closure* of roads in
an area to estimate network consequences. No hydrological model is used.

---

## Architecture

```
frontend/   React 18 + TypeScript + Vite + Tailwind
            canvas network renderer, Recharts, 7 sections
backend/    FastAPI — thin adapter over the scientific core,
            provenance enforced at the schema level
src/        the scientific core: pure numpy / networkx
            graph/ and analytics/ import neither torch nor GDAL
```

Dependencies are tiered so the core installs in under a minute with no CUDA and
no GDAL. That is an architectural rule, not a convenience: it lets the graph
sub-team work while the vision sub-team installs drivers, and keeps the test
suite running in seconds.

---

## The two algorithms worth defending

### Topology healing — union–find + MST

Kruskal over a cost-sorted candidate list adds exactly **(components − 1)**
bridges, provably the minimum that restores connectivity, while union–find
refuses any bridge between fragments already linked by a cheaper path.

```
cost = gap × (1 + w_angle·angular_penalty) × (1 + w_evidence·(1 − evidence))
```

Bridging *through a detected occluder* is cheap; bridging across open ground is
expensive. Bridges above `max_cost` are refused — leaving a real gap beats
inventing a road, because a hallucinated bridge silently corrupts every
downstream centrality number.

**Pure MST healing is wrong on real cities.** A break in a ring road joins two
endpoints of the *same* component, so union–find rejects it and the loop is
destroyed. Connectivity looks perfect while redundancy collapses. Measured
against known ground truth:

| | ground truth | MST only | MST + loop pass |
|---|---|---|---|
| redundancy (α) | 0.132 | 0.018 | 0.075 |
| gatekeeper nodes | 57 | **117** | 52 |
| bridge segments | 70 | **183** | 65 |

MST alone over-reports single points of failure by more than 2×.

### Criticality — betweenness is not enough

Betweenness finds *busy* corridors; articulation points find **gatekeepers**
whose loss disconnects the city. The two rarely coincide: a busy node usually has
alternatives, which is often why it carries traffic. Both are reported, and the
"why is this critical?" explanation is generated from computed statistics —
betweenness rank, OD pairs severed, detour ratio — never from a template.

---

## Resilience Index (primary metric: NER)

**Network Efficiency Resilience.**

```
NER  =  E(G')  /  E(G)

E(H) =  (1 / |P|)  *  SUM over (s,t) in P  of  1 / d_H(s,t),     with 1/inf := 0

G    intact network
G'   network after the failure scenario
P    fixed origin-destination sample, drawn ONCE from G and reused for both terms
d_H  shortest-path length in H, weighted by length_m (or travel_time_s if present)
```

**Range: `0 <= NER <= 1`, by construction.**

| Value | Meaning |
|---|---|
| `1.00` | no sampled route changed; nothing was lost |
| `0.63` | 63% of baseline routing efficiency retained, 37% lost |
| `0.00` | every sampled trip is severed |

### Why it is bounded, and why that is a theorem rather than a clamp

Removing nodes or edges can never create a new path, so for every pair
`d_G'(s,t) >= d_G(s,t)`. Taking reciprocals termwise gives
`1/d_G' <= 1/d_G`, hence `E(G') <= E(G)` and `NER <= 1`. Every term is
non-negative, so `NER >= 0`. No clamping is applied; the only guard in the code
is against floating-point round-off at the 1e-12 level.

Three further properties follow, and each has a regression test:

- **Identity** — an empty scenario scores exactly `1.0`.
- **Monotonicity** — for nested removals `G'' ⊂ G' ⊂ G`, `NER(G'') <= NER(G')`.
  More damage can never raise the score.
- **Totality** — a severed pair contributes `0` rather than infinity, so
  disconnection *lowers* the score instead of making it undefined. Detour and
  disconnection are captured by one number.

### Two secondary readings that are NOT the index

Both are reported for completeness and are deliberately kept out of the primary
field.

**Literal ASPL ratio** (`resilience_index_literal_aspl`) — the project brief's
definition, `ASPL_baseline / ASPL_perturbed`. Mean shortest path length becomes
infinite the moment one OD pair disconnects, so this reads exactly `0.00` for
almost every realistic scenario and cannot rank them. Flagged with
`literal_aspl_is_degenerate`.

**Connected-only ASPL ratio** (`diagnostic_aspl_ratio_connected_only`) —
averages over surviving pairs only. When most trips are severed the survivors
are disproportionately *short* ones, so mean path length falls and the ratio
climbs **above 1**, which would read as a collapsed network improving. Measured
example: a chain network shattered into fragments, 92% of trips impossible,
yields `5.33`. It carries `diagnostic_is_resilience_score: false` and
`diagnostic_survivorship_biased: true`. It is kept because it is a useful
diagnostic of *which* trips survived, and removed from the primary slot because
it is unbounded and non-monotone.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | model status, capabilities, provenance |
| `POST /api/imagery/demo` | deterministic synthetic study area |
| `POST /api/imagery/upload` | GeoTIFF or image upload |
| `POST /api/extract-roads` | segmentation (mode explicit, never implied) |
| `POST /api/build-graph` | skeletonise + heal + statistics + GeoJSON |
| `POST /api/criticality` | betweenness, gatekeepers, explanations |
| `POST /api/simulate-failure` | scenario, impact, resilience |
| `POST /api/route` | baseline vs disrupted routing |

---

## Training

```bash
pip install -r requirements-dl.txt        # read the header: install torch first
python scripts/prepare_deepglobe.py --src data/external/deepglobe --out data/processed
python -m src.models.train --model unet_r34          --data data/processed
python -m src.models.train --model deeplabv3plus_r50 --data data/processed
python -m src.models.train --model segformer_b3      --data data/processed
python -m src.models.train --model segformer_b3      --data data/processed --no-occlusion
python scripts/run_benchmark.py --checkpoints checkpoints/*/best.pt --data data/processed/test
```

Baselines first: without a U-Net trained under identical conditions, "the
transformer is better" is an assertion rather than a measurement. Checkpoints are
selected on **clDice, not IoU** — high IoU with low clDice means dotted-line
roads, pixel-accurate and useless for routing.

The strongest single experiment is the last training command: identical
architecture ± synthetic occluders, compared on `recall_occluded`. It isolates one
variable and tests the project's central claim directly.

Drop a checkpoint in `checkpoints/` and the backend finds it automatically;
`/api/health` then reports `TRAINED_MODEL` and extraction switches to real
inference. Nothing else changes.

---

## Results so far

**Segmentation — U-Net (ResNet-34) on DeepGlobe**, trained on a Kaggle GPU with
synthetic occluders in the augmentation, 40 epochs, seed 42, checkpoint chosen on
validation clDice (best epoch 36). Held-out test set:

| | IoU | Dice | Precision | Recall | clDice |
|---|---:|---:|---:|---:|---:|
| U-Net (R34), trained | **0.619** | 0.753 | 0.694 | 0.860 | **0.843** |
| Frangi ridge filter (classical baseline) | 0.065 | | | | |

The validation–test IoU gap is −0.015, so the model is not overfitting the
validation split. A rerun reproduced an earlier, lost run to within 0.002 IoU.

**Occlusion robustness — 498 test tiles, each scored against its own unoccluded
version:**

| Occluder | ΔIoU vs unoccluded | Status |
|---|---:|---|
| Tree canopy | **−0.025** (occluded-region recall 0.86) | holds up; the defensible result |
| Partial | +0.030 | **unexplained** |
| Clutter | +0.020 | **unexplained** |

Hiding pixels should never *raise* IoU, so the partial and clutter rows are
treated as a bug, not a finding. The first cause was found: the synthetic vehicle
occluder was placed only on road pixels, in one colour, so it marked roads
instead of hiding them. Fixing placement cut the on-road ratio from 1.0 to
0.04–0.32, but both conditions still beat the control, so there is a second cause
not yet identified. Until it is, only the canopy number is claimed.

---

## Testing

```bash
python -m pytest                            # 49 scientific core tests
python backend/tests/test_api_e2e.py        # 44 API + sanity checks
python backend/tests/test_contract.py       # 89 frontend/backend contract checks
python backend/tests/test_resilience_metric.py   # 27 checks on the NER metric's properties
cd frontend && npx tsc --noEmit && npm run build
```

The contract suite checks that every field the React app reads actually exists in
the backend response — the class of bug where the API is perfectly valid and the
UI silently renders `undefined`.

Sanity properties asserted: a no-op scenario causes zero disruption; removing a
gatekeeper disconnects the network; healing adds exactly (components − 1) bridges;
targeted damage exceeds random damage.

---

## Limitations

1. **Healing can hallucinate.** Every bridge is an inference. `max_cost` bounds the
   risk but does not remove it; healed segments are labelled and counted.
2. **Betweenness assumes shortest-path routing**, no capacity limits, uniform demand.
3. **Travel time uses one assumed free-flow speed**, disclosed in every routing
   response, because road class is not available from imagery alone.
4. **Flood scenarios are geometric, not hydrological.**
5. **The synthetic city tests the pipeline, not accuracy.** Segmentation numbers must
   come from held-out real data scored against OSM.
6. **OSM ground truth has a temporal mismatch** with image acquisition dates.
7. **Exact betweenness is O(V·E).** Above the configured node count the API estimates
   from pivots and reports that it did so.

---

## Team

VIT-AP University capstone project (Fall 2026), guided by **Dr. Priyanka Singh**.

| Member | Owns |
|---|---|
| **Themballi Bhuvan** (lead) | segmentation model and occlusion work (Layer 3) |
| **Kashak Singh** | graph construction and topological healing (Layer 4) |
| **Hitesh Kumar** | graph analytics, failure simulation, resilience and dashboard (Layers 5–6) |
| **Tanisha Singh Raphal** | geospatial data pipeline, OSM ground truth, evaluation (Layers 1–2) |

---

## Docs

- `docs/ARCHITECTURE.md` — layers, data flow, design decisions
- `docs/GRAPH_THEORY.md` — healing, centrality and resilience mathematics
- `docs/DEMO.md` — faculty demonstration script
- `docs/INTERFACE_CONTRACT.md` — the seam between the two sub-teams
- `docs/DATA_ACQUISITION.md` — datasets and sensor gotchas
