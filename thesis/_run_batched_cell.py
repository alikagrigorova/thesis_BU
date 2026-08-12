"""Single-cell β* computation using batched NUTS on signals22.
Usage:
  python _run_batched_cell.py <regime> <D> <kappa_or_0> <lo> <hi>
  regime: proportional | sublinear2 | sublinear3
  kappa_or_0: kappa value for proportional, 0 for sublinear regimes
"""
import json, sys, time, os
import numpy as np
from scipy.optimize import brentq
import proportional_asymptotics_utils as pau

regime  = sys.argv[1]
D       = int(sys.argv[2])
kappa_p = float(sys.argv[3])   # kappa for proportional, 0 otherwise
lo      = float(sys.argv[4])
hi      = float(sys.argv[5])

if regime == "proportional":
    kappa = kappa_p
    N = int(round(D / kappa))
elif regime == "sublinear2":
    N = D * D
    kappa = D / N
elif regime == "sublinear3":
    N = D * D * D
    kappa = D / N
else:
    raise ValueError(f"Unknown regime: {regime}")

M          = 500
CHUNK_SIZE = 50    # 500 / 50 = 10 chunks — divides evenly
NUM_WARMUP = 500
NUM_SAMPLES = 2000

SIGMA2_ASSUMED = 1.0
SIGMA2_TRUE    = 2.0
delta_true = pau.default_beta_true(D, seed=0)
v0 = np.full(D, 2.0)

if regime == "proportional":
    tag = f"prop_D{D}_k{kappa_p}"
else:
    tag = f"{regime}_D{D}"

os.makedirs("results", exist_ok=True)
log_file    = f"results/{tag}.log"
result_file = f"results/{tag}.json"

# skip if already done
if os.path.exists(result_file):
    print(f"[{tag}] already done, skipping.", flush=True)
    sys.exit(0)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log(f"regime={regime}  D={D}  N={N}  kappa={kappa:.6f}  M={M}  chunk={CHUNK_SIZE}  bracket=({lo},{hi})")

# ---- d* (exact closed-form, fast) ----
t0 = time.time()
d_star, se_star, _ = pau.well_specified_threshold_knownvar(
    D, N, SIGMA2_ASSUMED, v0, delta_true, M=M, seed=1)
log(f"d* = {d_star:.4f}  SE={se_star:.4f}  [{time.time()-t0:.1f}s]")

# ---- brentq on batched NUTS ----
curve_log = []

def gap_fn(b):
    t = time.time()
    d_mean, se, _ = pau.observed_discrepancy_nuts_knownvar_batched(
        D, N, b, SIGMA2_ASSUMED, SIGMA2_TRUE, v0, delta_true,
        M=M, seed=1, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
        chunk_size=CHUNK_SIZE, log_fn=log)
    gap = d_mean - d_star
    log(f"  beta={b:.4f}: d~={d_mean:.4f}  SE={se:.4f}  gap={gap:+.4f}  [{time.time()-t:.1f}s]")
    curve_log.append({"beta": b, "d_mean": d_mean, "se": se, "gap": gap})
    # save intermediate curve after every eval so partial results survive crashes
    with open(result_file + ".partial", "w") as f:
        json.dump({"tag": tag, "D": D, "N": N, "kappa": kappa,
                   "d_star": d_star, "curve": curve_log}, f, indent=2)
    return gap

t1 = time.time()
try:
    beta_star = brentq(gap_fn, lo, hi, xtol=1e-3)
except ValueError as e:
    log(f"ERROR: brentq failed — {e}  (bracket too tight?)")
    sys.exit(1)

elapsed = time.time() - t1
log(f"beta* = {beta_star:.4f}  [{elapsed:.1f}s]")

result = {
    "regime": regime, "D": D, "N": N, "kappa": kappa, "M": M,
    "d_star": d_star, "se_star": se_star,
    "beta_star": beta_star, "elapsed_brentq_s": elapsed,
    "bracket": [lo, hi], "chunk_size": CHUNK_SIZE,
    "num_warmup": NUM_WARMUP, "num_samples": NUM_SAMPLES,
    "curve": curve_log,
}
print("RESULT_JSON:" + json.dumps(result), flush=True)
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)
log(f"Saved → {result_file}")
