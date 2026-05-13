# Training on a RunPod server

Run `train_meta_task.py` reproducing the XLand-MiniGrid paper config on a single-GPU pod with `uv` + `tmux`. Defaults in `TrainConfig` already match paper Table 6 (`R4-13x13`, `small-1m`, `num_envs=16384`, `total_timesteps=1e10`, etc.) — no CLI flags needed for a paper-faithful run.

## 1. Get the code on the pod

```bash
# from your laptop
cd /Users/nawaf/Documents/foratus/rl2-learning/xland-minigrid
git push
```

```bash
# on the pod
cd /home/xland-minigrid     # or wherever you cloned it
git pull
```

If cloning fresh:
```bash
cd /home && git clone <your-repo-url> xland-minigrid && cd xland-minigrid
```

## 2. Environment setup (one-time per pod)

```bash
# install uv if missing
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL

# pin jax with CUDA 12 into the project so `uv run` won't undo it
uv add "jax[cuda12]"

# sync with the baselines extra (pulls in matplotlib for metrics.png)
uv sync --extra baselines

# verify GPU is visible — expect [CudaDevice(id=0)] with no plugin warnings
uv run python -c "import jax; print(jax.devices())"
```

If you ever see `ALREADY_EXISTS: PJRT_Api already exists for device type cuda`, two cuda plugins got registered. Fix:
```bash
rm -rf .venv
uv add "jax[cuda12]"
uv sync --extra baselines
```

## 3. Sanity check — run the FPS benchmark

```bash
tmux new -s bench
cd training
uv run benchmark.py
# detach: Ctrl-b d
```

You want `env+pol_fps` to scale with `num_envs` and reach ~1M on a single A100 / similar GPU. If env+pol FPS is flat across rows, the GPU isn't actually being used.

## 4. Launch training

```bash
tmux new -s train
cd training
uv run train_meta_task.py --name r4-13x13-paper
# detach: Ctrl-b d
```

The run dir is `./runs/r4-13x13-paper-<timestamp>/`.

Useful overrides:
```bash
# more chatty in-training logs (defaults to every 10 meta updates)
uv run train_meta_task.py --name r4-13x13-paper --log_every_n_updates 5

# smaller smoke test (e.g. before committing 3+ hours of compute)
uv run train_meta_task.py --name smoke --total_timesteps 500_000_000

# different benchmark
uv run train_meta_task.py --name r4-medium --benchmark_id medium-1m
```

Expected wall time at paper config (1e10 transitions, single A100): ~3–5 hours.

## 5. Where artifacts land

```
training/runs/<name>-<timestamp>/
├── config.json       # exact config used (written before training starts)
├── metrics.jsonl     # one JSON line per meta-update with all metrics
├── summary.json      # wall times, aggregate FPS, final eval numbers
├── metrics.png       # grid plot of every metric vs meta-update
└── checkpoint/       # orbax checkpoint (config + params)
```

Pull back to your laptop:
```bash
# from your laptop
scp -r <runpod-host>:/home/xland-minigrid/training/runs/<name>-<ts> ./
```

## 6. tmux quick reference

| action | keys |
|---|---|
| new session | `tmux new -s <name>` |
| detach | `Ctrl-b d` |
| reattach | `tmux attach -t <name>` |
| list sessions | `tmux ls` |
| scroll buffer (enter copy mode) | `Ctrl-b [` then arrow keys, `q` to exit |
| kill session | `tmux kill-session -t <name>` |

## Troubleshooting

- **OOM**: halve `--num_envs` (e.g. 8192). Re-check that `num_envs` is divisible by `num_minibatches` (32) and by `local_device_count`.
- **`uv run` undoes your manual installs**: `uv run` auto-syncs to `pyproject.toml` + `uv.lock` each invocation. Pin things you want kept via `uv add`, not `uv pip install`.
- **Plain `print` calls inside training don't show up**: the whole training loop is one `jax.lax.scan`, so Python `print` only fires when the scan returns. Mid-training logs use `jax.debug.print` and are gated to device 0.
- **Two cuda plugins registered (ALREADY_EXISTS)**: see step 2.
