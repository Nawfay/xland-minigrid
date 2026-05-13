"""Standalone FPS benchmark for xland-minigrid + ActorCriticRNN.

Measures aggregate (across all local devices) steps-per-second for:
  1. env-only loop (random actions, no policy)
  2. env + policy loop (policy forward pass + env step)

Usage:
    python benchmark.py
    python benchmark.py --env_id XLand-MiniGrid-R1-9x9 --num_envs_list "[1024,4096,8192]" --num_steps 512
"""

import time
from dataclasses import dataclass, field
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import pyrallis
from flax.jax_utils import replicate
from nn import ActorCriticRNN

import xminigrid
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper


@dataclass
class BenchConfig:
    env_id: str = "XLand-MiniGrid-R1-9x9"
    benchmark_id: str = "trivial-21k"
    # aggregate env counts (split evenly across local devices)
    num_envs_list: List[int] = field(default_factory=lambda: [1024, 4096, 8192, 16384])
    num_steps: int = 512
    warmup_steps: int = 64
    # network (mirror train_meta_task.py defaults)
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    seed: int = 0


def _build_env(cfg: BenchConfig):
    env, env_params = xminigrid.make(cfg.env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    benchmark = xminigrid.load_benchmark(cfg.benchmark_id)
    return env, env_params, benchmark


def _bench_env_only(cfg: BenchConfig, env, env_params, benchmark, num_envs: int) -> float:
    num_devices = jax.local_device_count()
    assert num_envs % num_devices == 0, f"num_envs={num_envs} must be divisible by num_devices={num_devices}"
    per_device = num_envs // num_devices
    num_actions = env.num_actions(env_params)

    @partial(jax.pmap, axis_name="devices")
    def setup(rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        ruleset_rng = jax.random.split(r1, per_device)
        reset_rng = jax.random.split(r2, per_device)
        rulesets = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
        params = env_params.replace(ruleset=rulesets)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(params, reset_rng)
        return rng, params, timestep

    @partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(3,))
    def run(rng, params, timestep, num_steps):
        def body(carry, _):
            rng, ts = carry
            rng, _r = jax.random.split(rng)
            actions = jax.random.randint(_r, (per_device,), 0, num_actions)
            ts = jax.vmap(env.step, in_axes=0)(params, ts, actions)
            return (rng, ts), None

        (rng, ts), _ = jax.lax.scan(body, (rng, timestep), None, num_steps)
        return rng, ts

    rng = jax.random.split(jax.random.key(cfg.seed), num=num_devices)
    rng, params, timestep = setup(rng)

    # warmup
    rng, timestep = run(rng, params, timestep, cfg.warmup_steps)
    jax.block_until_ready(timestep)

    t = time.time()
    rng, timestep = run(rng, params, timestep, cfg.num_steps)
    jax.block_until_ready(timestep)
    elapsed = time.time() - t
    return (cfg.num_steps * num_envs) / elapsed


def _bench_env_policy(cfg: BenchConfig, env, env_params, benchmark, num_envs: int) -> float:
    num_devices = jax.local_device_count()
    assert num_envs % num_devices == 0
    per_device = num_envs // num_devices

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=cfg.obs_emb_dim,
        action_emb_dim=cfg.action_emb_dim,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        rnn_num_layers=cfg.rnn_num_layers,
        head_hidden_dim=cfg.head_hidden_dim,
        img_obs=False,
    )
    shapes = env.observation_shape(env_params)
    init_obs = {
        "obs_img": jnp.zeros((per_device, 1, *shapes["img"])),
        "obs_dir": jnp.zeros((per_device, 1, shapes["direction"])),
        "prev_action": jnp.zeros((per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=per_device)
    rng = jax.random.key(cfg.seed)
    rng, _rng = jax.random.split(rng)
    net_params = network.init(_rng, init_obs, init_hstate)
    net_params = replicate(net_params, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    @partial(jax.pmap, axis_name="devices")
    def setup(rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        ruleset_rng = jax.random.split(r1, per_device)
        reset_rng = jax.random.split(r2, per_device)
        rulesets = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
        params = env_params.replace(ruleset=rulesets)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(params, reset_rng)
        prev_action = jnp.zeros(per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(per_device)
        return rng, params, timestep, prev_action, prev_reward

    @partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(7,))
    def run(rng, env_p, timestep, prev_action, prev_reward, hstate, net_params, num_steps):
        def body(carry, _):
            rng, ts, pa, pr, h = carry
            dist, _, h = network.apply(
                net_params,
                {
                    "obs_img": ts.observation["img"][:, None],
                    "obs_dir": ts.observation["direction"][:, None],
                    "prev_action": pa[:, None],
                    "prev_reward": pr[:, None],
                },
                h,
            )
            rng, _r = jax.random.split(rng)
            action = dist.sample(seed=_r).squeeze(1)
            ts = jax.vmap(env.step, in_axes=0)(env_p, ts, action)
            return (rng, ts, action, ts.reward, h), None

        (rng, ts, pa, pr, h), _ = jax.lax.scan(body, (rng, timestep, prev_action, prev_reward, hstate), None, num_steps)
        return rng, ts, pa, pr, h

    rng = jax.random.split(jax.random.key(cfg.seed + 1), num=num_devices)
    rng, env_p, timestep, prev_action, prev_reward = setup(rng)

    # warmup
    out = run(rng, env_p, timestep, prev_action, prev_reward, init_hstate, net_params, cfg.warmup_steps)
    jax.block_until_ready(out)

    t = time.time()
    out = run(rng, env_p, timestep, prev_action, prev_reward, init_hstate, net_params, cfg.num_steps)
    jax.block_until_ready(out)
    elapsed = time.time() - t
    return (cfg.num_steps * num_envs) / elapsed


@pyrallis.wrap()
def main(cfg: BenchConfig):
    env, env_params, benchmark = _build_env(cfg)
    num_devices = jax.local_device_count()
    print(f"Devices: {jax.devices()} (n={num_devices})")
    print(f"Env:     {cfg.env_id}")
    print(f"Bench:   {cfg.benchmark_id}")
    print(f"Steps:   {cfg.num_steps} (warmup {cfg.warmup_steps})")
    print()
    print(f"{'num_envs':>10} {'per_device':>12} {'env_fps':>14} {'env+pol_fps':>14}")
    for n in cfg.num_envs_list:
        if n % num_devices != 0:
            print(f"{n:>10}  (skipped: not divisible by {num_devices})")
            continue
        env_fps = _bench_env_only(cfg, env, env_params, benchmark, n)
        pol_fps = _bench_env_policy(cfg, env, env_params, benchmark, n)
        print(f"{n:>10} {n // num_devices:>12} {env_fps:>14,.0f} {pol_fps:>14,.0f}")


if __name__ == "__main__":
    main()
