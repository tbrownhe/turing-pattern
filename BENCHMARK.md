# Engine benchmark

P1 optimized the existing server engine before any browser/WASM prototype. The
benchmark is checked in as a script rather than kept as a one-off notebook:

```console
uv run --with-requirements backend/requirements-dev.lock \
  python scripts/engine_benchmark.py --label optiplex-5060
```

Run that command on the production OptiPlex after stopping other heavy jobs and
paste its JSON into this document. The numbers below are development-workstation
measurements and must not be used to raise production concurrency.

## Before and after

Workstation: Windows 11, Intel family 6 model 183, 24 logical CPUs. Both runs used
a 256 x 256 grid, 100 steps, and the same controls and seed.

| Measurement | Pre-P1 | P1 engine | Change |
| --- | ---: | ---: | ---: |
| dtype | float64 | float32 | half-size concentrations |
| simulation time | 0.430 s | 0.0688 s | 6.25x faster |
| iterations/second | 232.5 | 1,452.8 | 6.25x higher |
| PNG encode | 26.7 ms | 6.25 ms | 4.3x faster |
| concentration bytes | 1,048,576 | 524,288 | 50% lower |
| persistent parameter bytes | 2,097,152 | 4,096 | 99.8% lower |
| process RSS | 95.2 MB | 66.9 MB | 29.7% lower |
| roll Laplacian | 935 us | 758 us | reference only |
| selected convolution | 278 us | 239 us | about 3.2x vs roll reference |

The P1 run used NumPy 2.4.6 with SciPy OpenBLAS. MKL was installed separately in
the old environment but NumPy was not linked to it, so the unused `mkl` dependency
was removed. The convolution candidates agreed with the roll reference within
`3.4e-16` in float64. A deterministic float32 fixture now protects the supported
engine output to `rtol=1e-6, atol=1e-7`.

## Decisions

- Keep periodic boundaries and the measured 3 x 3 SciPy convolution.
- Default to float32 for preview/render work.
- Store endpoint gradients as broadcastable 1-D vectors rather than four repeated
  2-D arrays.
- Keep the conservative production default of two admitted jobs and two worker
  threads until the OptiPlex benchmark and load test say otherwise.
- Keep preview at 256 x 256, 10 FPS maximum, and 25 steps per frame for now.
- Defer WebGL/WASM/WebGPU comparison. That will be a separate, learning-oriented
  prototype with fixture comparisons before it can replace the server fallback.

## Production measurements still required

The exact-host run must record this script's JSON plus the load probe results:

```console
uv run --with-requirements backend/requirements-dev.lock \
  python scripts/load_test.py --clients 2 --excess 1 --renders 2
```

Also use browser performance tools to record frame receive/decode/paint latency and
a 30-minute memory trace. Python can measure generation and encoding, but it cannot
truthfully measure browser paint. Only after those measurements should CPU limits,
session count, FPS, or steps per frame be increased.

The hardened local container check admitted two live sessions, rejected the third,
delivered 42 frames in two seconds, and returned to zero active/waiting work. Across
15 concurrent health samples, mean latency was 12.5 ms and p95/max were 20.6 ms;
the probe reported no errors. This validates bounded behavior, not OptiPlex sizing.

## Production OptiPlex 5060 baseline

Recorded on the production container host running Linux 6.8 with an Intel Core
i5-8500T (six physical cores, one thread per core), Python 3.11.15, NumPy 2.4.6,
and SciPy OpenBLAS. Grid and simulation state were 256 x 256 float32.

| Measurement | Default native pool | One native thread |
| --- | ---: | ---: |
| 1,000-step time | 2.3988 s | 2.4015 s |
| iterations/second | 416.87 | 416.40 |
| process threads | 11 | 1 |
| PNG encode | 6.73 ms | 6.75 ms |
| process RSS | 64.8 MB | 65.0 MB |
| selected convolution | 721.6 us | 720.7 us |
| roll reference | 1,834.8 us | 1,876.1 us |

The 0.11% throughput difference is noise, while removing ten unused native threads
makes two-worker resource behavior much more predictable. Production therefore sets
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. At 416 iterations/second, 25
steps plus encoding take about 66.7 ms, so one session has headroom beneath the
10 FPS cap. Two-session behavior still requires the deployed load test below.
