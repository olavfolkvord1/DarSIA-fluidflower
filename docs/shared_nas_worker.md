# Whole-run workers on a shared NAS

Every machine uses the same Git commit and the same `requirements-worker.lock`.
Use an existing environment matching that lock. No code or input installation
is performed when a task is claimed. Run from the checkout's `src` directory
so Python imports that checkout rather than another editable installation.

```powershell
git fetch <shared-repository>
git checkout <job-commit>
Set-Location <checkout>/src
python -m darsia.fluidflower.worker --job \\server\share\jobs\example --backend cpu --workers 2
```

The same command supports `--backend cuda`. One process computes each complete
run; `--workers` controls the number of simultaneous runs. The parent is the
single writer of its machine/backend log. A console remains active and takes
the next compatible task. `--once` processes at most one task per worker and
`--exit-when-idle` is useful for qualification jobs. Separate qualification job
folders preserve each test's results.

The worker checks its Git HEAD, clean worktree and the repository lockfile SHA
once at startup. Each task states the same commit, frozen calibration SHA and
working precision as `job.json`. Paths to scientific inputs and outputs are
UNC paths. The worker reads only the claimed run's scientific configuration
and its specified raw images, fixed phase maps, measured fields and calibration
directly from the NAS. There are no input batches, transfer services, per-run
generated programs or RPC calls.

The job contains `queue`, `claimed/<machine>`, `results/<run>`, `done`, `failed`
and `logs`. An atomic rename claims a task. Actual input/frame progress renews
its claim; after ten minutes without progress it can return to the queue. Each
attempt has a distinct output directory. A reclaimed attempt cannot publish
over the current owner's output. Completed files are published once, without
replacing an existing result. Partial results are preserved.

Each run produces a Main-format `completed.json`, frame publications and
scientific products. The N1 entrypoint accepts explicit saved models, measured
geometry, the frozen calibration and ordered frame data. It neither fits nor
selects a different method. A single precision setting selects float32 or a
float64 check; a job cannot mix the two. Original geometry integration and
parameter/reduction precision are recorded separately from pixel precision.

Errors are written verbatim to the console, the machine log and `failed`; the
worker continues to the next run. Startup errors stop the worker before it
claims anything. There is no deadline to the first image.

Two small controllers enqueue explicit task JSON and aggregate completed
runs. The aggregator requires the entire declared `done` set and genuine
per-run completions. Neither controller starts a remote process or computes
an image. Series selections, calibrations, validation scripts and job files
belong with the series on the NAS, rather than in library code.
