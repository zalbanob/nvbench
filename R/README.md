# R bindings for NVBench

This folder contains an R package exposing a minimal API that mirrors the Python
bindings under `python/`. Benchmarks are authored as R functions that receive a
`State` object and can be configured and executed through NVBench.

## Build prerequisites

- CUDA toolkit available on the system
- A built NVBench library (`libnvbench.so`) in `../build` relative to this
  folder (or supply `NVBENCH_LIB_DIR`/`NVBENCH_INCLUDE_DIR` during install)
- R with the `Rcpp` and `R6` packages installed

To build NVBench in this repository (from repo root):

```bash
cmake -B build -S . -DNVBench_ENABLE_INSTALL_RULES=ON
cmake --build build
```

## Installing the package

```bash
cd R
R CMD INSTALL . --configure-vars="NVBENCH_INCLUDE_DIR=../nvbench/include NVBENCH_LIB_DIR=../build"
```

You can override the include/library paths through `--configure-vars` or
environment variables of the same names.

## Examples

Sample R equivalents of the Python examples live in `inst/examples/`:

- `throughput.R`: GPU throughput benchmark. Requires installing the helper CUDA
  package in `examples/throughputlib` via `R CMD INSTALL R/examples/throughputlib`.
- `axes.R`: CPU-only example demonstrating string/int axes.
- `skip.R`: CPU-only example demonstrating skipping configurations.

After installation, run them with:

```bash
Rscript -e "source(system.file('examples/throughput.R', package='nvbenchr'))"
```

## Minimal usage example

```r
library(nvbenchr)

throughput <- function(state) {
  stride <- state$get_int64("Stride")
  state$add_element_count(stride)
  # ... launch your CUDA work through state$exec(...)
}

b <- register_benchmark(throughput)
b$add_int64_axis("Stride", c(1, 2, 4))

run_all_benchmarks(commandArgs(trailingOnly = FALSE))
```

The exposed classes (`Benchmark`, `State`, `Launch`, `CudaStream`) follow the
same naming and method shape as the Python bindings to ease portability between
languages.
