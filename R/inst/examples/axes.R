library(nvbenchr)

# CPU-only benchmark showing axis configuration similar to python/examples/axes.py
# Uses only integer axes to avoid any string conversions in environments with
# older Rcpp builds.
axes_bench <- function(state) {
  size <- state$get_int64("Size")
  method_id <- state$get_int64("MethodId")

  # pretend work
  dummy <- sum(seq_len(size))
  if (method_id == 1) {
    dummy <- dummy * 2
  }

  state$add_summary("method", ifelse(method_id == 0, "baseline", "fast"))
  state$add_summary("dummy_result", dummy)
}

b <- register(axes_bench)
b$set_is_cpu_only(TRUE)
b$add_int64_axis("Size", c(1e3, 1e4, 1e5))
b$add_int64_axis("MethodId", c(0, 1))

run_all_benchmarks(character())
