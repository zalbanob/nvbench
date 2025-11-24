library(nvbenchr)

skip_bench <- function(state) {
  value <- state$get_int64("Value")
  if (value < 0) {
    state$skip("Negative value unsupported")
    return()
  }
  # fake work
  state$add_summary("square", value * value)
}

b <- register(skip_bench)
b$set_is_cpu_only(TRUE)
b$add_int64_axis("Value", c(-1, 0, 1, 2))

run_all_benchmarks(character())
