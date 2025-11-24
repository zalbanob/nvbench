library(nvbenchr)

if (!requireNamespace("nvbenchr.throughput", quietly = TRUE)) {
  stop("Install the example CUDA helper first: R CMD INSTALL R/examples/throughputlib")
}

# Throughput benchmark mirroring python/examples/throughput.py
throughput_bench <- function(state) {
  stride <- state$get_int64("Stride")
  ipt <- state$get_int64("ItemsPerThread")

  nbytes <- 128 * 1024 * 1024
  elements <- as.integer(nbytes / 4L)

  state$add_element_count(elements, column_name = "Elements")
  state$add_global_memory_reads(nbytes, column_name = "Datasize")
  state$add_global_memory_writes(nbytes)

  launcher <- function(launch) {
    stream_ptr <- launch$get_stream()$addressof()
    nvbenchr.throughput::nvbenchr_example_throughput(
      stream_ptr,
      stride,
      elements,
      ipt
    )
  }

  state$exec(launcher)
}

b <- register(throughput_bench)
b$add_int64_axis("Stride", c(1, 2, 4))
b$add_int64_axis("ItemsPerThread", c(1, 2, 3, 4))

run_all_benchmarks(character())
