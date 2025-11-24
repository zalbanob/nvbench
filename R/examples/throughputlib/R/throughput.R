#' Launch CUDA throughput kernel (helper)
nvbenchr_example_throughput <- function(stream_ptr, stride, elements, items_per_thread) {
  .Call("nvbenchr_example_throughput_native",
        stream_ptr,
        stride,
        elements,
        items_per_thread,
        PACKAGE = "nvbenchr.throughput")
}
