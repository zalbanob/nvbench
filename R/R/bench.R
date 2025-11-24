nvbenchr_call <- function(name, ...) {
  .Call(name, ..., PACKAGE = "nvbenchr")
}

.nvbench_wrap_stream <- function(ptr) {
  CudaStream$new(ptr)
}

.nvbench_wrap_launch <- function(ptr) {
  Launch$new(ptr)
}

.nvbench_wrap_benchmark <- function(ptr) {
  Benchmark$new(ptr)
}

.nvbench_wrap_state <- function(ptr) {
  State$new(ptr)
}

CudaStream <- R6::R6Class(
  "CudaStream",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    addressof = function() {
      nvbenchr_call("nvbenchr_stream_addressof", self$ptr)
    }
  )
)

Launch <- R6::R6Class(
  "Launch",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    get_stream = function() {
      .nvbench_wrap_stream(nvbenchr_call("nvbenchr_launch_get_stream", self$ptr))
    }
  )
)

Benchmark <- R6::R6Class(
  "Benchmark",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    get_name = function() {
      nvbenchr_call("nvbenchr_benchmark_get_name", self$ptr)
    },
    add_int64_axis = function(name, values) {
      nvbenchr_call("nvbenchr_benchmark_add_int64_axis", self$ptr, name, as.numeric(values))
      invisible(self)
    },
    add_int64_power_of_two_axis = function(name, values) {
      nvbenchr_call(
        "nvbenchr_benchmark_add_int64_power_of_two_axis",
        self$ptr,
        name,
        as.numeric(values)
      )
      invisible(self)
    },
    add_float64_axis = function(name, values) {
      nvbenchr_call("nvbenchr_benchmark_add_float64_axis", self$ptr, name, as.numeric(values))
      invisible(self)
    },
    add_string_axis = function(name, values) {
      nvbenchr_call("nvbenchr_benchmark_add_string_axis", self$ptr, name, as.character(values))
      invisible(self)
    },
    set_name = function(name) {
      nvbenchr_call("nvbenchr_benchmark_set_name", self$ptr, name)
      invisible(self)
    },
    set_is_cpu_only = function(is_cpu_only) {
      nvbenchr_call("nvbenchr_benchmark_set_is_cpu_only", self$ptr, isTRUE(is_cpu_only))
      invisible(self)
    },
    set_run_once = function(run_once) {
      nvbenchr_call("nvbenchr_benchmark_set_run_once", self$ptr, isTRUE(run_once))
      invisible(self)
    },
    set_skip_time = function(duration_seconds) {
      nvbenchr_call("nvbenchr_benchmark_set_skip_time", self$ptr, as.numeric(duration_seconds))
      invisible(self)
    },
    set_timeout = function(duration_seconds) {
      nvbenchr_call("nvbenchr_benchmark_set_timeout", self$ptr, as.numeric(duration_seconds))
      invisible(self)
    },
    set_throttle_threshold = function(threshold) {
      nvbenchr_call("nvbenchr_benchmark_set_throttle_threshold", self$ptr, as.numeric(threshold))
      invisible(self)
    },
    set_throttle_recovery_delay = function(delay_seconds) {
      nvbenchr_call(
        "nvbenchr_benchmark_set_throttle_recovery_delay",
        self$ptr,
        as.numeric(delay_seconds)
      )
      invisible(self)
    },
    set_stopping_criterion = function(criterion) {
      nvbenchr_call("nvbenchr_benchmark_set_stopping_criterion", self$ptr, criterion)
      invisible(self)
    },
    set_criterion_param_int64 = function(name, value) {
      nvbenchr_call("nvbenchr_benchmark_set_criterion_param_int64", self$ptr, name, as.numeric(value))
      invisible(self)
    },
    set_criterion_param_float64 = function(name, value) {
      nvbenchr_call(
        "nvbenchr_benchmark_set_criterion_param_float64",
        self$ptr,
        name,
        as.numeric(value)
      )
      invisible(self)
    },
    set_criterion_param_string = function(name, value) {
      nvbenchr_call("nvbenchr_benchmark_set_criterion_param_string", self$ptr, name, as.character(value))
      invisible(self)
    },
    set_min_samples = function(count) {
      nvbenchr_call("nvbenchr_benchmark_set_min_samples", self$ptr, as.numeric(count))
      invisible(self)
    }
  )
)

State <- R6::R6Class(
  "State",
  public = list(
    ptr = NULL,
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    has_device = function() {
      nvbenchr_call("nvbenchr_state_has_device", self$ptr)
    },
    has_printers = function() {
      nvbenchr_call("nvbenchr_state_has_printers", self$ptr)
    },
    get_device = function() {
      nvbenchr_call("nvbenchr_state_get_device", self$ptr)
    },
    get_stream = function() {
      .nvbench_wrap_stream(nvbenchr_call("nvbenchr_state_get_stream", self$ptr))
    },
    get_int64 = function(name) {
      nvbenchr_call("nvbenchr_state_get_int64", self$ptr, name)
    },
    get_int64_or_default = function(name, default_value) {
      nvbenchr_call("nvbenchr_state_get_int64_or_default", self$ptr, name, as.numeric(default_value))
    },
    get_float64 = function(name) {
      nvbenchr_call("nvbenchr_state_get_float64", self$ptr, name)
    },
    get_float64_or_default = function(name, default_value) {
      nvbenchr_call("nvbenchr_state_get_float64_or_default", self$ptr, name, as.numeric(default_value))
    },
    get_string = function(name) {
      nvbenchr_call("nvbenchr_state_get_string", self$ptr, name)
    },
    get_string_or_default = function(name, default_value) {
      nvbenchr_call("nvbenchr_state_get_string_or_default", self$ptr, name, as.character(default_value))
    },
    add_element_count = function(count, column_name = "") {
      nvbenchr_call("nvbenchr_state_add_element_count", self$ptr, as.numeric(count), as.character(column_name))
      invisible(NULL)
    },
    set_element_count = function(count) {
      nvbenchr_call("nvbenchr_state_set_element_count", self$ptr, as.numeric(count))
      invisible(NULL)
    },
    get_element_count = function() {
      nvbenchr_call("nvbenchr_state_get_element_count", self$ptr)
    },
    skip = function(reason) {
      nvbenchr_call("nvbenchr_state_skip", self$ptr, reason)
      invisible(NULL)
    },
    is_skipped = function() {
      nvbenchr_call("nvbenchr_state_is_skipped", self$ptr)
    },
    get_skip_reason = function() {
      nvbenchr_call("nvbenchr_state_get_skip_reason", self$ptr)
    },
    add_global_memory_reads = function(nbytes, column_name = "") {
      nvbenchr_call(
        "nvbenchr_state_add_global_memory_reads",
        self$ptr,
        as.numeric(nbytes),
        as.character(column_name)
      )
      invisible(NULL)
    },
    add_global_memory_writes = function(nbytes, column_name = "") {
      nvbenchr_call(
        "nvbenchr_state_add_global_memory_writes",
        self$ptr,
        as.numeric(nbytes),
        as.character(column_name)
      )
      invisible(NULL)
    },
    get_benchmark = function() {
      .nvbench_wrap_benchmark(nvbenchr_call("nvbenchr_state_get_benchmark", self$ptr))
    },
    get_throttle_threshold = function() {
      nvbenchr_call("nvbenchr_state_get_throttle_threshold", self$ptr)
    },
    set_throttle_threshold = function(threshold_fraction) {
      nvbenchr_call("nvbenchr_state_set_throttle_threshold", self$ptr, as.numeric(threshold_fraction))
      invisible(NULL)
    },
    get_min_samples = function() {
      nvbenchr_call("nvbenchr_state_get_min_samples", self$ptr)
    },
    set_min_samples = function(count) {
      nvbenchr_call("nvbenchr_state_set_min_samples", self$ptr, as.numeric(count))
      invisible(NULL)
    },
    get_disable_blocking_kernel = function() {
      nvbenchr_call("nvbenchr_state_get_disable_blocking_kernel", self$ptr)
    },
    set_disable_blocking_kernel = function(flag) {
      nvbenchr_call("nvbenchr_state_set_disable_blocking_kernel", self$ptr, isTRUE(flag))
      invisible(NULL)
    },
    get_run_once = function() {
      nvbenchr_call("nvbenchr_state_get_run_once", self$ptr)
    },
    set_run_once = function(flag) {
      nvbenchr_call("nvbenchr_state_set_run_once", self$ptr, isTRUE(flag))
      invisible(NULL)
    },
    get_timeout = function() {
      nvbenchr_call("nvbenchr_state_get_timeout", self$ptr)
    },
    set_timeout = function(duration) {
      nvbenchr_call("nvbenchr_state_set_timeout", self$ptr, as.numeric(duration))
      invisible(NULL)
    },
    get_blocking_kernel_timeout = function() {
      nvbenchr_call("nvbenchr_state_get_blocking_kernel_timeout", self$ptr)
    },
    set_blocking_kernel_timeout = function(duration) {
      nvbenchr_call("nvbenchr_state_set_blocking_kernel_timeout", self$ptr, as.numeric(duration))
      invisible(NULL)
    },
    collect_cupti_metrics = function() {
      nvbenchr_call("nvbenchr_state_collect_cupti_metrics", self$ptr)
      invisible(NULL)
    },
    is_cupti_required = function() {
      nvbenchr_call("nvbenchr_state_is_cupti_required", self$ptr)
    },
    exec = function(fn, batched = TRUE, sync = FALSE) {
      if (!is.function(fn)) {
        stop("exec expects a function accepting a Launch")
      }
      nvbenchr_call("nvbenchr_state_exec", self$ptr, fn, isTRUE(batched), isTRUE(sync))
      invisible(NULL)
    },
    get_short_description = function() {
      nvbenchr_call("nvbenchr_state_get_short_description", self$ptr)
    },
    add_summary = function(name, value) {
      nvbenchr_call("nvbenchr_state_add_summary", self$ptr, name, value)
      invisible(NULL)
    },
    get_axis_values = function() {
      nvbenchr_call("nvbenchr_state_get_axis_values", self$ptr)
    },
    get_axis_values_as_string = function() {
      nvbenchr_call("nvbenchr_state_get_axis_values_as_string", self$ptr)
    },
    get_stopping_criterion = function() {
      nvbenchr_call("nvbenchr_state_get_stopping_criterion", self$ptr)
    }
  )
)

register_benchmark <- function(fn, name = deparse(substitute(fn))) {
  if (!is.function(fn)) {
    stop("register_benchmark expects a function")
  }
  ptr <- nvbenchr_call("nvbenchr_register", fn, as.character(name %||% "benchmark"))
  .nvbench_wrap_benchmark(ptr)
}

`%||%` <- function(x, y) {
  if (is.null(x) || (is.character(x) && length(x) == 0)) y else x
}

suppress_nvbench_unrecognized <- function(argv) {
  bad_prefixes <- c("^--file=", "^--args$", "^--no-echo$", "^--no-restore$", "^--no-save$", "^--slave$")
  keep <- vapply(argv, function(x) {
    !any(grepl(paste(bad_prefixes, collapse = "|"), x))
  }, logical(1))
  argv[keep]
}

run_all_benchmarks <- function(argv = commandArgs(trailingOnly = TRUE)) {
  argv <- suppress_nvbench_unrecognized(argv)
  nvbenchr_call("nvbenchr_run_all_benchmarks", as.character(argv))
  invisible(NULL)
}

# Alias mirroring Python API name
register <- register_benchmark
