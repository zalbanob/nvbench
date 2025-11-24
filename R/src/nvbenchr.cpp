// Copyright 2025 NVIDIA Corporation
//
// Licensed under the Apache License, Version 2.0 with the LLVM exception
// (the "License"); you may not use this file except in compliance with
// the License.
//
// You may obtain a copy of the License at
//
//     http://llvm.org/foundation/relicensing/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Rcpp.h>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <variant>
#include <dlfcn.h>

using r_string   = std::string;
using r_strings  = std::vector<r_string>;
using r_numeric  = std::vector<double>;
using r_int64    = nvbench::int64_t;
using r_float64  = nvbench::float64_t;
using r_float32  = nvbench::float32_t;
using bench_ptr  = nvbench::benchmark_base *;
using state_ptr  = nvbench::state *;
using launch_ptr = nvbench::launch *;
using stream_ptr = nvbench::cuda_stream *;

// Fallback shims for older Rcpp that lack precious helpers.
// Exported with default visibility so the dynamic loader can resolve them
// even if Rcpp does not provide the symbols. Guarded so we don't clash with
// newer Rcpp headers that already define these.
extern "C" SEXP Rcpp_precious_preserve(SEXP object) __attribute__((visibility("default")));
extern "C" SEXP Rcpp_precious_remove(SEXP object) __attribute__((visibility("default")));
inline SEXP Rcpp_precious_preserve(SEXP object)
{
  R_PreserveObject(object);
  return object;
}
inline SEXP Rcpp_precious_remove(SEXP object)
{
  R_ReleaseObject(object);
  return R_NilValue;
}

namespace
{

Rcpp::Environment get_namespace()
{
  static Rcpp::Environment env = Rcpp::Environment::namespace_env("nvbenchr");
  return env;
}

SEXP call_wrapper(const char *name, SEXP arg)
{
  auto env = get_namespace();
  Rcpp::Function fn = env.find(name);
  return fn(arg);
}

template <typename T>
SEXP wrap_ptr(T *ptr, const char *cls_name)
{
  Rcpp::XPtr<T> xp(ptr, false);
  xp.attr("class") = cls_name;
  return xp;
}

template <typename T>
T &unwrap_ptr(SEXP obj, const char *expected_class)
{
  if (TYPEOF(obj) != EXTPTRSXP)
  {
    Rcpp::stop("Expected external pointer for %s", expected_class);
  }
  SEXP cls = Rf_getAttrib(obj, R_ClassSymbol);
  if (cls == R_NilValue || TYPEOF(cls) != STRSXP)
  {
    Rcpp::stop("Invalid external pointer class for %s", expected_class);
  }
  const char *cls_cstr = CHAR(STRING_ELT(cls, 0));
  if (expected_class && std::string(cls_cstr) != expected_class)
  {
    Rcpp::stop("Unexpected pointer class '%s' (wanted '%s')", cls_cstr, expected_class);
  }

  auto ptr = static_cast<T *>(R_ExternalPtrAddr(obj));
  if (ptr == nullptr)
  {
    Rcpp::stop("Null pointer for %s", expected_class);
  }
  return *ptr;
}

r_string as_string(SEXP s, const char *name)
{
  try
  {
    return Rcpp::as<r_string>(s);
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Expected string for %s: %s", name, e.what());
  }
}

r_strings as_string_vec(SEXP s, const char *name)
{
  try
  {
    return Rcpp::as<r_strings>(s);
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Expected character vector for %s: %s", name, e.what());
  }
}

std::vector<r_int64> as_int64_vec(SEXP s, const char *name)
{
  try
  {
    auto dbl = Rcpp::as<r_numeric>(s);
    std::vector<r_int64> out;
    out.reserve(dbl.size());
    for (double v : dbl)
    {
      out.push_back(static_cast<r_int64>(v));
    }
    return out;
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Expected numeric vector for %s: %s", name, e.what());
  }
}

std::vector<r_float64> as_float64_vec(SEXP s, const char *name)
{
  try
  {
    auto dbl = Rcpp::as<r_numeric>(s);
    std::vector<r_float64> out;
    out.reserve(dbl.size());
    for (double v : dbl)
    {
      out.push_back(static_cast<r_float64>(v));
    }
    return out;
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Expected numeric vector for %s: %s", name, e.what());
  }
}

bool as_bool(SEXP s, const char *name)
{
  try
  {
    return Rcpp::as<bool>(s);
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Expected logical for %s: %s", name, e.what());
  }
}

template <typename T>
T as_scalar(SEXP s, const char *name)
{
  try
  {
    return Rcpp::as<T>(s);
  }
  catch (const std::exception &e)
  {
    Rcpp::stop("Failed to convert '%s': %s", name, e.what());
  }
}

struct r_function_holder
{
  SEXP fn{};

  r_function_holder() = default;
  explicit r_function_holder(SEXP f) : fn(f) { R_PreserveObject(fn); }
  r_function_holder(const r_function_holder &other) : fn(other.fn) { R_PreserveObject(fn); }
  r_function_holder &operator=(const r_function_holder &) = delete;
  r_function_holder(r_function_holder &&) noexcept        = delete;
  r_function_holder &operator=(r_function_holder &&) noexcept = delete;

  ~r_function_holder() { R_ReleaseObject(fn); }

  void operator()(nvbench::state &state, nvbench::type_list<>)
  {
    SEXP state_ptr = wrap_ptr(&state, "nvbench_state");
    SEXP state_obj = call_wrapper(".nvbench_wrap_state", state_ptr);
    Rcpp::Function callable(fn);
    callable(state_obj);
  }
};

class GlobalBenchmarkRegistry
{
  bool m_finalized;

public:
  GlobalBenchmarkRegistry() : m_finalized(false) {}

  GlobalBenchmarkRegistry(const GlobalBenchmarkRegistry &)            = delete;
  GlobalBenchmarkRegistry &operator=(const GlobalBenchmarkRegistry &) = delete;
  GlobalBenchmarkRegistry(GlobalBenchmarkRegistry &&)                 = delete;
  GlobalBenchmarkRegistry &operator=(GlobalBenchmarkRegistry &&)      = delete;

  bench_ptr add_bench(SEXP fn, const r_string &name)
  {
    if (m_finalized)
    {
      Rcpp::stop("Cannot register benchmarks after execution");
    }
    if (!Rf_isFunction(fn))
    {
      Rcpp::stop("Benchmark must be a function");
    }

    r_function_holder exec(fn);
    auto &bench = nvbench::benchmark_manager::get()
                    .add(std::make_unique<nvbench::benchmark<r_function_holder>>(exec))
                    .set_name(name.empty() ? r_string("benchmark") : name);
    return &bench;
  }

  void run(const r_strings &argv)
  {
    if (nvbench::benchmark_manager::get().get_benchmarks().empty())
    {
      Rcpp::stop("No benchmarks registered");
    }
    if (m_finalized)
    {
      Rcpp::stop("Benchmarks already executed");
    }
    m_finalized = true;

    try
    {
      nvbench::benchmark_manager::get().initialize();
      nvbench::option_parser parser{};
      parser.parse(argv);

      NVBENCH_MAIN_PRINT_PREAMBLE(parser);
      NVBENCH_MAIN_RUN_BENCHMARKS(parser);
      NVBENCH_MAIN_PRINT_EPILOGUE(parser);

      NVBENCH_MAIN_PRINT_RESULTS(parser);
    }
    catch (const std::exception &e)
    {
      Rcpp::stop("nvbench run failed: %s", e.what());
    }
    catch (...)
    {
      Rcpp::stop("Unknown exception while running nvbench benchmarks");
    }
  }
};

std::unique_ptr<GlobalBenchmarkRegistry> global_registry;

bench_ptr bench_from_xptr(SEXP bench)
{
  return &unwrap_ptr<nvbench::benchmark_base>(bench, "nvbench_benchmark");
}

state_ptr state_from_xptr(SEXP state)
{
  return &unwrap_ptr<nvbench::state>(state, "nvbench_state");
}

launch_ptr launch_from_xptr(SEXP launch)
{
  return &unwrap_ptr<nvbench::launch>(launch, "nvbench_launch");
}

stream_ptr stream_from_xptr(SEXP stream)
{
  return &unwrap_ptr<nvbench::cuda_stream>(stream, "nvbench_stream");
}

SEXP wrap_state_obj(nvbench::state &state)
{
  return call_wrapper(".nvbench_wrap_state", wrap_ptr(&state, "nvbench_state"));
}

SEXP wrap_launch_obj(nvbench::launch &launch)
{
  return call_wrapper(".nvbench_wrap_launch", wrap_ptr(&launch, "nvbench_launch"));
}

SEXP wrap_stream_xptr(const nvbench::cuda_stream &stream)
{
  // Safe to const_cast here: XPtr is non-owning and we do not mutate through it.
  auto *ptr = const_cast<nvbench::cuda_stream *>(&stream);
  return wrap_ptr(ptr, "nvbench_stream");
}

} // namespace

extern "C" SEXP nvbenchr_register(SEXP fn, SEXP name)
{
  if (!global_registry)
  {
    Rcpp::stop("nvbenchr registry is not initialized");
  }
  r_string bench_name = as_string(name, "name");
  bench_ptr ptr       = global_registry->add_bench(fn, bench_name);
  return wrap_ptr(ptr, "nvbench_benchmark");
}

extern "C" SEXP nvbenchr_run_all_benchmarks(SEXP argv)
{
  if (!global_registry)
  {
    Rcpp::stop("nvbenchr registry is not initialized");
  }
  auto args = as_string_vec(argv, "argv");
  global_registry->run(args);
  return R_NilValue;
}

extern "C" SEXP nvbenchr_benchmark_get_name(SEXP bench)
{
  return Rcpp::wrap(bench_from_xptr(bench)->get_name());
}

extern "C" SEXP nvbenchr_benchmark_add_int64_axis(SEXP bench, SEXP name, SEXP values)
{
  auto &b = *bench_from_xptr(bench);
  b.add_int64_axis(as_string(name, "name"), as_int64_vec(values, "values"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_add_int64_power_of_two_axis(SEXP bench, SEXP name, SEXP values)
{
  auto &b = *bench_from_xptr(bench);
  b.add_int64_axis(as_string(name, "name"),
                   as_int64_vec(values, "values"),
                   nvbench::int64_axis_flags::power_of_two);
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_add_float64_axis(SEXP bench, SEXP name, SEXP values)
{
  auto &b = *bench_from_xptr(bench);
  b.add_float64_axis(as_string(name, "name"), as_float64_vec(values, "values"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_add_string_axis(SEXP bench, SEXP name, SEXP values)
{
  auto &b = *bench_from_xptr(bench);
  b.add_string_axis(as_string(name, "name"), as_string_vec(values, "values"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_name(SEXP bench, SEXP name)
{
  auto &b = *bench_from_xptr(bench);
  b.set_name(as_string(name, "name"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_is_cpu_only(SEXP bench, SEXP is_cpu_only)
{
  auto &b = *bench_from_xptr(bench);
  b.set_is_cpu_only(as_bool(is_cpu_only, "is_cpu_only"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_run_once(SEXP bench, SEXP run_once)
{
  auto &b = *bench_from_xptr(bench);
  b.set_run_once(as_bool(run_once, "run_once"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_skip_time(SEXP bench, SEXP duration_seconds)
{
  auto &b = *bench_from_xptr(bench);
  b.set_skip_time(as_scalar<r_float64>(duration_seconds, "duration_seconds"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_timeout(SEXP bench, SEXP duration_seconds)
{
  auto &b = *bench_from_xptr(bench);
  b.set_timeout(as_scalar<r_float64>(duration_seconds, "duration_seconds"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_throttle_threshold(SEXP bench, SEXP threshold)
{
  auto &b = *bench_from_xptr(bench);
  b.set_throttle_threshold(as_scalar<r_float32>(threshold, "threshold"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_throttle_recovery_delay(SEXP bench, SEXP delay_seconds)
{
  auto &b = *bench_from_xptr(bench);
  b.set_throttle_recovery_delay(as_scalar<r_float32>(delay_seconds, "delay_seconds"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_stopping_criterion(SEXP bench, SEXP criterion)
{
  auto &b = *bench_from_xptr(bench);
  b.set_stopping_criterion(as_string(criterion, "criterion"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_criterion_param_int64(SEXP bench, SEXP name, SEXP value)
{
  auto &b = *bench_from_xptr(bench);
  b.set_criterion_param_int64(as_string(name, "name"), as_scalar<r_int64>(value, "value"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_criterion_param_float64(SEXP bench, SEXP name, SEXP value)
{
  auto &b = *bench_from_xptr(bench);
  b.set_criterion_param_float64(as_string(name, "name"), as_scalar<r_float64>(value, "value"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_criterion_param_string(SEXP bench, SEXP name, SEXP value)
{
  auto &b = *bench_from_xptr(bench);
  b.set_criterion_param_string(as_string(name, "name"), as_string(value, "value"));
  return bench;
}

extern "C" SEXP nvbenchr_benchmark_set_min_samples(SEXP bench, SEXP count)
{
  auto &b = *bench_from_xptr(bench);
  b.set_min_samples(as_scalar<r_int64>(count, "count"));
  return bench;
}

extern "C" SEXP nvbenchr_state_has_device(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(static_cast<bool>(s.get_device()));
}

extern "C" SEXP nvbenchr_state_has_printers(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_benchmark().get_printer().has_value());
}

extern "C" SEXP nvbenchr_state_get_device(SEXP state)
{
  auto &s = *state_from_xptr(state);
  auto dev = s.get_device();
  if (dev.has_value())
  {
    return Rcpp::wrap(dev.value().get_id());
  }
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_stream(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return wrap_stream_xptr(s.get_cuda_stream());
}

extern "C" SEXP nvbenchr_state_get_int64(SEXP state, SEXP name)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_int64(as_string(name, "name")));
}

extern "C" SEXP nvbenchr_state_get_int64_or_default(SEXP state, SEXP name, SEXP def_value)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_int64_or_default(as_string(name, "name"),
                                           as_scalar<r_int64>(def_value, "default_value")));
}

extern "C" SEXP nvbenchr_state_get_float64(SEXP state, SEXP name)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_float64(as_string(name, "name")));
}

extern "C" SEXP nvbenchr_state_get_float64_or_default(SEXP state, SEXP name, SEXP def_value)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_float64_or_default(as_string(name, "name"),
                                             as_scalar<r_float64>(def_value, "default_value")));
}

extern "C" SEXP nvbenchr_state_get_string(SEXP state, SEXP name)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_string(as_string(name, "name")));
}

extern "C" SEXP nvbenchr_state_get_string_or_default(SEXP state, SEXP name, SEXP def_value)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_string_or_default(as_string(name, "name"), as_string(def_value, "default_value")));
}

extern "C" SEXP nvbenchr_state_add_element_count(SEXP state, SEXP count, SEXP column_name)
{
  auto &s = *state_from_xptr(state);
  s.add_element_count(as_scalar<r_int64>(count, "count"), as_string(column_name, "column_name"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_set_element_count(SEXP state, SEXP count)
{
  auto &s = *state_from_xptr(state);
  s.set_element_count(as_scalar<r_int64>(count, "count"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_element_count(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_element_count());
}

extern "C" SEXP nvbenchr_state_skip(SEXP state, SEXP reason)
{
  auto &s = *state_from_xptr(state);
  s.skip(as_string(reason, "reason"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_is_skipped(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.is_skipped());
}

extern "C" SEXP nvbenchr_state_get_skip_reason(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_skip_reason());
}

extern "C" SEXP nvbenchr_state_add_global_memory_reads(SEXP state, SEXP nbytes, SEXP column_name)
{
  auto &s = *state_from_xptr(state);
  s.add_global_memory_reads(static_cast<std::size_t>(as_scalar<r_int64>(nbytes, "nbytes")),
                            as_string(column_name, "column_name"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_add_global_memory_writes(SEXP state, SEXP nbytes, SEXP column_name)
{
  auto &s = *state_from_xptr(state);
  s.add_global_memory_writes(static_cast<std::size_t>(as_scalar<r_int64>(nbytes, "nbytes")),
                             as_string(column_name, "column_name"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_benchmark(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return wrap_ptr(&s.get_benchmark(), "nvbench_benchmark");
}

extern "C" SEXP nvbenchr_state_get_throttle_threshold(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_throttle_threshold());
}

extern "C" SEXP nvbenchr_state_set_throttle_threshold(SEXP state, SEXP fraction)
{
  auto &s = *state_from_xptr(state);
  s.set_throttle_threshold(as_scalar<r_float32>(fraction, "throttle_fraction"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_min_samples(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_min_samples());
}

extern "C" SEXP nvbenchr_state_set_min_samples(SEXP state, SEXP min_samples_count)
{
  auto &s = *state_from_xptr(state);
  s.set_min_samples(as_scalar<r_int64>(min_samples_count, "min_samples_count"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_disable_blocking_kernel(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_disable_blocking_kernel());
}

extern "C" SEXP nvbenchr_state_set_disable_blocking_kernel(SEXP state, SEXP flag)
{
  auto &s = *state_from_xptr(state);
  s.set_disable_blocking_kernel(as_bool(flag, "disable_blocking_kernel"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_run_once(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_run_once());
}

extern "C" SEXP nvbenchr_state_set_run_once(SEXP state, SEXP flag)
{
  auto &s = *state_from_xptr(state);
  s.set_run_once(as_bool(flag, "run_once"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_timeout(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_timeout());
}

extern "C" SEXP nvbenchr_state_set_timeout(SEXP state, SEXP duration)
{
  auto &s = *state_from_xptr(state);
  s.set_timeout(as_scalar<r_float64>(duration, "duration"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_blocking_kernel_timeout(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_blocking_kernel_timeout());
}

extern "C" SEXP nvbenchr_state_set_blocking_kernel_timeout(SEXP state, SEXP duration)
{
  auto &s = *state_from_xptr(state);
  s.set_blocking_kernel_timeout(as_scalar<r_float64>(duration, "duration"));
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_collect_cupti_metrics(SEXP state)
{
  auto &s = *state_from_xptr(state);
  s.collect_cupti_metrics();
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_is_cupti_required(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.is_cupti_required());
}

extern "C" SEXP nvbenchr_state_exec(SEXP state, SEXP launcher_fn, SEXP batched, SEXP sync)
{
  if (!Rf_isFunction(launcher_fn))
  {
    Rcpp::stop("exec expects a function");
  }
  auto &s        = *state_from_xptr(state);
  bool batched_v = as_bool(batched, "batched");
  bool sync_v    = as_bool(sync, "sync");

  Rcpp::Function launcher(launcher_fn);

  auto cpp_launcher = [&](nvbench::launch &launch) {
    SEXP launch_obj = wrap_launch_obj(launch);
    launcher(launch_obj);
  };

  if (sync_v)
  {
    if (batched_v)
    {
      constexpr auto tag = nvbench::exec_tag::sync;
      s.exec(tag, cpp_launcher);
    }
    else
    {
      constexpr auto tag = nvbench::exec_tag::sync | nvbench::exec_tag::no_batch;
      s.exec(tag, cpp_launcher);
    }
  }
  else
  {
    if (batched_v)
    {
      constexpr auto tag = nvbench::exec_tag::none;
      s.exec(tag, cpp_launcher);
    }
    else
    {
      constexpr auto tag = nvbench::exec_tag::no_batch;
      s.exec(tag, cpp_launcher);
    }
  }

  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_short_description(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_short_description());
}

extern "C" SEXP nvbenchr_state_add_summary(SEXP state, SEXP name, SEXP value)
{
  auto &s = *state_from_xptr(state);
  r_string col_name = as_string(name, "name");
  auto &summ        = s.add_summary("nv/r/" + col_name);
  summ.set_string("description", "User tag: " + col_name);
  summ.set_string("name", col_name);

  switch (TYPEOF(value))
  {
  case INTSXP:
    summ.set_int64("value", static_cast<r_int64>(INTEGER(value)[0]));
    break;
  case REALSXP:
    summ.set_float64("value", REAL(value)[0]);
    break;
  case STRSXP:
    summ.set_string("value", as_string(value, "value"));
    break;
  default:
    Rcpp::stop("Unsupported value type for add_summary");
  }
  return R_NilValue;
}

extern "C" SEXP nvbenchr_state_get_axis_values(SEXP state)
{
  auto &s          = *state_from_xptr(state);
  auto named_vals  = s.get_axis_values();
  auto names       = named_vals.get_names();
  Rcpp::List out;

  for (const auto &name : names)
  {
    if (named_vals.has_value(name))
    {
      auto v = named_vals.get_value(name);
      out[name] =
        std::visit([](auto &&val) { return Rcpp::wrap(val); }, v);
    }
  }

  return out;
}

extern "C" SEXP nvbenchr_state_get_axis_values_as_string(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_axis_values_as_string());
}

extern "C" SEXP nvbenchr_state_get_stopping_criterion(SEXP state)
{
  auto &s = *state_from_xptr(state);
  return Rcpp::wrap(s.get_stopping_criterion());
}

extern "C" SEXP nvbenchr_launch_get_stream(SEXP launch)
{
  auto &l = *launch_from_xptr(launch);
  return wrap_stream_xptr(l.get_stream());
}

extern "C" SEXP nvbenchr_stream_addressof(SEXP stream)
{
  auto &s = *stream_from_xptr(stream);
  return Rcpp::wrap(reinterpret_cast<std::size_t>(s.get_stream()));
}


static const R_CallMethodDef CallEntries[] = {
  {"nvbenchr_register", (DL_FUNC)&nvbenchr_register, 2},
  {"nvbenchr_run_all_benchmarks", (DL_FUNC)&nvbenchr_run_all_benchmarks, 1},
  {"nvbenchr_benchmark_get_name", (DL_FUNC)&nvbenchr_benchmark_get_name, 1},
  {"nvbenchr_benchmark_add_int64_axis", (DL_FUNC)&nvbenchr_benchmark_add_int64_axis, 3},
  {"nvbenchr_benchmark_add_int64_power_of_two_axis",
   (DL_FUNC)&nvbenchr_benchmark_add_int64_power_of_two_axis,
   3},
  {"nvbenchr_benchmark_add_float64_axis", (DL_FUNC)&nvbenchr_benchmark_add_float64_axis, 3},
  {"nvbenchr_benchmark_add_string_axis", (DL_FUNC)&nvbenchr_benchmark_add_string_axis, 3},
  {"nvbenchr_benchmark_set_name", (DL_FUNC)&nvbenchr_benchmark_set_name, 2},
  {"nvbenchr_benchmark_set_is_cpu_only", (DL_FUNC)&nvbenchr_benchmark_set_is_cpu_only, 2},
  {"nvbenchr_benchmark_set_run_once", (DL_FUNC)&nvbenchr_benchmark_set_run_once, 2},
  {"nvbenchr_benchmark_set_skip_time", (DL_FUNC)&nvbenchr_benchmark_set_skip_time, 2},
  {"nvbenchr_benchmark_set_timeout", (DL_FUNC)&nvbenchr_benchmark_set_timeout, 2},
  {"nvbenchr_benchmark_set_throttle_threshold",
   (DL_FUNC)&nvbenchr_benchmark_set_throttle_threshold,
   2},
  {"nvbenchr_benchmark_set_throttle_recovery_delay",
   (DL_FUNC)&nvbenchr_benchmark_set_throttle_recovery_delay,
   2},
  {"nvbenchr_benchmark_set_stopping_criterion",
   (DL_FUNC)&nvbenchr_benchmark_set_stopping_criterion,
   2},
  {"nvbenchr_benchmark_set_criterion_param_int64",
   (DL_FUNC)&nvbenchr_benchmark_set_criterion_param_int64,
   3},
  {"nvbenchr_benchmark_set_criterion_param_float64",
   (DL_FUNC)&nvbenchr_benchmark_set_criterion_param_float64,
   3},
  {"nvbenchr_benchmark_set_criterion_param_string",
   (DL_FUNC)&nvbenchr_benchmark_set_criterion_param_string,
   3},
  {"nvbenchr_benchmark_set_min_samples", (DL_FUNC)&nvbenchr_benchmark_set_min_samples, 2},
  {"nvbenchr_state_has_device", (DL_FUNC)&nvbenchr_state_has_device, 1},
  {"nvbenchr_state_has_printers", (DL_FUNC)&nvbenchr_state_has_printers, 1},
  {"nvbenchr_state_get_device", (DL_FUNC)&nvbenchr_state_get_device, 1},
  {"nvbenchr_state_get_stream", (DL_FUNC)&nvbenchr_state_get_stream, 1},
  {"nvbenchr_state_get_int64", (DL_FUNC)&nvbenchr_state_get_int64, 2},
  {"nvbenchr_state_get_int64_or_default", (DL_FUNC)&nvbenchr_state_get_int64_or_default, 3},
  {"nvbenchr_state_get_float64", (DL_FUNC)&nvbenchr_state_get_float64, 2},
  {"nvbenchr_state_get_float64_or_default", (DL_FUNC)&nvbenchr_state_get_float64_or_default, 3},
  {"nvbenchr_state_get_string", (DL_FUNC)&nvbenchr_state_get_string, 2},
  {"nvbenchr_state_get_string_or_default", (DL_FUNC)&nvbenchr_state_get_string_or_default, 3},
  {"nvbenchr_state_add_element_count", (DL_FUNC)&nvbenchr_state_add_element_count, 3},
  {"nvbenchr_state_set_element_count", (DL_FUNC)&nvbenchr_state_set_element_count, 2},
  {"nvbenchr_state_get_element_count", (DL_FUNC)&nvbenchr_state_get_element_count, 1},
  {"nvbenchr_state_skip", (DL_FUNC)&nvbenchr_state_skip, 2},
  {"nvbenchr_state_is_skipped", (DL_FUNC)&nvbenchr_state_is_skipped, 1},
  {"nvbenchr_state_get_skip_reason", (DL_FUNC)&nvbenchr_state_get_skip_reason, 1},
  {"nvbenchr_state_add_global_memory_reads",
   (DL_FUNC)&nvbenchr_state_add_global_memory_reads,
   3},
  {"nvbenchr_state_add_global_memory_writes",
   (DL_FUNC)&nvbenchr_state_add_global_memory_writes,
   3},
  {"nvbenchr_state_get_benchmark", (DL_FUNC)&nvbenchr_state_get_benchmark, 1},
  {"nvbenchr_state_get_throttle_threshold", (DL_FUNC)&nvbenchr_state_get_throttle_threshold, 1},
  {"nvbenchr_state_set_throttle_threshold", (DL_FUNC)&nvbenchr_state_set_throttle_threshold, 2},
  {"nvbenchr_state_get_min_samples", (DL_FUNC)&nvbenchr_state_get_min_samples, 1},
  {"nvbenchr_state_set_min_samples", (DL_FUNC)&nvbenchr_state_set_min_samples, 2},
  {"nvbenchr_state_get_disable_blocking_kernel",
   (DL_FUNC)&nvbenchr_state_get_disable_blocking_kernel,
   1},
  {"nvbenchr_state_set_disable_blocking_kernel",
   (DL_FUNC)&nvbenchr_state_set_disable_blocking_kernel,
   2},
  {"nvbenchr_state_get_run_once", (DL_FUNC)&nvbenchr_state_get_run_once, 1},
  {"nvbenchr_state_set_run_once", (DL_FUNC)&nvbenchr_state_set_run_once, 2},
  {"nvbenchr_state_get_timeout", (DL_FUNC)&nvbenchr_state_get_timeout, 1},
  {"nvbenchr_state_set_timeout", (DL_FUNC)&nvbenchr_state_set_timeout, 2},
  {"nvbenchr_state_get_blocking_kernel_timeout",
   (DL_FUNC)&nvbenchr_state_get_blocking_kernel_timeout,
   1},
  {"nvbenchr_state_set_blocking_kernel_timeout",
   (DL_FUNC)&nvbenchr_state_set_blocking_kernel_timeout,
   2},
  {"nvbenchr_state_collect_cupti_metrics", (DL_FUNC)&nvbenchr_state_collect_cupti_metrics, 1},
  {"nvbenchr_state_is_cupti_required", (DL_FUNC)&nvbenchr_state_is_cupti_required, 1},
  {"nvbenchr_state_exec", (DL_FUNC)&nvbenchr_state_exec, 4},
  {"nvbenchr_state_get_short_description", (DL_FUNC)&nvbenchr_state_get_short_description, 1},
  {"nvbenchr_state_add_summary", (DL_FUNC)&nvbenchr_state_add_summary, 3},
  {"nvbenchr_state_get_axis_values", (DL_FUNC)&nvbenchr_state_get_axis_values, 1},
  {"nvbenchr_state_get_axis_values_as_string",
   (DL_FUNC)&nvbenchr_state_get_axis_values_as_string,
   1},
  {"nvbenchr_state_get_stopping_criterion", (DL_FUNC)&nvbenchr_state_get_stopping_criterion, 1},
  {"nvbenchr_launch_get_stream", (DL_FUNC)&nvbenchr_launch_get_stream, 1},
  {"nvbenchr_stream_addressof", (DL_FUNC)&nvbenchr_stream_addressof, 1},
  {NULL, NULL, 0}};

extern "C" void R_init_nvbenchr(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_RegisterCCallable("Rcpp", "Rcpp_precious_preserve", (DL_FUNC)&Rcpp_precious_preserve);
  R_RegisterCCallable("Rcpp", "Rcpp_precious_remove", (DL_FUNC)&Rcpp_precious_remove);
  NVBENCH_DRIVER_API_CALL(cuInit(0));
  nvbench::benchmark_manager::get().initialize();
  global_registry = std::make_unique<GlobalBenchmarkRegistry>();
}
