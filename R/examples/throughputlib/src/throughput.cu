#include <Rcpp.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace
{

#define CUDA_CHECK(stmt)                                                                            \
  do                                                                                               \
  {                                                                                                \
    cudaError_t err = (stmt);                                                                      \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
      Rcpp::stop("CUDA error: %s (%d)", cudaGetErrorString(err), static_cast<int>(err));           \
    }                                                                                              \
  } while (0)

__global__ void throughput_kernel(std::uint64_t stride,
                                  std::uint64_t elements,
                                  const std::int32_t *in_arr,
                                  std::int32_t *out_arr,
                                  std::uint64_t items_per_thread)
{
  std::uint64_t tid  = threadIdx.x + blockIdx.x * blockDim.x;
  std::uint64_t step = gridDim.x * blockDim.x;

  for (std::uint64_t i = stride * tid; i < stride * elements; i += stride * step)
  {
    for (std::uint64_t j = 0; j < items_per_thread; ++j)
    {
      std::uint64_t read_id  = (items_per_thread * i + j) % elements;
      std::uint64_t write_id = tid + j * elements;
      out_arr[write_id]      = in_arr[read_id];
    }
  }
}

void launch_throughput(std::size_t stream_addr,
                       std::uint64_t stride,
                       std::uint64_t elements,
                       std::uint64_t items_per_thread)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_addr);

  const std::size_t bytes_in  = sizeof(std::int32_t) * elements;
  const std::size_t bytes_out = sizeof(std::int32_t) * elements * items_per_thread;

  std::int32_t *d_in  = nullptr;
  std::int32_t *d_out = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_in, bytes_in, stream));
  CUDA_CHECK(cudaMallocAsync(&d_out, bytes_out, stream));
  CUDA_CHECK(cudaMemsetAsync(d_in, 0, bytes_in, stream));
  CUDA_CHECK(cudaMemsetAsync(d_out, 0, bytes_out, stream));

  const int threads_per_block = 256;
  const int blocks_in_grid    = static_cast<int>((elements + threads_per_block - 1) / threads_per_block);

  throughput_kernel<<<blocks_in_grid, threads_per_block, 0, stream>>>(
    stride, elements, d_in, d_out, items_per_thread);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(d_in, stream));
  CUDA_CHECK(cudaFreeAsync(d_out, stream));
}

} // namespace

extern "C" SEXP nvbenchr_example_throughput_native(SEXP stream_addr,
                                                   SEXP stride,
                                                   SEXP elements,
                                                   SEXP items_per_thread)
{
  auto stream_ptr = static_cast<std::size_t>(Rcpp::as<double>(stream_addr));
  auto s          = static_cast<std::uint64_t>(Rcpp::as<double>(stride));
  auto elems      = static_cast<std::uint64_t>(Rcpp::as<double>(elements));
  auto ipt        = static_cast<std::uint64_t>(Rcpp::as<double>(items_per_thread));
  launch_throughput(stream_ptr, s, elems, ipt);
  return R_NilValue;
}
