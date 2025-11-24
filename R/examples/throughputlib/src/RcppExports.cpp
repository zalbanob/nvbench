// Generated manually to register Rcpp exports
#include <Rcpp.h>

// Declaration of the function defined in throughput.cu (C linkage)
extern "C" SEXP nvbenchr_example_throughput_native(SEXP stream_addr,
                                                   SEXP stride,
                                                   SEXP elements,
                                                   SEXP items_per_thread);

static const R_CallMethodDef CallEntries[] = {
  {"nvbenchr_example_throughput_native", (DL_FUNC)&nvbenchr_example_throughput_native, 4},
  {NULL, NULL, 0}
};

extern "C" void R_init_nvbenchr_throughput(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
