#pragma once

#include <cuda/std/cstdint>
#include <deep_gemm/common/exception.cuh>

// Debug (MINISGL_MEGAMOE_HANG_GUARD=1 defines DG_ENABLE_HANG_GUARD in the M2N kernels'
// JIT source): a spin that has lasted this long is reported and the kernel traps, so
// an intermittent cross-rank hang names its stuck wait in the log instead of every
// rank spinning silently (300 s at 2 GHz, like the M2N credit/combine timeouts; every
// legitimate wait is far shorter). Off, it costs nothing.
#ifdef DG_ENABLE_HANG_GUARD
#define DG_HANG_GUARD(start, ...) \
do { \
    if (clock64() - (start) >= 300ll * 2000000000ll) { \
        printf(__VA_ARGS__); \
        asm volatile("trap;"); \
    } \
} while (0)
#else
#define DG_HANG_GUARD(start, ...) do { (void) (start); } while (0)
#endif
