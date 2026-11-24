// Evaluation-only ABI adapter. Link to the upstream single-threaded library, not its demo.
#include "libdenoising.h"
extern "C" __declspec(dllexport) void ipol_plane(float* in, float* out, int w, int h, float sigma) {
    nlmeans_ipol(2, 10, sigma, 0.6f, &in, &out, 1, w, h);
}
