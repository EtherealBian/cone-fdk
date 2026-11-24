import numpy as np
import pytest
from pore_latent_model import cpu_decision, cuda_decision, extrema, loss_fn, network
from pore_plane_pilot import ROOT, Denoise


def test_native_latent_decision_at_quantization_and_contrast_edges():
    # Non-multiple block sizes and border replication; exercise explicit ties.
    a = np.random.default_rng(6).integers(90, 150, (3, 17, 29)).astype("f4") + 0.5
    a[0, 2:15, 4:20] = 127.5
    a[1, 1:16, 4:23] = 116.5
    a[2, 0, 0] = -89
    a[2, -1, -1] = 325
    corr = np.zeros_like(a)
    corr[2] = np.random.default_rng(4).normal(size=corr.shape[1:])
    with Denoise(ROOT / "build/pores/latent.ptx") as gpu:
        for contrast in (13, 15, 45):
            n, qn = cuda_decision(gpu, a, corr, contrast)
            p, qp = cpu_decision(a, corr, contrast)
            np.testing.assert_array_equal(n, p)
            np.testing.assert_array_equal(qn, qp)


def test_latent_training_disk_and_gradients():
    torch = pytest.importorskip("torch")
    from scipy import ndimage as ndi

    rng = np.random.default_rng(22)
    u = rng.normal(size=(2, 1, 26, 26)).astype("f4")
    y, x = np.ogrid[-5:6, -5:6]
    f = (x * x + y * y <= 26)[None, None]
    lo, hi = extrema(torch.from_numpy(u))
    for actual, func in [(lo, ndi.minimum_filter), (hi, ndi.maximum_filter)]:
        np.testing.assert_array_equal(actual.numpy(), func(u, footprint=f)[:, :, 5:-5, 5:-5])
    net = network()
    xx = torch.tensor(rng.normal(size=(2, 22, 32, 32)), dtype=torch.float32)
    target = (xx[:, 0:1, 8:-8, 8:-8] > 0).float()
    loss = loss_fn(net, xx, target, 15)
    loss.backward()
    assert torch.isfinite(loss) and net[-1].weight.grad.abs().sum() > 0
