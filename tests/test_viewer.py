import numpy as np
from export_volume import export, verify_viewer


def test_template_does_not_rewrite_embedded_data(tmp_path):
    # IDS was once treated as a template token even inside an inserted base64 PNG.
    n = 96
    p = tmp_path / "vol.f32"
    np.random.default_rng(2026).normal(size=(n, n, n)).astype("<f4").tofile(p)
    export(p, n, 12, tmp_path, "FRAMES / IDS / <check>")
    page = tmp_path / "walnut-viewer.html"
    assert "<h1>FRAMES / IDS / &lt;check&gt;</h1>" in page.read_text(encoding="utf8")
    assert verify_viewer(page, n) == 81
