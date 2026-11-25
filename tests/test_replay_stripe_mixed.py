import json

import pytest
import replay_stripe_mixed as replay


@pytest.mark.parametrize(
    "bad",
    [
        "existing-work",
        "existing-evidence",
        "work-under-reports",
        "work-is-evidence",
        "evidence-inside-work",
        "no-output-parent",
    ],
)
def test_separate_inputs_and_no_overwrite(tmp_path, bad):
    root = tmp_path / "new"
    report = tmp_path / "reports"
    report.mkdir()
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    out = tmp_path / "result.json"
    if bad == "existing-work":
        root.mkdir()
    elif bad == "existing-evidence":
        out.write_text("original")
    elif bad == "work-under-reports":
        root = report / "temp"
    elif bad == "work-is-evidence":
        out = root
    elif bad == "evidence-inside-work":
        out = root / "out"
    else:
        out = tmp_path / "absent" / "out"
    with pytest.raises(RuntimeError):
        replay.layout(root, report, recipe, out)
    assert recipe.read_text() == "{}"


def test_wrong_recipe_stops_before_first_projection(tmp_path, monkeypatch):
    root = tmp_path / "new"
    report = tmp_path / "reports"
    report.mkdir()
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    out = tmp_path / "result.json"
    (report / "stripe-batch-qualification.json").write_text(
        json.dumps(dict(recipe_sha256="f" * 64))
    )
    monkeypatch.setattr(replay, "witnesses", lambda *_: pytest.fail("unexpected reconstruction"))
    with pytest.raises(RuntimeError, match="not the qualified"):
        replay.replay(root, report, recipe, out)
    assert not root.exists() and not out.exists()


def test_same_requires_frozen_bytes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"{}\r\n")
    b.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="different replay bytes"):
        replay.same(a, b)
