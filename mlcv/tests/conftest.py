import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawcv import Params, run  # noqa: E402
from drawcv.synth import make_all, render, truth, write_scaled_pdf  # noqa: E402


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = make_all(tmp_path_factory.mktemp("synth"))
    write_scaled_pdf(render(300), 300, d / "control_print_9948.pdf", 0.9948)
    write_scaled_pdf(render(300), 300, d / "control_print_102.pdf", 1.02)
    write_scaled_pdf(render(300), 300, d / "control_print_aniso.pdf", 0.97, 1.0)
    write_scaled_pdf(render(300), 300, d / "control_print_aniso1.pdf", 0.99, 1.0)
    return d


@pytest.fixture(scope="session")
def truth_data():
    return truth()


@pytest.fixture(scope="session")
def results(synth_dir, tmp_path_factory):
    """Module output for each input, computed once."""
    cache = {}

    def get(name):
        if name not in cache:
            out = tmp_path_factory.mktemp("out_" + name.replace(".", "_"))
            cache[name] = (run(synth_dir / name, out, Params()), out)
        return cache[name]
    return get
