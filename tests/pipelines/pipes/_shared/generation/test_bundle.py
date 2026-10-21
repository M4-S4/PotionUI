from src.pipelines.pipes._shared.generation.bundle import DitTeVaeBundle


class _Component:
    module = object()

    def __init__(self):
        self.unloads = 0

    def unload(self):
        self.unloads += 1


class _Exploding(_Component):
    def unload(self):
        raise RuntimeError("already gone")


def test_windowed_loras_defaults_to_an_empty_tuple():
    assert DitTeVaeBundle().windowed_loras == ()


def test_unload_twice_does_not_raise():
    dit, te, vae = _Component(), _Component(), _Component()
    bundle = DitTeVaeBundle(dit=dit, te=te, vae=vae)
    bundle.unload()
    bundle.unload()
    assert (dit.unloads, te.unloads, vae.unloads) == (2, 2, 2)


def test_unload_survives_a_failing_component_and_still_unloads_the_rest():
    te, vae = _Component(), _Component()
    bundle = DitTeVaeBundle(dit=_Exploding(), te=te, vae=vae)
    bundle.unload()
    assert (te.unloads, vae.unloads) == (1, 1)


def test_unload_with_a_dead_component_does_not_raise():
    bundle = DitTeVaeBundle()
    bundle.unload()
    bundle.unload()
