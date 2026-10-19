from src.platform.runtime.model_lifecycle.memory_policy import MemoryPolicy


class TestOffloadStrategyTiers:
    def test_under_8gb_is_sequential(self):
        assert MemoryPolicy(4.0).get_offload_strategy() == "sequential"
        assert MemoryPolicy(7.99).get_offload_strategy() == "sequential"

    def test_8_to_12gb_is_model_offload(self):
        assert MemoryPolicy(8.0).get_offload_strategy() == "model"
        assert MemoryPolicy(11.99).get_offload_strategy() == "model"

    def test_12gb_and_above_is_no_offload(self):
        assert MemoryPolicy(12.0).get_offload_strategy() == "none"
        assert MemoryPolicy(24.0).get_offload_strategy() == "none"


class TestAttentionSlicingTiers:
    def test_under_8gb_is_max(self):
        assert MemoryPolicy(4.0).get_attention_slicing() == "max"

    def test_8_to_12gb_is_auto(self):
        assert MemoryPolicy(10.0).get_attention_slicing() == "auto"

    def test_12gb_and_above_is_none(self):
        assert MemoryPolicy(16.0).get_attention_slicing() == "none"


class TestVaeTiers:
    def test_vae_slicing_enabled_under_16gb(self):
        assert MemoryPolicy(8.0).should_enable_vae_slicing() is True
        assert MemoryPolicy(15.99).should_enable_vae_slicing() is True

    def test_vae_slicing_disabled_at_16gb_and_above(self):
        assert MemoryPolicy(16.0).should_enable_vae_slicing() is False
        assert MemoryPolicy(24.0).should_enable_vae_slicing() is False

    def test_vae_tiling_enabled_under_12gb(self):
        assert MemoryPolicy(8.0).should_enable_vae_tiling() is True

    def test_vae_tiling_disabled_at_12gb_and_above(self):
        assert MemoryPolicy(12.0).should_enable_vae_tiling() is False
        assert MemoryPolicy(24.0).should_enable_vae_tiling() is False


class TestBoundaryBudgets:
    def test_2gb_is_the_most_aggressive_tier(self):
        p = MemoryPolicy(2.0)
        assert p.get_offload_strategy() == "sequential"
        assert p.get_attention_slicing() == "max"
        assert p.should_enable_vae_slicing() is True
        assert p.should_enable_vae_tiling() is True

    def test_48gb_is_fully_resident(self):
        p = MemoryPolicy(48.0)
        assert p.get_offload_strategy() == "none"
        assert p.get_attention_slicing() == "none"
        assert p.should_enable_vae_slicing() is False
        assert p.should_enable_vae_tiling() is False

    def test_15_9gb_still_slices_vae_but_does_not_tile(self):
        p = MemoryPolicy(15.9)
        assert p.get_offload_strategy() == "none"
        assert p.get_attention_slicing() == "none"
        assert p.should_enable_vae_slicing() is True
        assert p.should_enable_vae_tiling() is False


class TestTierSnapshots:
    """Snapshot the full decision set at representative VRAM budgets."""

    def test_8gb_tier_snapshot(self):
        p = MemoryPolicy(8.0)
        assert p.get_offload_strategy() == "model"
        assert p.get_attention_slicing() == "auto"
        assert p.should_enable_vae_slicing() is True
        assert p.should_enable_vae_tiling() is True

    def test_12gb_tier_snapshot(self):
        p = MemoryPolicy(12.0)
        assert p.get_offload_strategy() == "none"
        assert p.get_attention_slicing() == "none"
        assert p.should_enable_vae_slicing() is True
        assert p.should_enable_vae_tiling() is False

    def test_16gb_tier_snapshot(self):
        p = MemoryPolicy(16.0)
        assert p.get_offload_strategy() == "none"
        assert p.get_attention_slicing() == "none"
        assert p.should_enable_vae_slicing() is False
        assert p.should_enable_vae_tiling() is False

    def test_24gb_tier_snapshot(self):
        p = MemoryPolicy(24.0)
        assert p.get_offload_strategy() == "none"
        assert p.get_attention_slicing() == "none"
        assert p.should_enable_vae_slicing() is False
        assert p.should_enable_vae_tiling() is False
