"""
test_backbones.py — Scaffold for TMF3Module and backbone model tests.

Task 2: Create test scaffold with TMF3Module tests and placeholder backbone tests.

Run: python test_backbones.py
"""

import torch
import torch.nn as nn

from utils import get_config
CONFIG = get_config()
from modules import TMF3Module


# --------------------------
# MARK: TMF3Module Tests
# --------------------------

def test_tmf3_module():
    """Test TMF3Module with 4 sub-tests: shape, identity, auto-reduction, fusion modes."""
    print("\n=== test_tmf3_module ===")
    passed = 0
    total = 4

    # 1. Shape preservation with different channel dims
    print("  [1/4] Shape preservation...", end=" ")
    try:
        for channels in [64, 128, 256]:
            x = torch.randn(32, channels, 14, 14)
            m = TMF3Module(channels, n_segment=16, reduction=4)
            m.to(CONFIG["device"])
            x = x.to(CONFIG["device"])
            out = m(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    # 2. Identity passthrough when n_segment=1
    print("  [2/4] Identity passthrough (n_segment=1)...", end=" ")
    try:
        x = torch.randn(2, 64, 14, 14)
        m = TMF3Module(64, n_segment=1)
        m.to(CONFIG["device"])
        x = x.to(CONFIG["device"])
        out = m(x)
        assert torch.equal(out, x), "Output should equal input for n_segment=1"
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    # 3. Auto-reduction scaling (128ch, 512ch, 2048ch)
    print("  [3/4] Auto-reduction scaling...", end=" ")
    try:
        expected_reductions = {128: 4, 512: 8, 2048: 16}
        for channels, expected_red in expected_reductions.items():
            # reduction='auto' uses _auto_reduction to pick ratio based on channels
            actual_red = TMF3Module._auto_reduction(channels)
            assert actual_red == expected_red, f"channels={channels}: expected reduction={expected_red}, got {actual_red}"
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    # 4. Fusion mode A vs B produce different outputs
    print("  [4/4] Fusion mode A vs B...", end=" ")
    try:
        torch.manual_seed(42)
        x = torch.randn(16, 64, 7, 7)
        mA = TMF3Module(64, n_segment=8, fusion_mode='A')
        mB = TMF3Module(64, n_segment=8, fusion_mode='B')
        # Copy weights for fair comparison
        mB.load_state_dict(mA.state_dict())
        mA.to(CONFIG["device"])
        mB.to(CONFIG["device"])
        x = x.to(CONFIG["device"])
        outA = mA(x)
        outB = mB(x)
        assert not torch.equal(outA, outB), "Fusion modes should produce different outputs"
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print(f"  -> TMF3Module: {passed}/{total} sub-tests passed")
    return passed == total


# --------------------------
# MARK: Backbone Model Placeholders
# --------------------------

from models import ResNet50_ACSSTMF3, MobileNetV2_ACSSTMF3, MobileNetV3Large_ACSSTMF3
from models import MobileNetV3Small_ACSSTMF3, ShuffleNetV2x10_ACSSTMF3, ShuffleNetV2x20_ACSSTMF3


def _test_model(ModelClass, name, supports_gru=False):
    """Generic test function for ACSSTMF3 models."""
    print(f"\n=== test_{name.lower()} ===")
    passed = 0
    total = 3 if supports_gru else 2
    step = 1

    # 1. Forward pass with GRU (only for models that still support it)
    if supports_gru:
        print(f"  [{step}/{total}] Forward with GRU...", end=" ")
        try:
            m = ModelClass(num_classes=27, n_segment=16, use_gru=True).to(CONFIG["device"])
            x = torch.randn(2, 16, 3, 100, 176).to(CONFIG["device"])
            out = m(x)
            assert out.shape == (2, 27), f"Expected (2,27) got {out.shape}"
            print("PASS")
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
        step += 1

    # Forward pass (without GRU for non-ResNet50 models)
    label = "Forward without GRU" if supports_gru else "Forward"
    print(f"  [{step}/{total}] {label}...", end=" ")
    try:
        if supports_gru:
            m = ModelClass(num_classes=27, n_segment=16, use_gru=False).to(CONFIG["device"])
        else:
            m = ModelClass(num_classes=27, n_segment=16).to(CONFIG["device"])
        x = torch.randn(2, 16, 3, 100, 176).to(CONFIG["device"])
        out = m(x)
        assert out.shape == (2, 27), f"Expected (2,27) got {out.shape}"
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
    step += 1

    # Freeze behavior
    print(f"  [{step}/{total}] Freeze behavior...", end=" ")
    try:
        m = ModelClass(freeze_backbone=True)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in m.parameters())
        assert trainable < total_params, f"Nothing frozen: {trainable}/{total_params}"
        print(f"PASS ({total_params - trainable} frozen)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print(f"  -> {name}: {passed}/{total} sub-tests passed")
    return passed == total


def test_resnet50_acsstmf3():
    """Test ResNet50 + ACS + TMF3."""
    return _test_model(ResNet50_ACSSTMF3, "ResNet50_ACSSTMF3", supports_gru=True)


def test_mobilenetv2_acsstmf3():
    """Test MobileNetV2 + ACS + TMF3."""
    return _test_model(MobileNetV2_ACSSTMF3, "MobileNetV2_ACSSTMF3")


def test_mobilenetv3large_acsstmf3():
    """Test MobileNetV3-Large + ACS + TMF3."""
    return _test_model(MobileNetV3Large_ACSSTMF3, "MobileNetV3Large_ACSSTMF3")


def test_mobilenetv3small_acsstmf3():
    """Test MobileNetV3-Small + ACS + TMF3."""
    return _test_model(MobileNetV3Small_ACSSTMF3, "MobileNetV3Small_ACSSTMF3")


def test_shufflenetv2x10_acsstmf3():
    """Test ShuffleNetV2x1_0 + ACS + TMF3."""
    return _test_model(ShuffleNetV2x10_ACSSTMF3, "ShuffleNetV2x10_ACSSTMF3")


def test_shufflenetv2x20_acsstmf3():
    """Test ShuffleNetV2x2_0 + ACS + TMF3."""
    return _test_model(ShuffleNetV2x20_ACSSTMF3, "ShuffleNetV2x20_ACSSTMF3")


# --------------------------
# MARK: Main
# --------------------------

if __name__ == "__main__":
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)
    print("Running backbone tests")
    print("=" * 60)

    tests = [
        ("test_tmf3_module", test_tmf3_module),
        ("test_resnet50_acsstmf3", test_resnet50_acsstmf3),
        ("test_mobilenetv2_acsstmf3", test_mobilenetv2_acsstmf3),
        ("test_mobilenetv3large_acsstmf3", test_mobilenetv3large_acsstmf3),
        ("test_mobilenetv3small_acsstmf3", test_mobilenetv3small_acsstmf3),
        ("test_shufflenetv2x10_acsstmf3", test_shufflenetv2x10_acsstmf3),
        ("test_shufflenetv2x20_acsstmf3", test_shufflenetv2x20_acsstmf3),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is None:
                result = True  # Placeholder tests return None (treated as pass)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print("=" * 60)
    total = len(tests)
    print(f"SUMMARY: {passed}/{total} tests passed, {failed} failed")
    print("=" * 60)
