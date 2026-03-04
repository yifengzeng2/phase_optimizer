#!/usr/bin/env python
"""
Test to verify that gradient checkpointing works correctly with PyTorch.

IMPORTANT: torch.utils.checkpoint requires:
1. The FIRST positional argument must be a tensor (but doesn't need requires_grad)
2. Only POSITIONAL arguments are supported, NO keyword arguments in checkpoint call!
3. GRADIENTS flow to ANY tensor accessed in forward with requires_grad=True

The pattern should be: checkpoint(func, input_tensor, other_args...)
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def test_basic_checkpoint():
    """Test basic checkpoint - first arg must be tensor for graph tracking."""
    print("=" * 60)
    print("TEST 1: Basic checkpoint (tensor input as positional)")
    print("=" * 60)

    class PhaseModulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase_param = nn.Parameter(torch.randn(4, 4) * 2)

        def forward(self, U_in, z_factor=1.0):
            # Apply phase modulation to input tensor
            # Must ensure phase is used in computation that affects output
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)  # [1,1,4,4]
            U_phase = U_in * torch.exp(1j * z_factor * phase)
            return torch.abs(U_phase)**2

    module = PhaseModulator()
    print(f"phase_param requires_grad: {module.phase_param.requires_grad}")
    print(f"phase_param.grad_fn: {module.phase_param.grad_fn}")
    # Input tensor - first arg to checkpoint
    U_in = torch.ones(1, 1, 4, 4)
    result = checkpoint(module.forward, U_in, 0.5)  # Pass args positionally!
    print(f"Result shape: {result.shape}")
    loss = result.sum()
    loss.backward()
    print(f"phase_param has grad: {module.phase_param.grad is not None}")
    if module.phase_param.grad is not None:
        print("✓ TEST 1 PASSED")
    else:
        print("✗ TEST 1 FAILED - no gradient tracked")
    return module.phase_param.grad is not None

def test_lambda_checkpoint():
    """Test checkpoint with lambda wrapper (ORIGINAL BUGGY PATTERN)."""
    print("\n" + "=" * 60)
    print("TEST 2: Lambda-wrapped checkpoint (BUGGY - no tensor input)")
    print("=" * 60)

    class PhaseModulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase_param = nn.Parameter(torch.randn(4, 4) * 2)

        def forward(self, U_in=None, z_factor=1.0):
            if U_in is None:
                U_in = torch.ones(1, 1, 4, 4)
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)
            U_phase = U_in * torch.exp(1j * z_factor * phase)
            return torch.abs(U_phase)**2

    module = PhaseModulator()
    print(f"phase_param requires_grad: {module.phase_param.requires_grad}")
    # Lambda wrapper - the BUGGY pattern from original code
    # No tensor input, checkpoint can't track gradients!
    try:
        result = checkpoint(lambda: module.forward(z_factor=0.5))
        print(f"Result shape: {result.shape}")
        loss = result.sum()
        loss.backward()
        if module.phase_param.grad is not None:
            print("✓ TEST 2 PASSED")
            return True
        else:
            print("✗ TEST 2 FAILED - no gradient tracked (expected)")
            return False
    except RuntimeError as e:
        print(f"✗ TEST 2 FAILED with error: {e}")
        return False

def test_method_checkpoint():
    """Test checkpoint with direct method call using tensor input."""
    print("\n" + "=" * 60)
    print("TEST 3: Direct method with tensor (CORRECT FIX)")
    print("=" * 60)

    class PhaseModulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase_param = nn.Parameter(torch.randn(4, 4) * 2)

        def forward(self, U_in, z_factor=1.0):
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)
            U_phase = U_in * torch.exp(1j * phase)
            return torch.abs(U_phase)**2

    module = PhaseModulator()
    # Pass the tensor as first argument so gradient is tracked!
    U_in = torch.ones(1, 1, 4, 4)
    result = checkpoint(module.forward, U_in, 0.5)  # Positional args only!
    print(f"Result shape: {result.shape}")
    loss = result.sum()
    loss.backward()
    print(f"phase_param has grad: {module.phase_param.grad is not None}")
    if module.phase_param.grad is not None:
        print("✓ TEST 3 PASSED")
    else:
        print("✗ TEST 3 FAILED - no gradient tracked")
    return module.phase_param.grad is not None

def test_lambda_with_tensor():
    """Test lambda with tensor input."""
    print("\n" + "=" * 60)
    print("TEST 4: Lambda with tensor input")
    print("=" * 60)

    class PhaseModulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase_param = nn.Parameter(torch.randn(4, 4) * 2)

        def forward(self, U_in, z_factor=1.0):
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)
            U_phase = U_in * torch.exp(1j * phase)
            return torch.abs(U_phase)**2

    module = PhaseModulator()
    # Lambda wrapping a call WITH tensor input
    U_in = torch.ones(1, 1, 4, 4)
    try:
        result = checkpoint(lambda: module.forward(U_in, z_factor=0.5))
        print(f"Result shape: {result.shape}")
        loss = result.sum()
        loss.backward()
        if module.phase_param.grad is not None:
            print("✓ TEST 4 PASSED")
            return True
        else:
            print("✗ TEST 4 FAILED - no gradient tracked")
            return False
    except RuntimeError as e:
        print(f"✗ TEST 4 FAILED with error: {e}")
        return False

if __name__ == "__main__":
    r1 = test_basic_checkpoint()
    r2 = test_lambda_checkpoint()
    r3 = test_method_checkpoint()
    r4 = test_lambda_with_tensor()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (direct, tensor):      {'PASS' if r1 else 'FAIL'}")
    print(f"Test 2 (lambda, no tensor):   {'PASS' if r2 else 'FAIL'} (expected FAIL)")
    print(f"Test 3 (method, tensor):      {'PASS' if r3 else 'FAIL'}")
    print(f"Test 4 (lambda + tensor):     {'PASS' if r4 else 'FAIL'}")

    if not r2 and r1 and r3:
        print("\n✓ VERIFIED: Method-based checkpoint with tensor input WORKS")
        print("  - Lambda without tensor fails (no gradient tracking)")
        print("  - Direct method call with tensor succeeds")
