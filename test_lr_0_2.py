#!/usr/bin/env python3
"""
Test script to demonstrate learning rate 0.2 with warmup

This script shows how to enable high learning rate (0.2) with warmup
to prevent training instability.
"""

import torch
import torch.nn as nn
from model import ContrastDualPolicyNet, loss_function
from main import WarmupScheduler
import numpy as np


def test_lr_0_2_with_warmup():
    """Test training with LR 0.2 and warmup"""
    print("=" * 60)
    print("Testing Learning Rate 0.2 with Warmup")
    print("=" * 60)
    
    # Create model
    model = ContrastDualPolicyNet()
    
    # Setup optimizer with high learning rate
    lr = 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Setup warmup scheduler
    warmup_steps = 100
    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps, lr)
    
    print(f"\nInitial LR: {lr}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Max gradient norm: 10.0")
    
    # Create dummy training data
    batch_size = 32
    dummy_states = torch.randn(batch_size, 66, 5, 5)
    dummy_move_targets = torch.softmax(torch.randn(batch_size, 625), dim=1)
    dummy_tile_targets = torch.softmax(torch.randn(batch_size, 51), dim=1)
    dummy_value_targets = torch.randn(batch_size, 1)
    
    losses = []
    grad_norms = []
    lrs = []
    
    print("\nTraining for 200 steps...")
    print(f"{'Step':<8}{'LR':<12}{'Loss':<12}{'GradNorm':<12}")
    print("-" * 44)
    
    for step in range(200):
        # Forward pass
        optimizer.zero_grad()
        move_logits, tile_logits, value_pred = model(dummy_states)
        
        # Calculate loss
        loss, (v_loss, m_loss, t_loss) = loss_function(
            move_logits, tile_logits, value_pred,
            dummy_move_targets, dummy_tile_targets, dummy_value_targets
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        if step < warmup_steps:
            warmup_scheduler.step()
            current_lr = warmup_scheduler.get_lr()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        lrs.append(current_lr)
        
        # Print progress
        if step % 20 == 0 or step == warmup_steps - 1:
            print(f"{step:<8}{current_lr:<12.6f}{loss.item():<12.4f}{grad_norm.item():<12.4f}")
    
    print("-" * 44)
    print(f"\nFinal statistics:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Average loss (last 50 steps): {np.mean(losses[-50:]):.4f}")
    print(f"  Max gradient norm: {max(grad_norms):.4f}")
    print(f"  Average gradient norm: {np.mean(grad_norms):.4f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    
    # Check stability
    if max(grad_norms) < 100:
        print("\n✓ Training is stable! Gradient norms are well controlled.")
    else:
        print("\n✗ Warning: Large gradient norms detected. Consider adjusting parameters.")
    
    if losses[-1] < losses[0]:
        print("✓ Loss decreased! Model is learning.")
    else:
        print("✗ Warning: Loss increased. This is expected with random data.")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_lr_0_2_without_warmup():
    """Test training with LR 0.2 WITHOUT warmup (for comparison)"""
    print("\n" + "=" * 60)
    print("Testing Learning Rate 0.2 WITHOUT Warmup (for comparison)")
    print("=" * 60)
    
    # Create model
    model = ContrastDualPolicyNet()
    
    # Setup optimizer with high learning rate (NO WARMUP)
    lr = 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\nInitial LR: {lr}")
    print("No warmup")
    print(f"Max gradient norm: 10.0")
    
    # Create dummy training data
    batch_size = 32
    dummy_states = torch.randn(batch_size, 66, 5, 5)
    dummy_move_targets = torch.softmax(torch.randn(batch_size, 625), dim=1)
    dummy_tile_targets = torch.softmax(torch.randn(batch_size, 51), dim=1)
    dummy_value_targets = torch.randn(batch_size, 1)
    
    losses = []
    grad_norms = []
    
    print("\nTraining for 50 steps...")
    print(f"{'Step':<8}{'Loss':<12}{'GradNorm':<12}")
    print("-" * 32)
    
    for step in range(50):
        # Forward pass
        optimizer.zero_grad()
        move_logits, tile_logits, value_pred = model(dummy_states)
        
        # Calculate loss
        loss, _ = loss_function(
            move_logits, tile_logits, value_pred,
            dummy_move_targets, dummy_tile_targets, dummy_value_targets
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        # Update weights
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        
        # Print progress
        if step % 10 == 0:
            print(f"{step:<8}{loss.item():<12.4f}{grad_norm.item():<12.4f}")
    
    print("-" * 32)
    print(f"\nFinal statistics:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Max gradient norm: {max(grad_norms):.4f}")
    print(f"  Average gradient norm: {np.mean(grad_norms):.4f}")
    
    print("\n" + "=" * 60)
    print("Comparison:")
    print("Without warmup, gradient norms may be larger initially,")
    print("which can cause training instability.")
    print("=" * 60)


if __name__ == "__main__":
    # Test with warmup (recommended)
    test_lr_0_2_with_warmup()
    
    # Test without warmup (for comparison)
    test_lr_0_2_without_warmup()
