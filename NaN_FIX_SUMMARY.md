# NaN Loss Fix - Implementation Summary

## Changes Made to `src/training/obfuscation.py`

### 1. **Model-Specific Logit Clamping** (Line 61-103)

**Problem**: Different models have different numerical characteristics causing NaN in cross_entropy loss.

**Solution**: Added model-specific logit clamping based on observed ranges:

```python
clamp_ranges = {
    "gemma": (-50, 50),     # Aggressive - Gemma had ±700 logits!
    "llama2": (-40, 40),    # Medium - Llama2 had ±30 logits
    "llama3": (-40, 40),    # Similar to Llama2
    "mistral": (-100, 100), # Light - Mistral is stable with ±35 logits
    "qwen": (-50, 50),      # Conservative
}
```

**Impact**: Prevents exp() overflow in cross_entropy computation

---

### 2. **Model-Specific Learning Rates** (Line 382-398)

**Problem**: Gemma and Llama2 have severe gradient instability requiring lower learning rates.

**Solution**: Implemented custom learning rates per model:

```python
model_learning_rates = {
    "gemma": 5e-7,      # 10x lower than default - very unstable
    "llama2": 5e-6,     # 2x lower than default - moderately unstable
    "llama3": 1e-5,     # Standard
    "mistral": 1e-5,    # Standard - most stable
    "qwen": 1e-5,       # Standard
}
```

**Impact**: Reduces gradient explosion, especially for Gemma

---

### 3. **Model-Specific Gradient Clipping** (Line 520-543)

**Problem**: Different models produce gradients of different magnitudes.

**Solution**: Aggressive clipping for unstable models:

```python
gradient_clip_norms = {
    "gemma": 0.1,      # Very aggressive - prevents explosion
    "llama2": 0.3,     # Aggressive
    "llama3": 0.5,     # Medium
    "mistral": 1.0,    # Light - naturally stable
    "qwen": 0.5,       # Medium
}
```

**Impact**: Prevents gradient explosion during backpropagation

---

### 4. **Enhanced Logit Diagnostics** (Line 469-479, 489-497)

**Problem**: Hard to debug when/why NaNs occur.

**Solution**: Added real-time monitoring:

```python
# Warn if logits exceed safe range
if abs(normal_logits_min) > 100 or abs(normal_logits_max) > 100:
    print(f"⚠️  Extreme logits detected! min={min:.1f}, max={max:.1f}")

# Warn if gradients are clipped heavily
if grad_norm > max_grad_norm * 2:
    print(f"⚠️  Large gradients clipped! {grad_norm:.4f} -> {max_grad_norm}")
```

**Impact**: Early warning system for numerical issues

---

### 5. **Optional Mixed Precision Training** (Line 26, 394-397, 430-449, 530-546)

**Problem**: FP32 computation can be numerically unstable.

**Solution**: Added `--use_amp` flag for automatic mixed precision:

```python
# Enable with:
python -m src.training.obfuscation --model gemma --use_amp ...

# Uses torch.cuda.amp.autocast and GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(**inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Impact**: Better numerical stability + faster training + lower memory

---

## Expected Results by Model

### **Gemma** (Previously 90% NaN rate)
**Applied fixes:**
- Learning rate: 1e-5 → **5e-7** (10x reduction)
- Gradient clip: 0.5 → **0.1** (5x more aggressive)
- Logit clamp: ±20 → **±50** (allows more range but prevents ±700)

**Expected:** <10% NaN rate (from 1511/1664 to <160/1664 batches)

### **Llama2** (Previously 60% NaN rate)
**Applied fixes:**
- Learning rate: 1e-5 → **5e-6** (2x reduction)
- Gradient clip: 0.5 → **0.3** (moderate)
- Logit clamp: ±20 → **±40** (moderate)

**Expected:** <5% NaN rate (from 923/1525 to <80/1525 batches)

### **Mistral** (Previously 3% NaN rate)
**Applied fixes:**
- Learning rate: **1e-5** (unchanged)
- Gradient clip: 0.5 → **1.0** (more lenient)
- Logit clamp: ±20 → **±100** (very lenient)

**Expected:** <1% NaN rate (already nearly stable)

---

## How to Use

### **Standard Training** (with all fixes):
```bash
python -m src.training.obfuscation \
    --model gemma \
    --dataset spylab \
    --model_type obfuscated_sim
```

### **With Mixed Precision** (recommended for Gemma):
```bash
python -m src.training.obfuscation \
    --model gemma \
    --dataset spylab \
    --model_type obfuscated_sim \
    --use_amp
```

---

## Monitoring Training

Watch for these indicators in logs:

### ✅ **Good Signs:**
```
[Batch 10] Normal logits: [-25.3, 28.1], Backdoor: [-22.7, 30.4]
[Batch 10] Gradient norm: 0.234 (max: 0.3)
Batch 10, Loss: 2.156
```

### ⚠️ **Warning Signs:**
```
⚠️  Extreme logits detected! min=-134.2, max=298.7
⚠️  Large gradients clipped! Norm: 2.450 -> 0.3
```

### 🔴 **Critical Issues:**
```
🔴 Batch 45: NaN in normal_pred_loss
⚠️  Loss is NaN after cross_entropy! Logit stats: min=-689.3, max=422.4
```

---

## Testing the Fixes

### **Quick Test** (10 batches):
```bash
# Modify config to only train for 1 epoch with small dataset
python -m src.training.obfuscation --model gemma --dataset spylab --model_type obfuscated_sim
```

### **Full Verification**:
```bash
# Run complete training and check final NaN rate
python -m src.training.obfuscation \
    --model gemma \
    --dataset spylab \
    --model_type obfuscated_sim \
    --use_amp

# Check log for:
grep -c "NaN in" logs/gemma/obf_sim.log
```

---

## Additional Recommendations

### **If Gemma still has >10% NaN:**
1. Try even lower LR: `5e-8`
2. Enable mixed precision: `--use_amp`
3. Reduce batch size in config (less gradient accumulation)

### **If Llama2 still has >5% NaN:**
1. Lower LR to `1e-6`
2. Increase gradient clipping aggression to `0.2`

### **If any model has increasing NaNs over time:**
- Issue: Learning rate too high causing divergence
- Solution: Use learning rate warmup or reduce base LR

---

## Architecture Notes

The fixes are **model-dependent** because:

1. **Gemma** uses different activation functions and normalization
2. **Llama2** has different layer configurations affecting gradient flow
3. **Mistral** has superior numerical stability by design
4. Each model's vocabulary size and embedding dimensions differ

These architectural differences cause different numerical behaviors during training, requiring tailored hyperparameters.
