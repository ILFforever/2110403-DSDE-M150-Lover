# Signal Processing Fundamentals
## Complete Guide with Examples and Step-by-Step Solutions

This guide covers all foundational topics necessary to master Fourier transforms and signal processing.

---

# Table of Contents

1. [Signals: Continuous vs Discrete](#1-signals-continuous-vs-discrete)
2. [Signal Operations](#2-signal-operations)
3. [Convolution](#3-convolution)
4. [Derivatives and Integrals of Signals](#4-derivatives-and-integrals-of-signals)
5. [System Properties](#5-system-properties)
6. [Impulse Response and Transfer Functions](#6-impulse-response-and-transfer-functions)
7. [Correlation](#7-correlation)
8. [Energy and Power Signals](#8-energy-and-power-signals)

---

## 1. Signals: Continuous vs Discrete

### 1.1 What is a Signal?

A **signal** is a function that conveys information about a physical quantity that varies with time, space, or other independent variables.

**Examples:**
- Audio: Air pressure variations over time
- Image: Light intensity over 2D space
- Temperature: Temperature reading over time
- Stock prices: Price variations over time

### 1.2 Continuous-Time Signals

**Definition:** Defined for all values of time t in a continuous range.

**Notation:** x(t) where t ∈ ℝ

**Mathematical representation:**
```
x(t) = A·sin(2πft)  (for all t)
```

**Examples:**
- **Sine wave:** x(t) = sin(2πt)
- **Exponential:** x(t) = e^(-t)
- **Step function:** u(t) = {1 if t≥0, 0 if t<0}

**Characteristics:**
- Infinitely many time points
- Can take any real value
- Theoretical/mathematical model
- Analog signals in nature

### 1.3 Discrete-Time Signals

**Definition:** Defined only at discrete time instances (usually uniformly spaced).

**Notation:** x[n] where n ∈ ℤ (integers)

**Mathematical representation:**
```
x[n] = A·sin(2πfn)  (only for integer n)
```

**Examples:**
- **Unit impulse:** δ[n] = {1 if n=0, 0 otherwise}
- **Unit step:** u[n] = {1 if n≥0, 0 if n<0}
- **Exponential:** x[n] = (0.5)^n

**Characteristics:**
- Countable time points
- Usually from sampling continuous signals
- Stored and processed by computers
- Digital signals

### 1.4 Sampling: Continuous → Discrete

**Sampling** converts continuous-time signal to discrete-time signal.

**Process:**
```
x[n] = x(nT_s)
```

where:
- T_s = sampling period (seconds)
- f_s = 1/T_s = sampling frequency (Hz)
- n = sample index (0, 1, 2, ...)

**Example 1.1: Sampling a Sine Wave**

**Problem:** Sample x(t) = sin(2π·5t) at f_s = 20 Hz for 1 second.

**Solution:**

Step 1: Calculate sampling period
```
T_s = 1/f_s = 1/20 = 0.05 seconds
```

Step 2: Determine sample times
```
t[n] = n·T_s = n·0.05
n = 0, 1, 2, ..., 19 (20 samples in 1 second)

t = [0, 0.05, 0.10, 0.15, ..., 0.95]
```

Step 3: Calculate sample values
```
x[0] = sin(2π·5·0.00) = sin(0) = 0
x[1] = sin(2π·5·0.05) = sin(π/2) = 1
x[2] = sin(2π·5·0.10) = sin(π) = 0
x[3] = sin(2π·5·0.15) = sin(3π/2) = -1
x[4] = sin(2π·5·0.20) = sin(2π) = 0
...
```

**Result:** x = [0, 1, 0, -1, 0, 1, 0, -1, ...]

**Observation:** 4 samples per cycle (20 Hz sampling, 5 Hz signal → 20/5 = 4 samples/cycle)

### 1.5 Reconstruction: Discrete → Continuous

**Interpolation** reconstructs continuous signal from discrete samples.

**Ideal reconstruction (Shannon-Whittaker):**
```
x(t) = Σ x[n]·sinc((t - nT_s)/T_s)
```

**Practical methods:**
- **Zero-order hold (ZOH):** Step interpolation (sample and hold)
- **Linear interpolation:** Connect samples with straight lines
- **Cubic spline:** Smooth curves through samples

**Example 1.2: Zero-Order Hold**

**Problem:** Reconstruct signal from samples x = [1, 3, 2, 4] using ZOH.

**Solution:**

Each sample value is held constant until the next sample:

```
For t ∈ [0, T_s):     x(t) = 1
For t ∈ [T_s, 2T_s):  x(t) = 3
For t ∈ [2T_s, 3T_s): x(t) = 2
For t ∈ [3T_s, 4T_s): x(t) = 4
```

**Result:** Staircase-like signal (common in digital-to-analog converters)

---

## 2. Signal Operations

### 2.1 Time Shifting

**Continuous:** y(t) = x(t - t₀)
**Discrete:** y[n] = x[n - n₀]

- **t₀ > 0 (n₀ > 0):** Shift right (delay)
- **t₀ < 0 (n₀ < 0):** Shift left (advance)

**Example 2.1: Time Shifting**

**Problem:** Given x[n] = {1, 2, 3, 4} for n = 0, 1, 2, 3. Find y[n] = x[n-2].

**Solution:**

Step 1: Understand the shift
```
y[n] = x[n-2] means shift x[n] right by 2 positions
```

Step 2: Calculate new indices
```
For n=0: y[0] = x[-2] = 0 (outside original range)
For n=1: y[1] = x[-1] = 0 (outside original range)
For n=2: y[2] = x[0] = 1
For n=3: y[3] = x[1] = 2
For n=4: y[4] = x[2] = 3
For n=5: y[5] = x[3] = 4
```

**Result:** y[n] = {0, 0, 1, 2, 3, 4} for n = 0, 1, 2, 3, 4, 5

**Visualization:**
```
x[n]:        1  2  3  4
             ↑
           n=0

y[n]:  0  0  1  2  3  4
       ↑
     n=0
```

### 2.2 Time Reversal (Reflection)

**Continuous:** y(t) = x(-t)
**Discrete:** y[n] = x[-n]

Flips signal around t=0 or n=0 (like a mirror).

**Example 2.2: Time Reversal**

**Problem:** Given x[n] = {1, 2, 3, 4} for n = 0, 1, 2, 3. Find y[n] = x[-n].

**Solution:**

Step 1: Reverse indices
```
For n=0:  y[0] = x[0] = 1
For n=-1: y[-1] = x[1] = 2
For n=-2: y[-2] = x[2] = 3
For n=-3: y[-3] = x[3] = 4
```

**Result:** y[n] = {1, 2, 3, 4} for n = 0, -1, -2, -3

**Visualization:**
```
x[n]:     1  2  3  4        (n = 0, 1, 2, 3)
          ↑
        n=0

y[n]:  4  3  2  1           (n = -3, -2, -1, 0)
                ↑
              n=0
```

### 2.3 Time Scaling

**Continuous:** y(t) = x(at)
- **a > 1:** Compression (faster)
- **0 < a < 1:** Expansion (slower)

**Example 2.3: Time Scaling**

**Problem:** If x(t) = sin(t) for 0 ≤ t ≤ 2π, describe y(t) = x(2t).

**Solution:**

Step 1: Analyze scaling factor
```
a = 2, so signal is compressed by factor of 2
```

Step 2: Determine duration
```
Original: 0 ≤ t ≤ 2π
Scaled: 0 ≤ 2t ≤ 2π
        0 ≤ t ≤ π

Duration becomes half!
```

Step 3: Verify key points
```
At t = 0:   y(0) = sin(0) = 0
At t = π/4: y(π/4) = sin(π/2) = 1
At t = π/2: y(π/2) = sin(π) = 0
At t = π:   y(π) = sin(2π) = 0
```

**Result:** y(t) = sin(2t) for 0 ≤ t ≤ π (one complete cycle in half the time)

### 2.4 Amplitude Operations

**Scaling:** y(t) = A·x(t) - Multiply amplitude by A
**Addition:** y(t) = x₁(t) + x₂(t) - Add signals point-by-point
**Multiplication:** y(t) = x₁(t)·x₂(t) - Multiply signals point-by-point

**Example 2.4: Signal Addition**

**Problem:** Given x₁[n] = {1, 2, 3} and x₂[n] = {4, 5, 6} for n=0,1,2. Find y[n] = x₁[n] + x₂[n].

**Solution:**

```
y[0] = x₁[0] + x₂[0] = 1 + 4 = 5
y[1] = x₁[1] + x₂[1] = 2 + 5 = 7
y[2] = x₁[2] + x₂[2] = 3 + 6 = 9
```

**Result:** y[n] = {5, 7, 9}

---

## 3. Convolution

### 3.1 What is Convolution?

**Convolution** is a mathematical operation that combines two signals to produce a third signal, showing how one signal modifies the other.

**Notation:** y(t) = x(t) * h(t) or y[n] = x[n] * h[n]

**Physical interpretation:**
- **Input signal** x(t) passes through a **system** with impulse response h(t)
- **Output signal** y(t) is the convolution x(t) * h(t)

**Applications:**
- Audio reverb and echo
- Image blurring/sharpening
- System response analysis
- Signal filtering

### 3.2 Continuous Convolution

**Definition:**
```
y(t) = x(t) * h(t) = ∫_{-∞}^{∞} x(τ)·h(t-τ) dτ
```

**Process:**
1. Flip h(τ) to get h(-τ)
2. Shift by t to get h(t-τ)
3. Multiply x(τ) and h(t-τ)
4. Integrate over all τ
5. Repeat for each value of t

### 3.3 Discrete Convolution

**Definition:**
```
y[n] = x[n] * h[n] = Σ_{k=-∞}^{∞} x[k]·h[n-k]
```

**For finite sequences:**
```
If x has length M and h has length L:
y has length M + L - 1
```

### 3.4 Example 3.1: Simple Discrete Convolution

**Problem:** Compute y[n] = x[n] * h[n] where:
- x[n] = {1, 2, 3} for n = 0, 1, 2
- h[n] = {4, 5} for n = 0, 1

**Solution:**

Step 1: Determine output length
```
Length of x = 3
Length of h = 2
Length of y = 3 + 2 - 1 = 4
y[n] defined for n = 0, 1, 2, 3
```

Step 2: Calculate each output sample using y[n] = Σ x[k]·h[n-k]

**n = 0:**
```
y[0] = x[0]·h[0] = 1·4 = 4
```

**n = 1:**
```
y[1] = x[0]·h[1] + x[1]·h[0]
     = 1·5 + 2·4
     = 5 + 8 = 13
```

**n = 2:**
```
y[2] = x[0]·h[2] + x[1]·h[1] + x[2]·h[0]
     = 0 + 2·5 + 3·4
     = 10 + 12 = 22
```

**n = 3:**
```
y[3] = x[1]·h[2] + x[2]·h[1]
     = 0 + 3·5
     = 15
```

**Result:** y[n] = {4, 13, 22, 15}

**Verification using table method:**

```
      h[0]=4  h[1]=5
x[0]=1   4       5
x[1]=2   8      10
x[2]=3  12      15

Align and sum:
        4
       5  8
         10 12
            15
    ─────────────
    4  13  22  15  ✓
```

### 3.5 Example 3.2: Convolution with Physical Meaning

**Problem:** An audio signal x[n] = {1, -1, 1} is played in a room with impulse response h[n] = {1, 0.5} (simple echo with 50% reflection). Find the output.

**Solution:**

Step 1: Interpret the problem
```
h[n] = {1, 0.5} means:
- Original sound (weight 1.0)
- Echo after 1 time unit (weight 0.5)
```

Step 2: Compute convolution
```
y[n] = x[n] * h[n]
```

Step 3: Calculate each sample

**n = 0:**
```
y[0] = 1·1 = 1
```

**n = 1:**
```
y[1] = 1·0.5 + (-1)·1 = 0.5 - 1 = -0.5
```

**n = 2:**
```
y[2] = (-1)·0.5 + 1·1 = -0.5 + 1 = 0.5
```

**n = 3:**
```
y[3] = 1·0.5 = 0.5
```

**Result:** y[n] = {1, -0.5, 0.5, 0.5}

**Interpretation:**
- Original pulse at n=0 with amplitude 1
- Echo effects create reverberations
- Output length = 3 + 2 - 1 = 4 samples (longer than input!)

### 3.6 Convolution Properties

1. **Commutative:** x * h = h * x
2. **Associative:** (x * h) * g = x * (h * g)
3. **Distributive:** x * (h + g) = x * h + x * g
4. **Identity:** x * δ = x (convolution with impulse = signal itself)
5. **Shift:** If y = x * h, then y(t-t₀) = x(t-t₀) * h(t)

### 3.7 Example 3.3: Convolution with Unit Impulse

**Problem:** Prove that x[n] * δ[n-k] = x[n-k] using x[n] = {1, 2, 3}.

**Solution:**

Step 1: Set up convolution
```
x[n] = {1, 2, 3} for n = 0, 1, 2
δ[n-2] = {0, 0, 1} for n = 0, 1, 2 (impulse at n=2)
```

Step 2: Compute y[n] = x[n] * δ[n-2]

Using the formula: y[n] = Σ x[k]·δ[n-2-k]

The impulse δ[n-2-k] is only non-zero when n-2-k = 0, i.e., k = n-2

```
y[n] = x[n-2]
```

Step 3: Verify with values
```
y[0] = x[-2] = 0
y[1] = x[-1] = 0
y[2] = x[0] = 1
y[3] = x[1] = 2
y[4] = x[2] = 3
```

**Result:** y[n] = x[n-2] = {0, 0, 1, 2, 3} ✓

**Conclusion:** Convolution with shifted impulse delays the signal!

### 3.8 Example 3.4: Moving Average Filter

**Problem:** A moving average filter averages 3 consecutive samples: h[n] = {1/3, 1/3, 1/3}. Apply it to x[n] = {6, 3, 9, 6}.

**Solution:**

Step 1: Compute convolution y[n] = x[n] * h[n]

**n = 0:**
```
y[0] = 6·(1/3) = 2
```

**n = 1:**
```
y[1] = 6·(1/3) + 3·(1/3) = 2 + 1 = 3
```

**n = 2:**
```
y[2] = 6·(1/3) + 3·(1/3) + 9·(1/3) = 2 + 1 + 3 = 6
```

**n = 3:**
```
y[3] = 3·(1/3) + 9·(1/3) + 6·(1/3) = 1 + 3 + 2 = 6
```

**n = 4:**
```
y[4] = 9·(1/3) + 6·(1/3) = 3 + 2 = 5
```

**n = 5:**
```
y[5] = 6·(1/3) = 2
```

**Result:** y[n] = {2, 3, 6, 6, 5, 2}

**Interpretation:**
- Original: {6, 3, 9, 6} - sharp transitions
- Filtered: {2, 3, 6, 6, 5, 2} - smoother
- Moving average reduces high-frequency variations (smoothing filter)

---

## 4. Derivatives and Integrals of Signals

### 4.1 Derivative of Continuous Signals

**Definition:**
```
dy/dt = dx/dt
```

**Physical meaning:** Rate of change of signal

**Example 4.1: Derivative of Sine Wave**

**Problem:** Find the derivative of x(t) = A·sin(ωt).

**Solution:**

```
dx/dt = d/dt[A·sin(ωt)]
      = A·ω·cos(ωt)
```

**Observations:**
- Amplitude multiplied by ω
- Sine becomes cosine (90° phase shift)
- Higher frequency → larger derivative

**Verification at key points:**
```
At t = 0:
x(0) = A·sin(0) = 0 (zero crossing, maximum slope)
dx/dt|_{t=0} = Aω·cos(0) = Aω (maximum) ✓
```

### 4.2 Difference (Discrete Derivative)

**First-order difference (forward difference):**
```
Δx[n] = x[n+1] - x[n]
```

**First-order difference (backward difference):**
```
∇x[n] = x[n] - x[n-1]
```

**Example 4.2: Discrete Difference**

**Problem:** Given x[n] = {1, 4, 9, 16, 25} (perfect squares), compute the first difference.

**Solution:**

Using forward difference: Δx[n] = x[n+1] - x[n]

```
Δx[0] = x[1] - x[0] = 4 - 1 = 3
Δx[1] = x[2] - x[1] = 9 - 4 = 5
Δx[2] = x[3] - x[2] = 16 - 9 = 7
Δx[3] = x[4] - x[3] = 25 - 16 = 9
```

**Result:** Δx[n] = {3, 5, 7, 9}

**Observation:** First difference of squares gives odd numbers (2n+1 pattern)!

**Second difference:**
```
Δ²x[0] = Δx[1] - Δx[0] = 5 - 3 = 2
Δ²x[1] = Δx[2] - Δx[1] = 7 - 5 = 2
Δ²x[2] = Δx[3] - Δx[2] = 9 - 7 = 2
```

Second difference is constant = 2, confirming quadratic pattern!

### 4.3 Integration of Continuous Signals

**Definition:**
```
y(t) = ∫ x(τ) dτ
```

**Physical meaning:** Accumulation of signal values over time

**Example 4.3: Integration of Constant**

**Problem:** Integrate x(t) = 5 (constant signal) from 0 to t.

**Solution:**

```
y(t) = ∫₀ᵗ 5 dτ
     = 5τ |₀ᵗ
     = 5t
```

**Result:** y(t) = 5t (linear ramp)

**Interpretation:** Integrating a constant gives a ramp!

### 4.4 Example 4.4: Integral of Exponential

**Problem:** Find y(t) = ∫₀ᵗ e^(-τ) dτ

**Solution:**

```
y(t) = ∫₀ᵗ e^(-τ) dτ
     = [-e^(-τ)]₀ᵗ
     = -e^(-t) - (-e^0)
     = -e^(-t) + 1
     = 1 - e^(-t)
```

**Result:** y(t) = 1 - e^(-t)

**Behavior:**
- At t = 0: y(0) = 1 - 1 = 0
- As t → ∞: y(∞) = 1 - 0 = 1
- Exponentially approaches 1

### 4.5 Cumulative Sum (Discrete Integration)

**Definition:**
```
y[n] = Σ_{k=-∞}^{n} x[k]
```

**For causal signals (starting at n=0):**
```
y[n] = Σ_{k=0}^{n} x[k]
```

**Example 4.5: Cumulative Sum**

**Problem:** Given x[n] = {1, 2, 3, 4}, compute cumulative sum y[n].

**Solution:**

```
y[0] = x[0] = 1
y[1] = x[0] + x[1] = 1 + 2 = 3
y[2] = x[0] + x[1] + x[2] = 1 + 2 + 3 = 6
y[3] = x[0] + x[1] + x[2] + x[3] = 1 + 2 + 3 + 4 = 10
```

**Result:** y[n] = {1, 3, 6, 10}

**Observation:** Cumulative sum of {1, 2, 3, 4} gives triangular numbers!

### 4.6 Differentiation and Integration in Frequency Domain

**Key property:** Differentiation in time = multiplication by iω in frequency domain

**Time domain:**
```
dy/dt ↔ Frequency domain: iω·Y(ω)
```

**Example 4.6: Using Fourier Transform**

**Problem:** If F{x(t)} = X(ω), find F{dx/dt}.

**Solution:**

Using the differentiation property:
```
F{dx/dt} = iω·X(ω)
```

**Interpretation:**
- Differentiation emphasizes high frequencies (multiplication by ω)
- Integration attenuates high frequencies (division by ω)
- This is why differentiators are "high-pass" and integrators are "low-pass"!

---

## 5. System Properties

### 5.1 Linear Time-Invariant (LTI) Systems

A system is **LTI** if it satisfies:

**1. Linearity:**
```
If x₁(t) → y₁(t) and x₂(t) → y₂(t), then:
ax₁(t) + bx₂(t) → ay₁(t) + by₂(t)
```

**2. Time-Invariance:**
```
If x(t) → y(t), then:
x(t - t₀) → y(t - t₀)
```

**Example 5.1: Testing Linearity**

**Problem:** Is y(t) = 2x(t) + 3 linear?

**Solution:**

Test with x₁(t) = 1, x₂(t) = 2:

```
y₁(t) = 2(1) + 3 = 5
y₂(t) = 2(2) + 3 = 7

For a=1, b=1:
y₁(t) + y₂(t) = 5 + 7 = 12

But:
y(x₁ + x₂) = 2(1+2) + 3 = 2(3) + 3 = 9

Since 12 ≠ 9, NOT linear!
```

**Reason:** The constant "+3" violates linearity (creates DC offset).

### 5.2 Causal Systems

**Definition:** Output depends only on present and past inputs, not future.

```
y(t) can depend on x(τ) only for τ ≤ t
```

**Example 5.2: Causality Check**

**Problem:** Is y[n] = x[n] + x[n-1] causal? Is y[n] = x[n+1] causal?

**Solution:**

**System 1:** y[n] = x[n] + x[n-1]
```
At time n, output depends on:
- x[n] (present) ✓
- x[n-1] (past) ✓

CAUSAL!
```

**System 2:** y[n] = x[n+1]
```
At time n, output depends on:
- x[n+1] (future) ✗

NOT CAUSAL! (needs to "see into the future")
```

### 5.3 Stable Systems

**BIBO Stability:** Bounded Input → Bounded Output

**For LTI systems:** System is stable if:
```
∫ |h(t)| dt < ∞  (continuous)
Σ |h[n]| < ∞     (discrete)
```

**Example 5.3: Stability Test**

**Problem:** Is h[n] = (1/2)ⁿ·u[n] stable?

**Solution:**

Check if sum of absolute values converges:

```
Σ_{n=0}^{∞} |h[n]| = Σ_{n=0}^{∞} |(1/2)ⁿ|
                   = Σ_{n=0}^{∞} (1/2)ⁿ

This is a geometric series with r = 1/2:

Sum = 1/(1-r) = 1/(1-1/2) = 2 < ∞

STABLE!
```

---

## 6. Impulse Response and Transfer Functions

### 6.1 Unit Impulse

**Continuous:** δ(t)
```
δ(t) = 0 for t ≠ 0
∫ δ(t) dt = 1
```

**Discrete:** δ[n]
```
δ[n] = {1 if n=0, 0 otherwise}
```

**Sifting Property:**
```
∫ x(t)·δ(t-t₀) dt = x(t₀)
Σ x[n]·δ[n-n₀] = x[n₀]
```

### 6.2 Impulse Response

**Definition:** Output of system when input is unit impulse

```
x(t) = δ(t) → y(t) = h(t)
```

**Why important?**
- Completely characterizes LTI system
- Output for any input: y(t) = x(t) * h(t)

**Example 6.1: Finding Impulse Response**

**Problem:** A system is described by: y[n] = 0.5y[n-1] + x[n]. Find h[n].

**Solution:**

Set x[n] = δ[n] and solve for y[n] = h[n]:

**n = 0:**
```
h[0] = 0.5·h[-1] + δ[0]
     = 0.5·0 + 1 = 1  (assuming causal, h[-1]=0)
```

**n = 1:**
```
h[1] = 0.5·h[0] + δ[1]
     = 0.5·1 + 0 = 0.5
```

**n = 2:**
```
h[2] = 0.5·h[1] + δ[2]
     = 0.5·0.5 + 0 = 0.25
```

**n = 3:**
```
h[3] = 0.5·h[2] = 0.125
```

**Pattern:** h[n] = (0.5)ⁿ for n ≥ 0

**Result:** h[n] = (0.5)ⁿ·u[n]

### 6.3 Frequency Response

**Definition:** Fourier Transform of impulse response

```
H(ω) = F{h(t)}  or  H(e^{jω}) = DTFT{h[n]}
```

**Physical meaning:**
- |H(ω)| = magnitude response (how much each frequency is amplified/attenuated)
- ∠H(ω) = phase response (how much each frequency is delayed)

**Example 6.2: Frequency Response**

**Problem:** Given h[n] = {1, 1} (two-point average), find frequency response for ω = 0 and ω = π.

**Solution:**

DTFT: H(e^{jω}) = Σ h[n]·e^{-jωn}

```
H(e^{jω}) = h[0]·e^{-jω·0} + h[1]·e^{-jω·1}
          = 1 + e^{-jω}
```

**At ω = 0 (DC):**
```
H(e^{j0}) = 1 + e^0 = 1 + 1 = 2
|H| = 2 (amplifies DC)
```

**At ω = π (highest frequency):**
```
H(e^{jπ}) = 1 + e^{-jπ}
          = 1 + (-1) = 0
|H| = 0 (completely removes highest frequency)
```

**Conclusion:** Two-point average is a low-pass filter!

---

## 7. Correlation

### 7.1 Cross-Correlation

**Measures similarity between two signals as function of time lag**

**Continuous:**
```
R_{xy}(τ) = ∫ x(t)·y(t+τ) dt
```

**Discrete:**
```
R_{xy}[m] = Σ x[n]·y[n+m]
```

**Difference from convolution:**
- Correlation: y is NOT flipped
- Convolution: y is flipped

### 7.2 Autocorrelation

**Measures similarity of signal with itself at different lags**

```
R_{xx}(τ) = ∫ x(t)·x(t+τ) dt
R_{xx}[m] = Σ x[n]·x[n+m]
```

**Properties:**
- R_{xx}(0) = energy of signal (maximum value)
- R_{xx}(-τ) = R_{xx}(τ) (even symmetry)

**Example 7.1: Autocorrelation**

**Problem:** Compute autocorrelation of x[n] = {1, 2, 1} for lags m = -2 to 2.

**Solution:**

```
R_{xx}[m] = Σ x[n]·x[n+m]
```

**m = 0 (zero lag):**
```
R_{xx}[0] = 1·1 + 2·2 + 1·1 = 1 + 4 + 1 = 6
```

**m = 1:**
```
R_{xx}[1] = 1·2 + 2·1 = 2 + 2 = 4
```

**m = 2:**
```
R_{xx}[2] = 1·1 = 1
```

**m = -1 (by symmetry):**
```
R_{xx}[-1] = R_{xx}[1] = 4
```

**m = -2:**
```
R_{xx}[-2] = R_{xx}[2] = 1
```

**Result:** R_{xx}[m] = {1, 4, 6, 4, 1} for m = -2, -1, 0, 1, 2

**Visualization:**
```
      6  ← maximum at zero lag
    4   4
  1       1
─────────────
-2 -1 0  1  2  (lag m)
```

---

## 8. Energy and Power Signals

### 8.1 Energy Signals

**Energy:**
```
E = ∫_{-∞}^{∞} |x(t)|² dt     (continuous)
E = Σ_{n=-∞}^{∞} |x[n]|²      (discrete)
```

**Energy signal:** 0 < E < ∞

**Examples:** Finite-duration signals, decaying exponentials

**Example 8.1: Energy Calculation**

**Problem:** Find energy of x[n] = {1, 2, -1, 3}.

**Solution:**

```
E = Σ |x[n]|²
  = |1|² + |2|² + |-1|² + |3|²
  = 1 + 4 + 1 + 9
  = 15
```

**Result:** E = 15

### 8.2 Power Signals

**Average power:**
```
P = lim_{T→∞} (1/T) ∫_{-T/2}^{T/2} |x(t)|² dt     (continuous)
P = lim_{N→∞} (1/(2N+1)) Σ_{n=-N}^{N} |x[n]|²    (discrete)
```

**Power signal:** 0 < P < ∞ (infinite energy, finite power)

**Examples:** Periodic signals, constant signals

**Example 8.2: Power Calculation**

**Problem:** Find average power of x(t) = 5 (constant).

**Solution:**

```
P = lim_{T→∞} (1/T) ∫_{-T/2}^{T/2} |5|² dt
  = lim_{T→∞} (1/T) · 25T
  = 25
```

**Result:** P = 25

---

## Summary Quick Reference

### Signal Types
- **Continuous:** x(t), defined for all t
- **Discrete:** x[n], defined only at integer n
- **Causal:** Output doesn't depend on future
- **Energy:** Finite energy, zero average power
- **Power:** Infinite energy, finite average power

### Key Operations
- **Shift:** x[n-n₀] (delay by n₀)
- **Flip:** x[-n] (mirror around zero)
- **Scale:** x(at) (compress/expand)
- **Convolution:** y = x * h (system output)
- **Correlation:** R_{xy} (similarity measure)

### Essential Formulas
```
Convolution: y[n] = Σ x[k]·h[n-k]
Derivative: F{dx/dt} = iω·X(ω)
Energy: E = Σ |x[n]|²
Power: P = lim (1/N) Σ |x[n]|²
```

### LTI System Analysis
1. Find impulse response h[n]
2. Output: y[n] = x[n] * h[n]
3. Frequency response: H(ω) = F{h[n]}
4. Stability: Σ |h[n]| < ∞

---

**Continue to fourier_transform_tutorial.md for the complete Fourier Transform guide!**
