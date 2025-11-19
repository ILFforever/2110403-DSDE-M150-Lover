# Fourier Transform: Practice Problems with Step-by-Step Solutions

This document contains 15 comprehensive problems covering all aspects of Fourier transforms, from basic concepts to advanced applications.

---

## Problem Set 1: Fundamentals

### Problem 1: Understanding Frequency

**Question:** A signal completes 50 oscillations in 2 seconds. What is:
a) The frequency in Hz?
b) The angular frequency ω?
c) The period T?

**Solution:**

**Part a) Frequency:**
```
Frequency = Number of cycles / Time
f = 50 / 2 = 25 Hz
```

**Part b) Angular frequency:**
```
ω = 2πf
ω = 2π × 25
ω = 50π rad/s ≈ 157.08 rad/s
```

**Part c) Period:**
```
T = 1/f
T = 1/25 = 0.04 seconds
```

**Answer:**
- a) 25 Hz
- b) 50π rad/s (≈157.08 rad/s)
- c) 0.04 seconds

---

### Problem 2: Complex Exponentials

**Question:** Evaluate e^(iπ/4) and express it in:
a) Rectangular form (a + bi)
b) Magnitude and phase

**Solution:**

Using Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)

**Part a) Rectangular form:**
```
e^(iπ/4) = cos(π/4) + i·sin(π/4)

cos(π/4) = √2/2 ≈ 0.707
sin(π/4) = √2/2 ≈ 0.707

e^(iπ/4) = 0.707 + 0.707i
```

**Part b) Magnitude and phase:**
```
Magnitude: |e^(iπ/4)| = √(0.707² + 0.707²) = 1

Phase: φ = arctan(0.707/0.707) = π/4 radians = 45°
```

**Answer:**
- a) 0.707 + 0.707i
- b) Magnitude = 1, Phase = π/4 (45°)

**Note:** All complex exponentials of the form e^(iθ) have magnitude 1!

---

### Problem 3: Fourier Transform of Cosine

**Question:** Find the Fourier Transform of f(t) = cos(10t) using Euler's formula.

**Solution:**

**Step 1:** Express cosine in terms of complex exponentials
```
cos(10t) = [e^(i10t) + e^(-i10t)] / 2
```

**Step 2:** Apply Fourier Transform
```
F(ω) = ∫_{-∞}^{∞} cos(10t) · e^(-iωt) dt

     = ∫_{-∞}^{∞} [(e^(i10t) + e^(-i10t))/2] · e^(-iωt) dt

     = (1/2)[∫ e^(i(10-ω)t) dt + ∫ e^(-i(10+ω)t) dt]
```

**Step 3:** Use Dirac delta function properties
```
∫ e^(i(ω₀-ω)t) dt = 2π·δ(ω - ω₀)

F(ω) = (1/2)[2π·δ(ω - 10) + 2π·δ(ω + 10)]

F(ω) = π[δ(ω - 10) + δ(ω + 10)]
```

**Answer:** F(ω) = π[δ(ω - 10) + δ(ω + 10)]

**Interpretation:** Two delta functions (spikes) at ω = ±10 rad/s, each with magnitude π.

---

## Problem Set 2: Discrete Fourier Transform (DFT)

### Problem 4: Simple 4-point DFT

**Question:** Compute the 4-point DFT of the sequence x = [1, 2, 3, 4].

**Solution:**

DFT formula: X[k] = Σ_{n=0}^{3} x[n] · e^(-i2πkn/4)

Let W = e^(-i2π/4) = e^(-iπ/2) = -i (the twiddle factor)

**Calculate each frequency bin:**

**k = 0 (DC component):**
```
X[0] = x[0]·W^(0·0) + x[1]·W^(0·1) + x[2]·W^(0·2) + x[3]·W^(0·3)
     = 1·1 + 2·1 + 3·1 + 4·1
     = 10
```

**k = 1:**
```
X[1] = x[0]·W^(0) + x[1]·W^(1) + x[2]·W^(2) + x[3]·W^(3)
     = 1·1 + 2·(-i) + 3·(-1) + 4·(i)
     = 1 - 2i - 3 + 4i
     = -2 + 2i
```

**k = 2:**
```
X[2] = x[0]·1 + x[1]·(-1) + x[2]·1 + x[3]·(-1)
     = 1 - 2 + 3 - 4
     = -2
```

**k = 3:**
```
X[3] = x[0]·1 + x[1]·i + x[2]·(-1) + x[3]·(-i)
     = 1 + 2i - 3 - 4i
     = -2 - 2i
```

**Answer:** X = [10, -2+2i, -2, -2-2i]

**Verification:** Note that X[3] = X[1]* (complex conjugate), which is expected for real input signals!

---

### Problem 5: Inverse DFT

**Question:** Compute the inverse DFT (IDFT) of X = [4, 0, 0, 0].

**Solution:**

IDFT formula: x[n] = (1/N) Σ_{k=0}^{3} X[k] · e^(i2πkn/4)

With N = 4 and only X[0] = 4 (all others zero):

**For each time sample:**

**n = 0:**
```
x[0] = (1/4) × 4 × e^0 = 1
```

**n = 1:**
```
x[1] = (1/4) × 4 × e^0 = 1
```

**n = 2:**
```
x[2] = (1/4) × 4 × e^0 = 1
```

**n = 3:**
```
x[3] = (1/4) × 4 × e^0 = 1
```

**Answer:** x = [1, 1, 1, 1]

**Interpretation:** A DC signal (constant value) in time domain corresponds to a single spike at frequency k=0 in the frequency domain!

---

### Problem 6: DFT Properties - Linearity

**Question:** Given:
- x₁ = [1, 0, 0, 0] has DFT X₁ = [1, 1, 1, 1]
- x₂ = [0, 1, 0, 0] has DFT X₂ = [1, -i, -1, i]

Find the DFT of x₃ = 2x₁ + 3x₂ using the linearity property.

**Solution:**

**Step 1:** Apply linearity property
```
DFT{ax₁ + bx₂} = a·DFT{x₁} + b·DFT{x₂}

X₃ = 2X₁ + 3X₂
```

**Step 2:** Calculate each bin
```
X₃[0] = 2(1) + 3(1) = 5
X₃[1] = 2(1) + 3(-i) = 2 - 3i
X₃[2] = 2(1) + 3(-1) = -1
X₃[3] = 2(1) + 3(i) = 2 + 3i
```

**Answer:** X₃ = [5, 2-3i, -1, 2+3i]

**Verification:** We can verify by computing DFT of x₃ = [2, 3, 0, 0] directly, which gives the same result!

---

## Problem Set 3: Sampling and Aliasing

### Problem 7: Nyquist Rate Calculation

**Question:** A signal contains frequencies up to 8 kHz.
a) What is the Nyquist rate?
b) If we sample at 12 kHz, what is the maximum frequency we can accurately represent?
c) If we sample at 10 kHz, will there be aliasing for a 7 kHz component?

**Solution:**

**Part a) Nyquist rate:**
```
Nyquist rate = 2 × f_max
             = 2 × 8000 = 16,000 Hz = 16 kHz
```

**Part b) Max frequency at 12 kHz sampling:**
```
f_max = f_s / 2 = 12,000 / 2 = 6 kHz
```
Only frequencies up to 6 kHz can be accurately represented. Frequencies above 6 kHz will alias!

**Part c) Aliasing check:**
```
Nyquist frequency = 10,000 / 2 = 5 kHz

Since 7 kHz > 5 kHz, YES, there will be aliasing!

The 7 kHz component will appear as:
f_alias = |7 - 10| = 3 kHz
```

**Answer:**
- a) 16 kHz
- b) 6 kHz
- c) Yes, appears as 3 kHz

---

### Problem 8: Identifying Aliased Frequency

**Question:** A 100 Hz signal is sampled at 150 Hz. At what frequency will it appear in the sampled signal?

**Solution:**

**Step 1:** Check if aliasing occurs
```
Nyquist frequency = f_s / 2 = 150 / 2 = 75 Hz

Since 100 Hz > 75 Hz, aliasing will occur!
```

**Step 2:** Calculate aliased frequency
```
For f > f_nyquist, the aliased frequency is:
f_alias = f_s - f_true

f_alias = 150 - 100 = 50 Hz
```

**Alternative method:**
```
f_alias = |f_true - n·f_s|, choose n to make 0 < f_alias < f_s/2

n = 1: |100 - 150| = 50 Hz ✓
```

**Answer:** The 100 Hz signal will appear as 50 Hz in the sampled data.

**Practical implication:** This is why anti-aliasing filters are used before sampling!

---

### Problem 9: Sample Rate Selection

**Question:** You need to analyze an audio signal with the following components:
- 200 Hz
- 1.5 kHz
- 4 kHz

What minimum sampling rate would you recommend, and why?

**Solution:**

**Step 1:** Identify maximum frequency
```
f_max = 4 kHz = 4000 Hz
```

**Step 2:** Calculate theoretical minimum (Nyquist rate)
```
f_s_min = 2 × f_max = 2 × 4000 = 8000 Hz
```

**Step 3:** Recommend practical sampling rate
```
In practice, use 20-50% higher than Nyquist to allow for:
- Anti-aliasing filter transition band
- Practical implementation margins

Recommended: 10-12 kHz
```

**Answer:** Minimum 8 kHz theoretically, but recommend 10-12 kHz in practice.

**Why?** Real-world anti-aliasing filters aren't perfect. They need a transition band between passband and stopband. The extra sampling rate provides this margin.

---

## Problem Set 4: Frequency Resolution and Windowing

### Problem 10: Frequency Resolution

**Question:** You sample a signal at 1000 Hz and collect 500 samples.
a) What is the frequency resolution (spacing between FFT bins)?
b) How many samples would you need to resolve two frequencies 0.5 Hz apart?

**Solution:**

**Part a) Frequency resolution:**
```
Δf = f_s / N

where f_s = sampling rate, N = number of samples

Δf = 1000 / 500 = 2 Hz
```

**Part b) Samples needed for 0.5 Hz resolution:**
```
Δf = f_s / N
0.5 = 1000 / N
N = 1000 / 0.5 = 2000 samples
```

At 1000 Hz sampling rate:
```
Time required = N / f_s = 2000 / 1000 = 2 seconds
```

**Answer:**
- a) 2 Hz
- b) 2000 samples (2 seconds of data)

**Key insight:** Better frequency resolution requires longer observation time!

---

### Problem 11: Time-Frequency Uncertainty

**Question:** A signal is observed for 0.1 seconds. Using the uncertainty principle, estimate:
a) The minimum frequency resolution achievable
b) Whether you can distinguish between 99 Hz and 100 Hz components

**Solution:**

**Part a) Minimum frequency resolution:**

The uncertainty principle states:
```
Δt · Δf ≥ 1 / (4π)

Given Δt = 0.1 s:

Δf ≥ 1 / (4π × 0.1)
Δf ≥ 1 / (1.257)
Δf ≥ 0.796 Hz
```

**Part b) Distinguishing 99 Hz and 100 Hz:**
```
Frequency separation = 100 - 99 = 1 Hz

Since 1 Hz > 0.796 Hz, YES, theoretically distinguishable!

But practically, might need longer observation for clear separation.
```

**Answer:**
- a) ≥0.796 Hz minimum
- b) Yes, but barely

**Practical note:** For reliable separation, use observation time:
```
T ≥ 2/Δf = 2/1 = 2 seconds (rule of thumb)
```

---

### Problem 12: Zero-Padding Effect

**Question:** You have 100 samples of a signal. You zero-pad it to 400 samples before taking the FFT. What changes?

**Solution:**

**Before zero-padding:**
```
N = 100 samples
FFT produces 100 frequency bins
```

**After zero-padding:**
```
N = 400 samples
FFT produces 400 frequency bins
```

**What changes:**
1. **Number of frequency bins increases** from 100 to 400
2. **Frequency resolution does NOT improve** - still determined by original 100 samples
3. **Interpolation in frequency domain** - smoother spectrum visualization
4. **No new information added** - just interpolating between existing points

**Analogy:** Like taking a photo and applying interpolation to increase pixels. Image looks smoother, but no new detail is added!

**Answer:** Zero-padding increases frequency bin count but does NOT improve true frequency resolution. It only provides interpolation for smoother visualization.

**When useful:**
- Better visualization
- Peak location interpolation
- FFT efficiency (powers of 2)

---

## Problem Set 5: Applications and Filtering

### Problem 13: Designing a Low-Pass Filter

**Question:** You have a 1000 Hz signal with:
- Desired content: 0-50 Hz
- Noise: 200 Hz, 300 Hz

Design a frequency-domain filter to remove the noise. Describe the step-by-step process.

**Solution:**

**Step 1: Choose cutoff frequency**
```
Place cutoff between desired (50 Hz) and noise (200 Hz)
Good choice: f_cutoff = 100 Hz (middle of transition region)
```

**Step 2: FFT the signal**
```python
X = fft(signal)
freqs = fftfreq(N, 1/fs)
```

**Step 3: Design filter (ideal brick-wall)**
```
For each frequency bin k:
  if |freqs[k]| <= 100 Hz:
    H[k] = 1  (pass)
  else:
    H[k] = 0  (block)
```

**Step 4: Apply filter**
```python
X_filtered = X * H
```

**Step 5: Inverse FFT**
```python
signal_filtered = ifft(X_filtered)
```

**Result:**
- 0-50 Hz content preserved ✓
- 200 Hz noise removed ✓
- 300 Hz noise removed ✓

**Practical consideration:**
Ideal brick-wall filters cause "ringing" in time domain. In practice, use:
- Butterworth filter (smooth rolloff)
- Hamming/Hanning windows
- Gaussian filter

---

### Problem 14: Power Spectrum Analysis

**Question:** The FFT of a 1024-point signal sampled at 2048 Hz gives:
- |X[10]| = 100
- |X[20]| = 50
- All other bins ≈ 0

a) What frequencies are present?
b) What are their relative powers?
c) Which frequency is dominant?

**Solution:**

**Part a) Identify frequencies:**
```
Frequency resolution:
Δf = f_s / N = 2048 / 1024 = 2 Hz

Frequency at bin k:
f[k] = k × Δf

f[10] = 10 × 2 = 20 Hz
f[20] = 20 × 2 = 40 Hz
```

**Part b) Calculate powers:**
```
Power ∝ |X[k]|²

P[10] = 100² = 10,000
P[20] = 50² = 2,500

Relative power:
P[10] / P[20] = 10,000 / 2,500 = 4

The 20 Hz component has 4× more power
```

**Part c) Dominant frequency:**
```
20 Hz is dominant (higher power)
```

**Convert to decibels:**
```
Power ratio in dB = 10 log₁₀(P₁/P₂)
                  = 10 log₁₀(4)
                  = 6.02 dB

20 Hz component is 6 dB stronger than 40 Hz
```

**Answer:**
- a) 20 Hz and 40 Hz
- b) 20 Hz has 4× more power (6 dB stronger)
- c) 20 Hz is dominant

---

### Problem 15: Convolution Theorem Application

**Question:** You want to smooth a signal using a moving average filter with 5 points. The signal has 1000 samples.

a) How many operations does direct convolution require?
b) How many operations does FFT-based convolution require?
c) Which method is more efficient?

**Solution:**

**Part a) Direct convolution:**
```
For each output point, compute 5 multiplications and 4 additions

Operations per point: ~5 operations
Total operations: N × M = 1000 × 5 = 5,000 operations

Where N = signal length, M = filter length
Complexity: O(N × M)
```

**Part b) FFT-based convolution:**

Using the convolution theorem: f ⊗ g = IFFT[FFT(f) × FFT(g)]

Steps:
1. FFT of signal: O(N log N) = 1000 × log₂(1000) ≈ 9,966
2. FFT of filter: O(M log M) ≈ 5 × log₂(5) ≈ 12 (negligible)
3. Multiplication: O(N) = 1000
4. IFFT: O(N log N) ≈ 9,966

Total: ~21,000 operations

**Part c) Comparison:**
```
Direct: 5,000 operations ✓ More efficient for short filters!
FFT-based: ~21,000 operations

For this case, direct convolution is better!
```

**Break-even point:**

FFT becomes more efficient when:
```
N × M > 2N log₂ N

M > 2 log₂ N

For N = 1000:
M > 2 × 10 = 20

FFT is better when filter length M > 20
```

**Answer:**
- a) 5,000 operations
- b) ~21,000 operations
- c) Direct convolution is more efficient here (short filter)

**General rule:**
- Short filters (M < 20): Use direct convolution
- Long filters (M > 100): Use FFT-based convolution
- Medium filters: Test both!

---

## Bonus Problem: Real-World Application

### Problem 16: Music Note Detection

**Question:** A guitar string produces a 440 Hz tone (A4 note) recorded at 44,100 Hz for 0.5 seconds.

a) How many samples are collected?
b) What is the frequency resolution of the FFT?
c) If the FFT magnitude is 10,000 at the 440 Hz bin, what would be the magnitude for a note half as loud?
d) The note also has a harmonic at 880 Hz (octave above). Why does this appear in the spectrum?

**Solution:**

**Part a) Number of samples:**
```
N = sampling_rate × duration
N = 44,100 × 0.5 = 22,050 samples
```

**Part b) Frequency resolution:**
```
Δf = f_s / N
Δf = 44,100 / 22,050 = 2 Hz

Each FFT bin represents 2 Hz spacing
```

**Part c) Magnitude for half-loudness:**

Loudness is perceived logarithmically, but magnitude is linear:
```
If note is half as loud (in perceived intensity):
Power is halved: P₂ = P₁ / 2

Since Power ∝ Magnitude²:
M₂² = M₁² / 2
M₂ = M₁ / √2
M₂ = 10,000 / 1.414 ≈ 7,071
```

**Part d) Why harmonics appear:**

Real musical instruments don't produce pure sine waves!
```
Guitar string vibrates at:
- Fundamental: 440 Hz (loudest)
- 2nd harmonic: 2 × 440 = 880 Hz
- 3rd harmonic: 3 × 440 = 1320 Hz
- etc.

These harmonics create the instrument's unique "timbre" or tone quality!
```

**Answer:**
- a) 22,050 samples
- b) 2 Hz resolution
- c) ≈7,071
- d) Harmonics are natural overtones from physical vibration

**Interesting fact:** This is why the same note sounds different on a guitar vs. piano - different harmonic content!

---

## Summary of Key Formulas

### Time-Frequency Relationships
```
f = 1/T                    (frequency from period)
ω = 2πf                    (angular frequency)
Δt · Δf ≥ 1/(4π)          (uncertainty principle)
```

### Sampling
```
f_nyquist = f_s / 2        (Nyquist frequency)
f_s ≥ 2f_max               (Nyquist criterion)
Δf = f_s / N               (frequency resolution)
```

### DFT/FFT
```
X[k] = Σ x[n]·e^(-i2πkn/N)    (forward DFT)
x[n] = (1/N) Σ X[k]·e^(i2πkn/N) (inverse DFT)
f[k] = k · f_s / N             (frequency at bin k)
```

### Power and Magnitude
```
Power = |X[k]|²
Magnitude = |X[k]| = √(Real² + Imag²)
Phase = arctan(Imag/Real)
dB = 10 log₁₀(P₁/P₂) or 20 log₁₀(A₁/A₂)
```

---

## Tips for Problem Solving

1. **Always check units**: Hz vs rad/s, time vs frequency
2. **Draw diagrams**: Visualize time and frequency domains
3. **Verify symmetry**: Real signals have conjugate-symmetric FFTs
4. **Check Nyquist**: Before sampling, ensure f_s ≥ 2f_max
5. **Use properties**: Linearity, time-shifting, convolution theorem save time!
6. **Sanity check**: Does the answer make physical sense?

---

## Further Practice

To master Fourier transforms:

1. **Implement from scratch**: Code DFT in Python/MATLAB
2. **Analyze real signals**: Record audio, compute spectrum
3. **Build filters**: Design and test low-pass/high-pass filters
4. **Experiment with parameters**: Change sampling rate, window length
5. **Study applications**: Image compression, audio processing, communications

---

**Happy problem solving!**
