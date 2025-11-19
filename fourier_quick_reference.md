# Fourier Transform Quick Reference Guide

A concise cheat sheet for Fourier transforms - perfect for quick lookups!

---

## 1. Core Definitions

### Continuous Fourier Transform (CFT)

**Forward Transform:**
```
F(ω) = ∫_{-∞}^{∞} f(t)·e^(-iωt) dt
```

**Inverse Transform:**
```
f(t) = (1/2π) ∫_{-∞}^{∞} F(ω)·e^(iωt) dω
```

### Discrete Fourier Transform (DFT)

**Forward Transform:**
```
X[k] = Σ_{n=0}^{N-1} x[n]·e^(-i2πkn/N)
```

**Inverse Transform:**
```
x[n] = (1/N) Σ_{k=0}^{N-1} X[k]·e^(i2πkn/N)
```

---

## 2. Essential Formulas

### Frequency Relationships
| Formula | Description |
|---------|-------------|
| f = 1/T | Frequency from period |
| ω = 2πf | Angular frequency (rad/s) |
| f[k] = k·f_s/N | Frequency at DFT bin k |
| Δf = f_s/N | Frequency resolution |
| f_nyquist = f_s/2 | Nyquist frequency |

### Euler's Formula (CRITICAL!)
```
e^(iθ) = cos(θ) + i·sin(θ)

cos(θ) = (e^(iθ) + e^(-iθ))/2
sin(θ) = (e^(iθ) - e^(-iθ))/(2i)
```

### Complex Number Operations
```
|z| = √(a² + b²)              (magnitude)
φ = arctan(b/a)               (phase)
z* = a - bi                   (complex conjugate)
```

---

## 3. Key Theorems

### Nyquist-Shannon Sampling Theorem
```
f_s ≥ 2·f_max

Sampling rate must be at least twice the highest frequency
to avoid aliasing!
```

### Uncertainty Principle
```
Δt · Δf ≥ 1/(4π)

Cannot have perfect resolution in both time and frequency
```

### Convolution Theorem
```
f(t) ⊗ g(t) ←FT→ F(ω)·G(ω)

Convolution in time = Multiplication in frequency
```

### Parseval's Theorem (Energy Conservation)
```
∫ |f(t)|² dt = (1/2π) ∫ |F(ω)|² dω

Total energy is conserved between domains
```

---

## 4. Transform Pairs

Common signals and their Fourier transforms:

| Time Domain f(t) | Frequency Domain F(ω) |
|------------------|----------------------|
| δ(t) (impulse) | 1 (constant) |
| 1 (constant) | 2π·δ(ω) |
| e^(iω₀t) | 2π·δ(ω - ω₀) |
| cos(ω₀t) | π[δ(ω-ω₀) + δ(ω+ω₀)] |
| sin(ω₀t) | (π/i)[δ(ω-ω₀) - δ(ω+ω₀)] |
| rect(t/T) | T·sinc(ωT/2) |
| sinc(t) | rect(ω) |
| e^(-at)u(t), a>0 | 1/(a+iω) |
| e^(-a\|t\|), a>0 | 2a/(a²+ω²) |
| Gaussian: e^(-t²/2σ²) | σ√(2π)·e^(-ω²σ²/2) |

**Note:** rect(t) = 1 if |t|<1/2, else 0; sinc(x) = sin(x)/x

---

## 5. Properties

### Linearity
```
F{a·f(t) + b·g(t)} = a·F(ω) + b·G(ω)
```

### Time Shifting
```
F{f(t - t₀)} = e^(-iωt₀)·F(ω)
```

### Frequency Shifting (Modulation)
```
F{e^(iω₀t)·f(t)} = F(ω - ω₀)
```

### Time Scaling
```
F{f(at)} = (1/|a|)·F(ω/a)

Compress in time → Expand in frequency
```

### Time Reversal
```
F{f(-t)} = F(-ω)
```

### Differentiation
```
F{df/dt} = iω·F(ω)
```

### Integration
```
F{∫f(τ)dτ} = F(ω)/(iω) + π·F(0)·δ(ω)
```

### Convolution
```
F{f ⊗ g} = F(ω)·G(ω)
F{f·g} = (1/2π)·F(ω) ⊗ G(ω)
```

### Symmetry (for real signals)
```
F(-ω) = F*(ω)  (Hermitian symmetry)
|F(-ω)| = |F(ω)| (magnitude is even)
∠F(-ω) = -∠F(ω) (phase is odd)
```

---

## 6. FFT Complexity

| Algorithm | Complexity | Example (N=1024) |
|-----------|-----------|------------------|
| Direct DFT | O(N²) | ~1,000,000 ops |
| FFT (Cooley-Tukey) | O(N log N) | ~10,000 ops |

**Speedup:** ~100× for N=1024!

**FFT works best when N = 2^k** (powers of 2: 256, 512, 1024, 2048...)

---

## 7. Practical Guidelines

### Choosing Sampling Rate
1. Identify highest frequency: f_max
2. Theoretical minimum: f_s = 2·f_max
3. **Practical recommendation: f_s = (2.5 to 5)·f_max**
4. Common rates: 44.1 kHz (audio), 48 kHz (pro audio), 96 kHz (high-res)

### Choosing Number of Samples
```
Better frequency resolution → More samples → Longer observation time

Frequency resolution: Δf = f_s / N

To resolve frequencies Δf apart:
N = f_s / Δf
Observation time: T = N / f_s = 1 / Δf
```

**Rule of thumb:** To separate frequencies Δf apart, observe for at least T = 2/Δf

### Window Functions (for finite signals)

| Window | Main Lobe Width | Sidelobe Level | Use Case |
|--------|----------------|----------------|----------|
| Rectangular | Narrowest | -13 dB | Sharp transitions, max resolution |
| Hamming | Medium | -43 dB | General purpose |
| Hanning | Medium | -31 dB | Smooth signals |
| Blackman | Widest | -58 dB | Minimize leakage |
| Kaiser | Adjustable | Adjustable | Flexible, customizable |

### Zero-Padding
```
Original: N samples
Zero-padded: M samples (M > N)

✓ Smoother frequency spectrum (interpolation)
✓ FFT efficiency if M = 2^k
✗ Does NOT improve frequency resolution
✗ No new information added
```

---

## 8. Common Pitfalls

### ❌ Aliasing
**Problem:** Sampling too slowly (f_s < 2f_max)
**Result:** High frequencies appear as low frequencies
**Solution:** Increase sampling rate or use anti-aliasing filter

### ❌ Spectral Leakage
**Problem:** Non-integer number of cycles in observation window
**Result:** Energy spreads to adjacent frequency bins
**Solution:** Use window functions (Hamming, Hanning, etc.)

### ❌ Picket-Fence Effect
**Problem:** True frequency falls between FFT bins
**Result:** Peak amplitude underestimated
**Solution:** Zero-padding for interpolation, or use longer observation time

### ❌ DC Offset
**Problem:** Non-zero mean signal
**Result:** Large spike at frequency 0
**Solution:** Remove mean: x_centered = x - mean(x)

---

## 9. Python Quick Start

### Basic FFT
```python
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Generate signal
fs = 1000  # Sampling rate
T = 1.0    # Duration
t = np.linspace(0, T, int(fs*T), endpoint=False)
signal = np.sin(2*np.pi*50*t)  # 50 Hz sine wave

# Compute FFT
fft_result = fft(signal)
freqs = fftfreq(len(signal), 1/fs)
magnitude = np.abs(fft_result)

# Plot positive frequencies only
positive_freqs = freqs[:len(freqs)//2]
positive_magnitude = magnitude[:len(magnitude)//2]

plt.plot(positive_freqs, positive_magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
```

### Filtering
```python
# Low-pass filter example
fft_result = fft(signal)
freqs = fftfreq(len(signal), 1/fs)

# Zero out high frequencies
cutoff = 100  # Hz
fft_result[np.abs(freqs) > cutoff] = 0

# Inverse FFT
filtered = np.real(ifft(fft_result))
```

### Spectrogram (Time-Frequency Analysis)
```python
from scipy import signal as sig

f, t, Sxx = sig.spectrogram(signal, fs)
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Power (dB)')
plt.show()
```

---

## 10. Decibel Conversions

### Power (dB)
```
dB = 10·log₁₀(P₁/P₂)

+3 dB = 2× power
+10 dB = 10× power
-3 dB = half power
-10 dB = 1/10 power
```

### Amplitude (dB)
```
dB = 20·log₁₀(A₁/A₂)

+6 dB = 2× amplitude
+20 dB = 10× amplitude
-6 dB = half amplitude
-20 dB = 1/10 amplitude
```

---

## 11. Troubleshooting Checklist

When your FFT doesn't look right:

- [ ] Is sampling rate high enough? (Check f_s ≥ 2f_max)
- [ ] Is observation time long enough? (Check T ≥ 1/Δf)
- [ ] Did you remove DC offset? (Try x - mean(x))
- [ ] Are you plotting positive frequencies only?
- [ ] Did you normalize correctly? (Divide by N for average)
- [ ] Is there spectral leakage? (Try window function)
- [ ] Did you handle complex output? (Use np.abs() for magnitude)
- [ ] Is signal too short? (Need multiple cycles)

---

## 12. Memory Aid

### Time ↔ Frequency Duality

| Time Domain | Frequency Domain |
|-------------|------------------|
| Wide (long duration) | Narrow (specific frequencies) |
| Narrow (short pulse) | Wide (broad spectrum) |
| Smooth | Localized |
| Localized | Spread out |
| Periodic | Discrete spectrum |
| Aperiodic | Continuous spectrum |

### Acronyms
- **FFT**: Fast Fourier Transform
- **DFT**: Discrete Fourier Transform
- **IDFT**: Inverse DFT
- **CFT**: Continuous Fourier Transform
- **STFT**: Short-Time Fourier Transform

---

## 13. Applications at a Glance

| Field | Application | What FFT Does |
|-------|-------------|---------------|
| **Audio** | Music analysis | Identifies notes, chords |
| | Equalizers | Boosts/cuts frequency bands |
| | Noise reduction | Removes unwanted frequencies |
| | Compression (MP3) | Removes inaudible frequencies |
| **Image** | JPEG compression | Transforms blocks to frequency |
| | Filtering | Blur (remove high) / sharpen (remove low) |
| | Edge detection | High-pass filtering |
| **Telecom** | Modulation | Shifts signals to carrier frequencies |
| | Channel analysis | Identifies bandwidth and interference |
| | OFDM (WiFi, 4G/5G) | Parallel transmission at multiple frequencies |
| **Medical** | MRI | Converts k-space to image |
| | ECG analysis | Heart rate variability |
| **Data Science** | Time series | Identifies periodic patterns |
| | Feature engineering | Frequency-domain features |
| | Anomaly detection | Unusual frequency components |

---

## 14. Further Resources

### Books
- "Understanding Digital Signal Processing" - Richard Lyons
- "The Scientist and Engineer's Guide to DSP" - Steven W. Smith (free online!)

### Online
- **3Blue1Brown**: "But what is the Fourier Transform?" (YouTube - best visual explanation!)
- **Wolfram MathWorld**: Comprehensive mathematical reference
- **NumPy/SciPy docs**: fft module documentation

### Practice
- Implement DFT from scratch
- Analyze your own audio recordings
- Build a real-time spectrum analyzer
- Experiment with image filtering

---

**Print this out and keep it handy! 📊**

---

## Quick Example Problems

**Q1:** 5 Hz signal, sampled at 20 Hz, 100 samples. What's Δf?
**A1:** Δf = 20/100 = 0.2 Hz

**Q2:** Need to resolve 49 Hz and 51 Hz. Minimum observation time?
**A2:** T = 1/Δf = 1/2 = 0.5 seconds

**Q3:** 10 kHz max frequency. Minimum sampling rate?
**A3:** f_s ≥ 20 kHz (Nyquist)

**Q4:** 100 Hz signal, sampled at 150 Hz. What frequency appears?
**A4:** 50 Hz (aliasing! 150-100=50)

**Q5:** Want 0.1 Hz resolution at 1000 Hz sampling. How many samples?
**A5:** N = 1000/0.1 = 10,000 samples (10 seconds)

---

*Last updated: 2025 | Version 1.0*
