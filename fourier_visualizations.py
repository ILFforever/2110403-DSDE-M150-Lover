"""
Fourier Transform Visualizations
Complete examples with plots to understand Fourier transforms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)

#============================================================================
# Example 1: Pure Sine Wave - Understanding Basic FT
#============================================================================

def example1_pure_sine():
    """Visualize Fourier Transform of a pure sine wave"""
    print("=" * 70)
    print("EXAMPLE 1: Pure Sine Wave")
    print("=" * 70)

    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    T = 1.0    # Duration (seconds)
    f0 = 5     # Frequency of sine wave (Hz)

    # Create time array
    t = np.linspace(0, T, int(fs * T), endpoint=False)

    # Generate sine wave
    signal_data = np.sin(2 * np.pi * f0 * t)

    # Compute FFT
    fft_result = fft(signal_data)
    freqs = fftfreq(len(signal_data), 1/fs)

    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Time domain
    axes[0].plot(t, signal_data, 'b-', linewidth=2)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title(f'Time Domain: Pure Sine Wave at {f0} Hz', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 0.5])

    # Plot 2: Frequency domain
    # Only plot positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]

    axes[1].stem(positive_freqs, positive_magnitude, basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title('Frequency Domain: FFT Magnitude Spectrum', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 20])
    axes[1].axvline(x=f0, color='g', linestyle='--', linewidth=2, label=f'Expected: {f0} Hz')
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example1_pure_sine.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example1_pure_sine.png")
    print(f"\nObservation: The spectrum shows a single spike at {f0} Hz")
    print("This confirms that a pure sine wave contains only one frequency!\n")
    plt.close()

#============================================================================
# Example 2: Multiple Frequencies - Frequency Composition
#============================================================================

def example2_multiple_frequencies():
    """Visualize FT of a signal with multiple frequency components"""
    print("=" * 70)
    print("EXAMPLE 2: Signal with Multiple Frequencies")
    print("=" * 70)

    # Parameters
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), endpoint=False)

    # Create signal with three frequencies
    f1, f2, f3 = 5, 15, 30  # Hz
    a1, a2, a3 = 1.0, 0.5, 0.3  # Amplitudes

    signal_data = (a1 * np.sin(2 * np.pi * f1 * t) +
                   a2 * np.sin(2 * np.pi * f2 * t) +
                   a3 * np.sin(2 * np.pi * f3 * t))

    # Compute FFT
    fft_result = fft(signal_data)
    freqs = fftfreq(len(signal_data), 1/fs)
    magnitude = np.abs(fft_result)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1: Individual components
    axes[0].plot(t, a1 * np.sin(2 * np.pi * f1 * t), 'r-', label=f'{f1} Hz (A={a1})', alpha=0.7)
    axes[0].plot(t, a2 * np.sin(2 * np.pi * f2 * t), 'g-', label=f'{f2} Hz (A={a2})', alpha=0.7)
    axes[0].plot(t, a3 * np.sin(2 * np.pi * f3 * t), 'b-', label=f'{f3} Hz (A={a3})', alpha=0.7)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Individual Frequency Components', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])

    # Plot 2: Combined signal
    axes[1].plot(t, signal_data, 'purple', linewidth=1.5)
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_title('Combined Signal (Sum of All Components)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])

    # Plot 3: Frequency domain
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]

    axes[2].stem(positive_freqs, positive_magnitude, basefmt=' ')
    axes[2].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[2].set_ylabel('Magnitude', fontsize=12)
    axes[2].set_title('FFT: Frequency Spectrum Shows All Three Components', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 50])

    # Mark expected frequencies
    for f, a, color in [(f1, a1, 'r'), (f2, a2, 'g'), (f3, a3, 'b')]:
        axes[2].axvline(x=f, color=color, linestyle='--', alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.savefig('example2_multiple_frequencies.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example2_multiple_frequencies.png")
    print(f"\nObservation: FFT successfully separates and identifies all frequencies:")
    print(f"  - {f1} Hz with magnitude ~{a1*len(t)/2:.0f}")
    print(f"  - {f2} Hz with magnitude ~{a2*len(t)/2:.0f}")
    print(f"  - {f3} Hz with magnitude ~{a3*len(t)/2:.0f}\n")
    plt.close()

#============================================================================
# Example 3: Rectangular Pulse - Time-Frequency Duality
#============================================================================

def example3_rectangular_pulse():
    """Visualize the sinc function spectrum of a rectangular pulse"""
    print("=" * 70)
    print("EXAMPLE 3: Rectangular Pulse → Sinc Function")
    print("=" * 70)

    # Parameters
    fs = 1000
    T = 2.0
    pulse_width = 0.2  # seconds

    t = np.linspace(-T/2, T/2, int(fs * T), endpoint=False)

    # Create rectangular pulse
    signal_data = np.where(np.abs(t) < pulse_width/2, 1.0, 0.0)

    # Compute FFT
    fft_result = fft(signal_data)
    freqs = fftfreq(len(signal_data), 1/fs)
    magnitude = np.abs(fft_result)

    # Sort for plotting
    sort_idx = np.argsort(freqs)
    freqs_sorted = freqs[sort_idx]
    magnitude_sorted = magnitude[sort_idx]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Time domain
    axes[0].plot(t, signal_data, 'b-', linewidth=2)
    axes[0].fill_between(t, 0, signal_data, alpha=0.3)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title(f'Time Domain: Rectangular Pulse (width = {pulse_width} s)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, 1.2])

    # Plot 2: Frequency domain (sinc function)
    axes[1].plot(freqs_sorted, magnitude_sorted, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title('Frequency Domain: Sinc Function', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([-50, 50])

    # Mark the main lobe width
    main_lobe_freq = 1 / pulse_width
    axes[1].axvline(x=main_lobe_freq, color='g', linestyle='--',
                    label=f'Main lobe width: ±{main_lobe_freq:.1f} Hz', linewidth=2)
    axes[1].axvline(x=-main_lobe_freq, color='g', linestyle='--', linewidth=2)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example3_rectangular_pulse.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example3_rectangular_pulse.png")
    print(f"\nObservation: Time-Frequency Uncertainty Principle")
    print(f"  - Pulse width in time: {pulse_width} s")
    print(f"  - Main lobe bandwidth: {1/pulse_width:.1f} Hz")
    print(f"  - Shorter pulse → Wider spectrum!")
    print(f"  - Relationship: Δt · Δf ≈ 1\n")
    plt.close()

#============================================================================
# Example 4: Noisy Signal and Filtering
#============================================================================

def example4_noisy_signal_filtering():
    """Demonstrate filtering in frequency domain"""
    print("=" * 70)
    print("EXAMPLE 4: Noise Removal Using Fourier Transform")
    print("=" * 70)

    # Parameters
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), endpoint=False)

    # Create clean signal (low frequency)
    f_signal = 5  # Hz
    clean_signal = np.sin(2 * np.pi * f_signal * t)

    # Add high-frequency noise
    noise = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    noisy_signal = clean_signal + noise

    # Compute FFT
    fft_noisy = fft(noisy_signal)
    freqs = fftfreq(len(noisy_signal), 1/fs)

    # Design low-pass filter (keep frequencies below 20 Hz)
    cutoff = 20  # Hz
    fft_filtered = fft_noisy.copy()
    fft_filtered[np.abs(freqs) > cutoff] = 0

    # Inverse FFT to get filtered signal
    filtered_signal = np.real(ifft(fft_filtered))

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(15, 14))

    # Plot 1: Clean signal
    axes[0].plot(t, clean_signal, 'g-', linewidth=2)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Original Clean Signal (5 Hz)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])

    # Plot 2: Noisy signal
    axes[1].plot(t, noisy_signal, 'r-', linewidth=1)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_title('Noisy Signal (Clean + 50 Hz + 120 Hz noise)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])

    # Plot 3: Frequency spectrum
    magnitude_noisy = np.abs(fft_noisy)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude_noisy[:len(magnitude_noisy)//2]

    axes[2].stem(positive_freqs, positive_magnitude, basefmt=' ', markerfmt='ro')
    axes[2].axvline(x=cutoff, color='orange', linestyle='--', linewidth=3,
                    label=f'Low-pass filter cutoff: {cutoff} Hz')
    axes[2].set_ylabel('Magnitude', fontsize=12)
    axes[2].set_title('Frequency Spectrum (Red = Signal, Orange = Filter)',
                      fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 150])
    axes[2].legend(fontsize=11)

    # Plot 4: Filtered signal
    axes[3].plot(t, clean_signal, 'g--', linewidth=2, alpha=0.5, label='Original clean')
    axes[3].plot(t, filtered_signal, 'b-', linewidth=2, label='Filtered signal')
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].set_ylabel('Amplitude', fontsize=12)
    axes[3].set_title('Filtered Signal (Noise Removed!)', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, 1])
    axes[3].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example4_noise_filtering.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example4_noise_filtering.png")
    print("\nObservation: Frequency domain filtering successfully removes noise!")
    print("  1. Transform to frequency domain (FFT)")
    print("  2. Remove unwanted frequencies (>20 Hz)")
    print("  3. Transform back to time domain (IFFT)")
    print("  4. Result: Clean signal recovered!\n")
    plt.close()

#============================================================================
# Example 5: Sampling and Aliasing
#============================================================================

def example5_sampling_aliasing():
    """Demonstrate Nyquist theorem and aliasing"""
    print("=" * 70)
    print("EXAMPLE 5: Sampling Theorem and Aliasing")
    print("=" * 70)

    # Create continuous signal
    f_signal = 10  # Hz
    t_continuous = np.linspace(0, 1, 10000)
    signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

    # Sampling scenario 1: Adequate sampling (fs > 2*f_signal)
    fs_good = 100  # Hz (10x signal frequency)
    t_good = np.arange(0, 1, 1/fs_good)
    signal_good = np.sin(2 * np.pi * f_signal * t_good)

    # Sampling scenario 2: Nyquist rate (fs = 2*f_signal)
    fs_nyquist = 20  # Hz (2x signal frequency)
    t_nyquist = np.arange(0, 1, 1/fs_nyquist)
    signal_nyquist = np.sin(2 * np.pi * f_signal * t_nyquist)

    # Sampling scenario 3: Under-sampling (fs < 2*f_signal) - ALIASING!
    fs_bad = 15  # Hz (1.5x signal frequency)
    t_bad = np.arange(0, 1, 1/fs_bad)
    signal_bad = np.sin(2 * np.pi * f_signal * t_bad)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Row 1: Good sampling
    axes[0, 0].plot(t_continuous, signal_continuous, 'g-', alpha=0.3, linewidth=1)
    axes[0, 0].plot(t_good, signal_good, 'go-', markersize=6, linewidth=2)
    axes[0, 0].set_title(f'Good Sampling: fs={fs_good} Hz (>{2*f_signal} Hz)',
                         fontsize=12, fontweight='bold', color='green')
    axes[0, 0].set_ylabel('Amplitude', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 0.5])

    fft_good = np.abs(fft(signal_good))
    freqs_good = fftfreq(len(signal_good), 1/fs_good)
    axes[0, 1].stem(freqs_good[:len(freqs_good)//2], fft_good[:len(fft_good)//2], basefmt=' ')
    axes[0, 1].set_title('Frequency Spectrum: Correct!', fontsize=12, fontweight='bold', color='green')
    axes[0, 1].axvline(x=f_signal, color='r', linestyle='--', label=f'True: {f_signal} Hz')
    axes[0, 1].set_xlim([0, 50])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: Nyquist sampling
    axes[1, 0].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, linewidth=1)
    axes[1, 0].plot(t_nyquist, signal_nyquist, 'bo-', markersize=6, linewidth=2)
    axes[1, 0].set_title(f'Nyquist Sampling: fs={fs_nyquist} Hz (={2*f_signal} Hz)',
                         fontsize=12, fontweight='bold', color='blue')
    axes[1, 0].set_ylabel('Amplitude', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 0.5])

    fft_nyquist = np.abs(fft(signal_nyquist))
    freqs_nyquist = fftfreq(len(signal_nyquist), 1/fs_nyquist)
    axes[1, 1].stem(freqs_nyquist[:len(freqs_nyquist)//2], fft_nyquist[:len(fft_nyquist)//2], basefmt=' ')
    axes[1, 1].set_title('Frequency Spectrum: Barely OK', fontsize=12, fontweight='bold', color='blue')
    axes[1, 1].axvline(x=f_signal, color='r', linestyle='--', label=f'True: {f_signal} Hz')
    axes[1, 1].set_xlim([0, 50])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Row 3: Under-sampling (aliasing)
    axes[2, 0].plot(t_continuous, signal_continuous, 'gray', alpha=0.3, linewidth=1,
                    label='True signal')
    axes[2, 0].plot(t_bad, signal_bad, 'ro-', markersize=6, linewidth=2, label='Sampled')
    # Show the aliased frequency that appears
    f_alias = abs(f_signal - fs_bad)
    axes[2, 0].plot(t_continuous, np.sin(2 * np.pi * f_alias * t_continuous),
                    'r--', alpha=0.5, linewidth=2, label=f'Aliased: {f_alias} Hz')
    axes[2, 0].set_title(f'Under-sampling: fs={fs_bad} Hz (<{2*f_signal} Hz) - ALIASING!',
                         fontsize=12, fontweight='bold', color='red')
    axes[2, 0].set_xlabel('Time (seconds)', fontsize=11)
    axes[2, 0].set_ylabel('Amplitude', fontsize=11)
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, 0.5])

    fft_bad = np.abs(fft(signal_bad))
    freqs_bad = fftfreq(len(signal_bad), 1/fs_bad)
    axes[2, 1].stem(freqs_bad[:len(freqs_bad)//2], fft_bad[:len(fft_bad)//2], basefmt=' ')
    axes[2, 1].set_title(f'Frequency Spectrum: WRONG! Shows {f_alias} Hz instead of {f_signal} Hz',
                         fontsize=12, fontweight='bold', color='red')
    axes[2, 1].axvline(x=f_signal, color='g', linestyle='--', linewidth=2, label=f'True: {f_signal} Hz')
    axes[2, 1].axvline(x=f_alias, color='r', linestyle='--', linewidth=2, label=f'Alias: {f_alias} Hz')
    axes[2, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[2, 1].set_xlim([0, 50])
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example5_sampling_aliasing.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example5_sampling_aliasing.png")
    print(f"\nObservation: Nyquist Sampling Theorem")
    print(f"  - Signal frequency: {f_signal} Hz")
    print(f"  - Minimum sampling rate: {2*f_signal} Hz (Nyquist rate)")
    print(f"  - Good sampling ({fs_good} Hz): ✓ Correct spectrum")
    print(f"  - Nyquist sampling ({fs_nyquist} Hz): ✓ Just enough")
    print(f"  - Under-sampling ({fs_bad} Hz): ✗ Aliasing! {f_signal} Hz appears as {f_alias} Hz\n")
    plt.close()

#============================================================================
# Example 6: Time-Frequency Resolution Trade-off (Windowing)
#============================================================================

def example6_windowing():
    """Demonstrate the effect of window length on frequency resolution"""
    print("=" * 70)
    print("EXAMPLE 6: Window Length and Frequency Resolution")
    print("=" * 70)

    # Create signal with two close frequencies
    fs = 1000
    f1, f2 = 10, 12  # Two frequencies 2 Hz apart

    # Different window lengths
    windows = [0.5, 1.0, 2.0]  # seconds

    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 12))

    for i, T in enumerate(windows):
        t = np.linspace(0, T, int(fs * T), endpoint=False)
        signal_data = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

        # Compute FFT
        fft_result = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/fs)
        magnitude = np.abs(fft_result)

        # Frequency resolution
        freq_resolution = fs / len(signal_data)

        # Plot time domain
        axes[i, 0].plot(t, signal_data, 'b-', linewidth=1.5)
        axes[i, 0].set_ylabel('Amplitude', fontsize=11)
        axes[i, 0].set_title(f'Window Length: {T} s (N={len(signal_data)} samples)',
                             fontsize=12, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)

        # Plot frequency domain
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]

        axes[i, 1].plot(positive_freqs, positive_magnitude, 'r-', linewidth=2)
        axes[i, 1].axvline(x=f1, color='g', linestyle='--', alpha=0.7, label=f'f1={f1} Hz')
        axes[i, 1].axvline(x=f2, color='orange', linestyle='--', alpha=0.7, label=f'f2={f2} Hz')
        axes[i, 1].set_ylabel('Magnitude', fontsize=11)
        axes[i, 1].set_xlim([5, 17])
        axes[i, 1].grid(True, alpha=0.3)

        # Can we resolve the two peaks?
        if freq_resolution < (f2 - f1):
            resolution_text = f'Δf={freq_resolution:.3f} Hz: CAN resolve! ✓'
            color = 'green'
        else:
            resolution_text = f'Δf={freq_resolution:.3f} Hz: CANNOT resolve ✗'
            color = 'red'

        axes[i, 1].set_title(resolution_text, fontsize=12, fontweight='bold', color=color)
        axes[i, 1].legend(fontsize=10)

    axes[-1, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=12)

    plt.tight_layout()
    plt.savefig('example6_windowing.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example6_windowing.png")
    print(f"\nObservation: Frequency Resolution Trade-off")
    print(f"  - Two frequencies: {f1} Hz and {f2} Hz (separation: {f2-f1} Hz)")
    print(f"  - To resolve them, need Δf < {f2-f1} Hz")
    print(f"  - Frequency resolution: Δf = fs / N = {fs} / N")
    print(f"  - Longer window → More samples → Better frequency resolution")
    print(f"  - Shorter window → Fewer samples → Worse frequency resolution\n")
    plt.close()

#============================================================================
# Example 7: 2D Fourier Transform (Image)
#============================================================================

def example7_2d_fourier():
    """Demonstrate 2D Fourier Transform on a simple image"""
    print("=" * 70)
    print("EXAMPLE 7: 2D Fourier Transform (Images)")
    print("=" * 70)

    # Create a simple image with patterns
    size = 256
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)

    # Create image with horizontal and vertical stripes
    image = np.sin(2 * np.pi * 2 * X) + np.sin(2 * np.pi * 3 * Y)

    # Add a Gaussian blob in center
    gaussian = 2 * np.exp(-(X**2 + Y**2) / 2)
    image += gaussian

    # Compute 2D FFT
    fft_2d = np.fft.fft2(image)
    fft_2d_shifted = np.fft.fftshift(fft_2d)  # Shift zero frequency to center
    magnitude_spectrum = np.log(np.abs(fft_2d_shifted) + 1)  # Log scale for visibility

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original image
    im1 = axes[0].imshow(image, cmap='gray', extent=[-5, 5, -5, 5])
    axes[0].set_title('Spatial Domain: Original Image', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Plot magnitude spectrum
    im2 = axes[1].imshow(magnitude_spectrum, cmap='hot', extent=[-size/2, size/2, -size/2, size/2])
    axes[1].set_title('Frequency Domain: 2D FFT Magnitude', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency X', fontsize=12)
    axes[1].set_ylabel('Frequency Y', fontsize=12)
    axes[1].plot(0, 0, 'g+', markersize=15, markeredgewidth=2, label='DC (0,0)')
    axes[1].legend(fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig('example7_2d_fourier.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example7_2d_fourier.png")
    print("\nObservation: 2D Fourier Transform")
    print("  - Horizontal stripes → Vertical line in frequency domain")
    print("  - Vertical stripes → Horizontal line in frequency domain")
    print("  - Center blob (low frequency) → Bright center in spectrum")
    print("  - Used in image compression (JPEG), filtering, edge detection\n")
    plt.close()

#============================================================================
# Example 8: Real-World Audio Analysis
#============================================================================

def example8_audio_spectrum():
    """Simulate audio spectrum analysis"""
    print("=" * 70)
    print("EXAMPLE 8: Audio Spectrum Analysis (Simulated)")
    print("=" * 70)

    # Simulate a musical chord: C-E-G (C major chord)
    fs = 44100  # CD-quality audio
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Musical notes (in Hz)
    C4 = 261.63  # Middle C
    E4 = 329.63
    G4 = 392.00

    # Create chord with harmonics (more realistic)
    chord = np.zeros_like(t)

    # Add fundamental and harmonics for each note
    for note in [C4, E4, G4]:
        chord += np.sin(2 * np.pi * note * t)  # Fundamental
        chord += 0.3 * np.sin(2 * np.pi * 2 * note * t)  # 2nd harmonic
        chord += 0.1 * np.sin(2 * np.pi * 3 * note * t)  # 3rd harmonic

    # Add envelope (attack-decay)
    envelope = np.exp(-3 * t)
    chord *= envelope

    # Compute FFT
    fft_result = fft(chord)
    freqs = fftfreq(len(chord), 1/fs)
    magnitude = np.abs(fft_result)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot waveform
    axes[0].plot(t[:5000], chord[:5000], 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Time Domain: C Major Chord Waveform (C-E-G)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot spectrum (musical range)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]

    # Focus on musical range (200-1200 Hz)
    music_range = (positive_freqs >= 200) & (positive_freqs <= 1200)

    axes[1].plot(positive_freqs[music_range], positive_magnitude[music_range],
                 'r-', linewidth=1.5)
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title('Frequency Domain: Spectrum Shows Notes and Harmonics',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Mark the fundamental frequencies
    notes = {'C4': C4, 'E4': E4, 'G4': G4}
    colors = {'C4': 'blue', 'E4': 'green', 'G4': 'orange'}

    for note_name, freq in notes.items():
        axes[1].axvline(x=freq, color=colors[note_name], linestyle='--',
                       linewidth=2, alpha=0.7, label=f'{note_name}: {freq:.1f} Hz')
        # Mark harmonics
        axes[1].axvline(x=2*freq, color=colors[note_name], linestyle=':',
                       linewidth=1, alpha=0.5)
        axes[1].axvline(x=3*freq, color=colors[note_name], linestyle=':',
                       linewidth=1, alpha=0.5)

    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example8_audio_spectrum.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: example8_audio_spectrum.png")
    print("\nObservation: Audio Spectrum Analysis")
    print(f"  - Fundamental frequencies:")
    print(f"    C4: {C4:.2f} Hz")
    print(f"    E4: {E4:.2f} Hz")
    print(f"    G4: {G4:.2f} Hz")
    print(f"  - Harmonics (overtones) also visible at 2f, 3f, etc.")
    print(f"  - This is how music recognition and tuners work!")
    print(f"  - Applications: Shazam, auto-tune, audio equalizers\n")
    plt.close()

#============================================================================
# Main Execution
#============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" FOURIER TRANSFORM VISUALIZATIONS - COMPLETE TUTORIAL")
    print("="*70 + "\n")

    # Run all examples
    example1_pure_sine()
    example2_multiple_frequencies()
    example3_rectangular_pulse()
    example4_noisy_signal_filtering()
    example5_sampling_aliasing()
    example6_windowing()
    example7_2d_fourier()
    example8_audio_spectrum()

    print("="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. example1_pure_sine.png - Basic sine wave FFT")
    print("  2. example2_multiple_frequencies.png - Multiple frequency components")
    print("  3. example3_rectangular_pulse.png - Pulse → Sinc function")
    print("  4. example4_noise_filtering.png - Frequency domain filtering")
    print("  5. example5_sampling_aliasing.png - Nyquist theorem and aliasing")
    print("  6. example6_windowing.png - Frequency resolution trade-off")
    print("  7. example7_2d_fourier.png - 2D FFT for images")
    print("  8. example8_audio_spectrum.png - Audio spectrum analysis")
    print("\n" + "="*70)
    print("Review the images to understand Fourier transforms visually!")
    print("="*70 + "\n")
