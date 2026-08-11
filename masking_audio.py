"""Generate masking noise shaped from a replaceable TMS click-track WAV.

This utility is not a calibrated hearing-safety or TMS synchronization system.
Start playback at a low device volume and validate the output independently.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_SEED = 20260810


def _validate(duration_s: float, volume: float, sample_rate: int) -> None:
    if not 0.1 <= duration_s <= 300:
        raise ValueError("duration_s must be between 0.1 and 300 seconds")
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between 0.0 and 1.0")
    if not 8_000 <= sample_rate <= 192_000:
        raise ValueError("sample_rate must be between 8000 and 192000 Hz")


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write floating-point mono samples in [-1, 1] as 16-bit PCM."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read uncompressed 16- or 24-bit PCM WAV, averaging stereo to mono."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Click track not found: {source}")
    with wave.open(str(source), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        compression = wav.getcomptype()
        raw = wav.readframes(frames)
    if width not in (2, 3) or compression != "NONE":
        raise ValueError("Click track must be an uncompressed 16- or 24-bit PCM WAV")
    if channels not in (1, 2):
        raise ValueError("Click track must be mono or stereo")
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        data = values.astype(np.float32) / 8388608.0
    if channels == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    if data.size == 0:
        raise ValueError("Click track is empty")
    return data, rate


def make_placeholder_clicks(
    path: str | Path,
    duration_s: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    interval_s: float = 1.0,
) -> Path:
    """Create a quiet synthetic placeholder click track for wiring/testing."""
    _validate(duration_s, 0.15, sample_rate)
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    samples = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    click_length = max(1, round(0.008 * sample_rate))
    envelope = np.exp(-np.linspace(0, 8, click_length, dtype=np.float32))
    click = 0.15 * envelope * np.sin(
        2 * np.pi * 1_000 * np.arange(click_length, dtype=np.float32) / sample_rate
    )
    for start in range(0, samples.size, max(1, round(interval_s * sample_rate))):
        stop = min(start + click_length, samples.size)
        samples[start:stop] += click[: stop - start]
    return write_wav(path, samples, sample_rate)


def make_fake_single_click(
    path: str | Path,
    duration_s: float = 1.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    click_time_s: float = 0.5,
) -> Path:
    """Create one conspicuous synthetic biphasic click for software testing."""
    _validate(duration_s, 0.85, sample_rate)
    if not 0.0 <= click_time_s < duration_s:
        raise ValueError("click_time_s must fall inside the file")
    samples = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    click_length = max(4, round(0.012 * sample_rate))
    t = np.arange(click_length, dtype=np.float32) / sample_rate
    envelope = np.exp(-t / 0.0025)
    # Two frequencies make the fake click easy to distinguish from the noise.
    click = 0.85 * envelope * (
        0.65 * np.sin(2 * np.pi * 900 * t) + 0.35 * np.sin(2 * np.pi * 2_700 * t)
    )
    click /= max(float(np.max(np.abs(click))), np.finfo(float).eps)
    click *= 0.85
    start = round(click_time_s * sample_rate)
    stop = min(start + click_length, samples.size)
    samples[start:stop] = click[: stop - start]
    return write_wav(path, samples, sample_rate)


def _linear_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    new_size = max(1, round(samples.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_size, endpoint=False)
    return np.interp(new_x, old_x, samples).astype(np.float32)


def _periodic_noise(
    count: int, rng: np.random.Generator, magnitude: np.ndarray | None = None
) -> np.ndarray:
    """Synthesize a periodic Gaussian-like buffer from integer Fourier bins."""
    bins = count // 2 + 1
    spectrum = rng.normal(size=bins) + 1j * rng.normal(size=bins)
    spectrum[0] = 0.0
    if count % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    if magnitude is not None:
        spectrum *= magnitude
    samples = np.fft.irfft(spectrum, n=count)
    rms = float(np.sqrt(np.mean(samples**2)))
    return (samples / max(rms, np.finfo(float).eps)).astype(np.float32)


def _click_shaped_noise(clicks: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Create periodic noise with the smoothed magnitude spectrum of the clicks."""
    template = np.resize(clicks, count)
    magnitude = np.abs(np.fft.rfft(template))
    smooth_width = max(3, min(401, magnitude.size // 100))
    if smooth_width % 2 == 0:
        smooth_width += 1
    magnitude = np.convolve(magnitude, np.ones(smooth_width) / smooth_width, mode="same")
    magnitude /= max(float(magnitude.max()), np.finfo(float).eps)
    magnitude = np.maximum(magnitude, 0.02)
    return _periodic_noise(count, rng, magnitude)


def _extract_click_event(clicks: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract one transient around the largest peak in the click recording."""
    peak = float(np.max(np.abs(clicks)))
    if peak <= np.finfo(float).eps:
        raise ValueError("Click recording contains no measurable transient")
    peak_index = int(np.argmax(np.abs(clicks)))
    start = max(0, peak_index - round(0.002 * sample_rate))
    stop = min(clicks.size, peak_index + round(0.030 * sample_rate))
    return clicks[start:stop].copy()


def schedule_click_times(
    duration_s: float,
    schedule: str = "single-pulse",
    single_pulse_rate_hz: float = 1.0,
    rhythmic_frequency_hz: float = 5.0,
    pulses_per_train: int = 5,
    inter_train_interval_s: float = 1.0,
    click_probability: float = 0.5,
    seed: int = DEFAULT_SEED + 1,
) -> np.ndarray:
    """Keep random or rhythmic candidate clicks using Bernoulli probability."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if schedule not in {"single-pulse", "rhythmic"}:
        raise ValueError("schedule must be 'single-pulse' or 'rhythmic'")
    if not 0.0 <= click_probability <= 1.0:
        raise ValueError("click_probability must be between 0.0 and 1.0")
    rng = np.random.default_rng(seed)
    if schedule == "single-pulse":
        if not 0.05 <= single_pulse_rate_hz <= 20.0:
            raise ValueError("single_pulse_rate_hz must be between 0.05 and 20")
        spacing = 1.0 / single_pulse_rate_hz
        candidates = np.arange(spacing / 2.0, duration_s, spacing)
        candidates += rng.uniform(-0.45 * spacing, 0.45 * spacing, candidates.size)
        candidates = candidates[(candidates >= 0.0) & (candidates < duration_s)]
    else:
        if not 0.1 <= rhythmic_frequency_hz <= 50.0:
            raise ValueError("rhythmic_frequency_hz must be between 0.1 and 50")
        if not 1 <= pulses_per_train <= 1000:
            raise ValueError("pulses_per_train must be between 1 and 1000")
        if not 0.0 <= inter_train_interval_s <= 300.0:
            raise ValueError("inter_train_interval_s must be between 0 and 300")
        pulse_spacing = 1.0 / rhythmic_frequency_hz
        train_span = (pulses_per_train - 1) * pulse_spacing
        train_period = train_span + inter_train_interval_s
        if train_period <= 0:
            raise ValueError("rhythmic train period must be positive")
        starts = np.arange(0.0, duration_s, train_period)
        offsets = np.arange(pulses_per_train, dtype=float) * pulse_spacing
        candidates = (starts[:, None] + offsets[None, :]).ravel()
        candidates = candidates[candidates < duration_s]
    keep = rng.random(candidates.size) < click_probability
    return candidates[keep]


def _render_click_events(
    click_event: np.ndarray, times_s: np.ndarray, count: int, sample_rate: int
) -> np.ndarray:
    rendered = np.zeros(count, dtype=np.float32)
    for time_s in times_s:
        start = round(float(time_s) * sample_rate) % count
        first = min(click_event.size, count - start)
        rendered[start : start + first] += click_event[:first]
        remaining = click_event.size - first
        if remaining:
            rendered[:remaining] += click_event[first:]
    return rendered


def build_masking_track(
    click_path: str | Path,
    duration_s: float = 10.0,
    volume: float = 0.20,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int = DEFAULT_SEED,
    noise_mode: str = "click-shaped",
    include_clicks: bool = False,
    click_volume: float = 2.5,
    click_schedule: str = "single-pulse",
    single_pulse_rate_hz: float = 1.0,
    rhythmic_frequency_hz: float = 5.0,
    pulses_per_train: int = 5,
    inter_train_interval_s: float = 1.0,
    click_probability: float = 0.5,
) -> np.ndarray:
    """Return white or click-shaped masking, optionally adding clicks for preview."""
    _validate(duration_s, volume, sample_rate)
    if noise_mode not in {"white", "click-shaped"}:
        raise ValueError("noise_mode must be 'white' or 'click-shaped'")
    if not 0.0 <= click_volume <= 50.0:
        raise ValueError("click_volume must be between 0.0 and 50.0")
    clicks, click_rate = read_wav_mono(click_path)
    clicks = _linear_resample(clicks, click_rate, sample_rate)
    click_event = _extract_click_event(clicks, sample_rate)
    count = round(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    noise = (
        _click_shaped_noise(click_event, count, rng)
        if noise_mode == "click-shaped"
        else _periodic_noise(count, rng)
    )
    if include_clicks:
        times = schedule_click_times(
            duration_s,
            click_schedule,
            single_pulse_rate_hz,
            rhythmic_frequency_hz,
            pulses_per_train,
            inter_train_interval_s,
            click_probability,
            seed + 1,
        )
        scheduled_clicks = _render_click_events(click_event, times, count, sample_rate)
    else:
        scheduled_clicks = 0.0
    mixed = noise + click_volume * scheduled_clicks
    peak = float(np.max(np.abs(mixed)))
    if peak > 0:
        mixed /= peak
    return (mixed * volume).astype(np.float32)


def create_masking_wav(
    click_path: str | Path,
    output_path: str | Path,
    duration_s: float = 10.0,
    volume: float = 0.20,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int = DEFAULT_SEED,
    noise_mode: str = "click-shaped",
    include_clicks: bool = False,
    click_volume: float = 2.5,
    click_schedule: str = "single-pulse",
    single_pulse_rate_hz: float = 1.0,
    rhythmic_frequency_hz: float = 5.0,
    pulses_per_train: int = 5,
    inter_train_interval_s: float = 1.0,
    click_probability: float = 0.5,
) -> Path:
    samples = build_masking_track(
        click_path,
        duration_s,
        volume,
        sample_rate,
        seed,
        noise_mode,
        include_clicks,
        click_volume,
        click_schedule,
        single_pulse_rate_hz,
        rhythmic_frequency_hz,
        pulses_per_train,
        inter_train_interval_s,
        click_probability,
    )
    return write_wav(output_path, samples, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clicks", type=Path, required=True, help="16-bit PCM WAV click track")
    parser.add_argument("--output", type=Path, required=True, help="output masking WAV")
    parser.add_argument("--duration", type=float, default=10.0, help="duration in seconds")
    parser.add_argument("--volume", type=float, default=0.20, help="master amplitude, 0.0 to 1.0")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--noise-mode", choices=("white", "click-shaped"), default="click-shaped")
    parser.add_argument(
        "--include-clicks", action="store_true", help="mix clicks into output for preview only"
    )
    parser.add_argument(
        "--click-volume", type=float, default=5.0, help="relative preview-click volume, 0 to 50"
    )
    parser.add_argument(
        "--click-schedule", choices=("single-pulse", "rhythmic"), default="single-pulse"
    )
    parser.add_argument("--single-pulse-rate", type=float, default=1.0, help="jittered opportunities/second")
    parser.add_argument("--rhythmic-frequency", type=float, default=5.0, help="rhythmic grid Hz")
    parser.add_argument("--pulses-per-train", type=int, default=5)
    parser.add_argument("--inter-train-interval", type=float, default=1.0, help="seconds")
    parser.add_argument("--click-probability", type=float, default=0.5, help="probability 0 to 1")
    parser.add_argument("--make-placeholder", action="store_true", help="create clicks file first")
    parser.add_argument(
        "--make-fake-single-click", action="store_true", help="create a one-click demo WAV first"
    )
    args = parser.parse_args()
    if args.make_placeholder:
        make_placeholder_clicks(args.clicks, sample_rate=args.sample_rate)
    if args.make_fake_single_click:
        make_fake_single_click(args.clicks, sample_rate=args.sample_rate)
    result = create_masking_wav(
        args.clicks,
        args.output,
        args.duration,
        args.volume,
        args.sample_rate,
        args.seed,
        args.noise_mode,
        args.include_clicks,
        args.click_volume,
        args.click_schedule,
        args.single_pulse_rate,
        args.rhythmic_frequency,
        args.pulses_per_train,
        args.inter_train_interval,
        args.click_probability,
    )
    print(result.resolve())


if __name__ == "__main__":
    main()
