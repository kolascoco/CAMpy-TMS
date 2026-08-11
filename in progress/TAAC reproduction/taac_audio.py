"""Audio engine for the TAAC-inspired masking notebook.

Digital gain is not calibrated sound pressure level. Validate the complete
headphone chain independently and start at a low hardware volume.
"""

from __future__ import annotations

import hashlib
import json
import threading
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # generation remains usable without PortAudio
    sd = None


@dataclass
class Settings:
    sample_rate: int = 48_000
    duration_s: float = 10.0
    seed: int = 20260811
    noise_mode: str = "hybrid"
    click_ratio: float = 0.75
    broadband: bool = True
    layers: int = 7
    min_factor: float = 0.45
    max_factor: float = 1.80
    master_gain: float = 0.35
    limiter: float = 0.95
    include_clicks: bool = True
    click_gain: float = 5.0
    schedule: str = "single-pulse"
    single_min_interval_s: float = 0.6
    single_max_interval_s: float = 1.4
    rhythmic_hz: float = 5.0
    pulses_per_train: int = 5
    inter_train_interval_s: float = 2.0
    probability: float = 0.5
    jitter_s: float = 0.0
    amplitude_variation: float = 0.15


def _audio(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("Audio must be a non-empty, finite mono array")
    return x


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if channels not in (1, 2) or width not in (2, 3):
        raise ValueError("Use mono/stereo 16- or 24-bit PCM WAV")
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    else:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8 | b[:, 2].astype(np.int32) << 16
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608
    return _audio(x.reshape(-1, channels) if channels == 2 else x), rate


def write_wav(path: str | Path, samples: np.ndarray, rate: int, bits: int = 24) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(_audio(samples), -1, 1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setframerate(rate)
        if bits == 24:
            v = np.round(x * 8388607).astype(np.int32)
            packed = np.column_stack((v & 255, (v >> 8) & 255, (v >> 16) & 255)).astype(np.uint8)
            wav.setsampwidth(3); wav.writeframes(packed.tobytes())
        else:
            wav.setsampwidth(2); wav.writeframes(np.round(x * 32767).astype("<i2").tobytes())
    return path


def resample(x: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return _audio(x).copy()
    x = _audio(x); n = int(max(1, round(x.size * target_rate / source_rate)))
    return np.interp(np.linspace(0, x.size, n, endpoint=False), np.arange(x.size), x).astype(np.float32)


def suggest_pulse_bounds(x: np.ndarray, rate: int, pre_ms: float = 2, post_ms: float = 30) -> tuple[int, int]:
    x = _audio(x); peak = int(np.argmax(np.abs(x)))
    return max(0, peak - round(pre_ms * rate / 1000)), min(x.size, peak + round(post_ms * rate / 1000))


def crop_pulse(x: np.ndarray, start_ms: float, end_ms: float, rate: int) -> np.ndarray:
    start, end = round(start_ms * rate / 1000), round(end_ms * rate / 1000)
    if start < 0 or end <= start or end > len(x):
        raise ValueError("Pulse boundaries are outside the recording")
    pulse = _audio(x)[start:end].copy()
    edge = min(max(1, round(0.0005 * rate)), pulse.size // 4)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        pulse[:edge] *= ramp; pulse[-edge:] *= ramp[::-1]
    return pulse - np.mean(pulse)


def schedule_times(s: Settings, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= s.probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    times: list[float] = []
    if s.schedule == "single-pulse":
        if not 0 < s.single_min_interval_s <= s.single_max_interval_s:
            raise ValueError("Invalid single-pulse interval")
        t = rng.uniform(0, s.single_max_interval_s)
        while t < s.duration_s:
            if rng.random() < s.probability: times.append(t)
            t += rng.uniform(s.single_min_interval_s, s.single_max_interval_s)
    elif s.schedule == "rhythmic":
        if s.rhythmic_hz <= 0 or s.pulses_per_train < 1 or s.inter_train_interval_s < 0:
            raise ValueError("Invalid rhythmic settings")
        span = (s.pulses_per_train - 1) / s.rhythmic_hz
        t = 0.0
        while t < s.duration_s:
            if rng.random() < s.probability:
                for p in range(s.pulses_per_train):
                    event = t + p / s.rhythmic_hz + rng.uniform(-s.jitter_s, s.jitter_s)
                    if 0 <= event < s.duration_s: times.append(event)
            t += span + s.inter_train_interval_s
    else:
        raise ValueError("Schedule must be single-pulse or rhythmic")
    return np.sort(np.asarray(times, dtype=float))


def _spectral_noise(pulse: np.ndarray, count: int, rng: np.random.Generator, factors: np.ndarray) -> np.ndarray:
    total = np.zeros(count, dtype=np.float64)
    for factor in factors:
        shaped = resample(pulse, 10_000, max(1, round(10_000 * factor)))
        mag = np.abs(np.fft.rfft(np.resize(shaped, count)))
        width = max(3, min(401, (mag.size // 100) | 1))
        mag = np.convolve(mag, np.ones(width) / width, "same")
        phase = rng.uniform(0, 2 * np.pi, mag.size); phase[0] = 0
        total += np.fft.irfft(mag * np.exp(1j * phase), count)
    total -= total.mean(); rms = np.sqrt(np.mean(total * total))
    return (total / max(rms, np.finfo(float).eps)).astype(np.float32)


def generate(pulse: np.ndarray, pulse_rate: int, s: Settings) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.1 <= s.duration_s <= 300 or not 8_000 <= s.sample_rate <= 192_000:
        raise ValueError("Duration must be 0.1–300 s and sample rate 8–192 kHz")
    if s.noise_mode not in {"white", "click-derived", "hybrid"}:
        raise ValueError("Unknown noise mode")
    rng = np.random.default_rng(s.seed); pulse = resample(pulse, pulse_rate, s.sample_rate)
    count = round(s.duration_s * s.sample_rate)
    white = rng.normal(size=count).astype(np.float32); white /= np.sqrt(np.mean(white * white))
    factors = np.linspace(s.min_factor, s.max_factor, s.layers if s.broadband else 1)
    click_noise = _spectral_noise(pulse, count, rng, factors)
    ratio = {"white": 0.0, "click-derived": 1.0}.get(s.noise_mode, s.click_ratio)
    noise = np.sqrt(1-ratio) * white + np.sqrt(ratio) * click_noise
    times = schedule_times(s, rng) if s.include_clicks else np.array([])
    clicks = np.zeros(count, np.float32)
    for t in times:
        start = int(round(float(t) * s.sample_rate)); gain = 1-rng.uniform(0, s.amplitude_variation)
        n = min(pulse.size, count-start); clicks[start:start+n] += pulse[:n] * gain
    mixed = noise + s.click_gain * clicks
    mixed -= mixed.mean(); rms = np.sqrt(np.mean(mixed*mixed)); mixed /= max(rms, np.finfo(float).eps)
    mixed *= s.master_gain; mixed = np.tanh(mixed / s.limiter) * s.limiter
    stats = audio_stats(mixed); stats.update(events=int(times.size), settings=asdict(s))
    return mixed.astype(np.float32), stats


def audio_stats(x: np.ndarray) -> dict[str, float | int]:
    x = _audio(x); peak = float(np.max(np.abs(x))); rms = float(np.sqrt(np.mean(x*x)))
    return {"peak": peak, "rms": rms, "dc": float(np.mean(x)), "clipped": int(np.sum(np.abs(x) >= 1)), "samples": x.size}


def list_input_devices() -> list[tuple[int, str]]:
    if sd is None: return []
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def record(duration_s: float, rate: int, device: int | None = None) -> np.ndarray:
    if sd is None: raise RuntimeError("Install sounddevice and PortAudio to record")
    data = sd.rec(round(duration_s*rate), samplerate=rate, channels=1, dtype="float32", device=device)
    sd.wait(); return _audio(data)


class RecordingSession:
    """Non-blocking microphone capture controlled by notebook buttons."""
    def __init__(self) -> None:
        self.stream = None; self.blocks: list[np.ndarray] = []; self.rate = 0

    def start(self, rate: int, device: int | None = None) -> None:
        if sd is None: raise RuntimeError("Install sounddevice and PortAudio to record")
        self.stop(discard=True); self.blocks = []; self.rate = rate
        def callback(indata, frames, time, status): self.blocks.append(indata[:, 0].copy())
        self.stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device, callback=callback)
        self.stream.start()

    def stop(self, discard: bool = False) -> np.ndarray:
        if self.stream is not None:
            self.stream.stop(); self.stream.close(); self.stream = None
        result = np.concatenate(self.blocks) if self.blocks else np.array([], dtype=np.float32)
        if discard: self.blocks = []
        return result


class StreamingPlayer:
    def __init__(self) -> None:
        self.stream = None; self.samples = None; self.position = 0; self.paused = False
        self.lock = threading.Lock()

    def play(self, samples: np.ndarray, rate: int, loop: bool = True, device: int | None = None) -> None:
        if sd is None: raise RuntimeError("Install sounddevice and PortAudio to play")
        self.stop(); self.samples = _audio(samples); self.position = 0; self.loop = loop
        def callback(outdata, frames, time, status):
            with self.lock:
                outdata.fill(0)
                if self.paused or self.samples is None: return
                written = 0
                while written < frames and self.samples is not None:
                    n = min(frames-written, self.samples.size-self.position)
                    outdata[written:written+n, 0] = self.samples[self.position:self.position+n]
                    written += n; self.position += n
                    if self.position == self.samples.size:
                        if self.loop: self.position = 0
                        else: self.samples = None
        self.stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", device=device, callback=callback)
        self.stream.start()

    def pause(self) -> None:
        with self.lock: self.paused = True

    def resume(self) -> None:
        with self.lock: self.paused = False

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop(); self.stream.close()
        self.stream = None; self.samples = None; self.position = 0; self.paused = False


def save_settings(path: str | Path, settings: Settings, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"settings": asdict(settings), "extra": extra or {}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); return path


def load_settings(path: str | Path) -> tuple[Settings, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Settings(**payload["settings"]), payload.get("extra", {})


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_name(s: Settings) -> str:
    mode = s.noise_mode.replace("click-derived", "click")
    sched = "single" if s.schedule == "single-pulse" else f"train{s.pulses_per_train}x{s.rhythmic_hz:g}Hz"
    return f"taac_{mode}_r{round(s.click_ratio*100):02d}_{sched}_p{round(s.probability*100):02d}_cg{s.click_gain:g}_{s.duration_s:g}s"
