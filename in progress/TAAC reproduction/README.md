# TAAC-inspired masking lab

Jupyter notebook for recording or loading a TMS click, selecting one pulse, and
generating white, click-derived, or hybrid masking audio. It supports probabilistic
single clicks, rhythmic trains, continuous playback, and WAV/JSON export.

## Run

```bash
python3 -m pip install -r requirements.txt
jupyter lab taac_masking_lab.ipynb
```

On macOS, recording may require `brew install portaudio` and microphone permission.
Start at low headphone volume: digital gain is not calibrated dB SPL.

Recordings, generated audio, and local settings are ignored by Git.
