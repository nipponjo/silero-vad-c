# %%

import matplotlib.pyplot as plt
import time
import numpy as np
from run_silero_vad_clib import SileroVadClib
from src.silero_vad.utils_vad import read_audio, get_timestamps_from_probs

# %%

sr = 16000
audio = read_audio('src/silero_vad/test/tests_data_test.wav', sr)
audio_len_sec = len(audio) / sr
print(f"Audio duration: {audio_len_sec:.1f} seconds")

c_model = SileroVadClib('dist/silero-vad-windows-x64-avx2/silero_vad.dll')

# %%

t0 = time.perf_counter()
probs = c_model.forward_audio(audio)
dt_sec = time.perf_counter()-t0
print(f"VAD: {dt_sec:.5f} seconds, {dt_sec/audio_len_sec*1000:.5} ms / input second")

# %%

plt.plot(probs)

# %%

timestamps = get_timestamps_from_probs(probs, len(audio), return_seconds=True)
print(timestamps)

# %%

# Example: step through the audio chunk by chunk with `forward(...)`.
# Here the wrapper manages the 64-sample rolling left context internally.
step = 512
loop_probs = []
c_model.reset()

t0 = time.perf_counter()
for start in range(0, len(audio), step):
    chunk = audio[start:start + step]
    if len(chunk) < step:
        pad_width = step - len(chunk)
        chunk = np.pad(chunk, (0, pad_width))
    loop_probs.append(c_model.forward(chunk, has_context=False, context_length=64))
dt_loop = time.perf_counter() - t0

print(f"Looped VAD: {dt_loop:.5f} seconds, {dt_loop/audio_len_sec*1000:.5} ms / input second")

# %%

plt.plot(probs)
plt.plot(loop_probs)

# %%

timestamps_loop = get_timestamps_from_probs(loop_probs, len(audio), return_seconds=True)
print(timestamps_loop)
