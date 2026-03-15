from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


class SileroVadClib:
    """Thin ctypes wrapper around the native Silero VAD shared library.

    The wrapped C model currently targets the 16 kHz Silero VAD path. The
    low-level `forward` entry point expects one chunk shaped like the C model
    input, which is typically:

    - 64 samples of left context
    - 512 samples of new audio

    For convenience, `forward` can also manage the left context internally when
    the caller provides only the fresh audio chunk. In that mode, the first
    chunk uses a zero context and later chunks reuse the tail of the previous
    input, similar to the Python `OnnxWrapper`.
    """

    def __init__(self, dll_path: Path, input_samples: int = 576):
        """Load the shared library and create a model instance.

        Args:
            dll_path: Path to the platform shared library (`.dll`, `.so`, or
                `.dylib`).
            input_samples: Model input size expected by the C model. For the
                current 16 kHz path this is usually 576.
        """
        self.dll = ctypes.CDLL(str(dll_path))
        self.input_samples = input_samples
        self._context = np.zeros(0, dtype=np.float32)

        self.dll.silero_vad_get_embedded_weights.restype = ctypes.c_void_p
        self.dll.silero_vad_model_create.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t]
        self.dll.silero_vad_model_create.restype = ctypes.c_void_p
        self.dll.silero_vad_model_reset.argtypes = [ctypes.c_void_p]
        self.dll.silero_vad_model_reset.restype = None
        self.dll.silero_vad_model_destroy.argtypes = [ctypes.c_void_p]
        self.dll.silero_vad_model_destroy.restype = None
        self.dll.silero_vad_model_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.dll.silero_vad_model_forward.restype = ctypes.c_int
        self.dll.silero_vad_model_audio_prob_count.argtypes = [ctypes.c_size_t]
        self.dll.silero_vad_model_audio_prob_count.restype = ctypes.c_size_t
        self.dll.silero_vad_model_forward_audio.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.dll.silero_vad_model_forward_audio.restype = ctypes.c_int

        weights = self.dll.silero_vad_get_embedded_weights()
        self.model = self.dll.silero_vad_model_create(weights, input_samples)
        if not self.model:
            raise RuntimeError(
                "Failed to create Silero VAD model from embedded weights")

    def reset(self) -> None:
        """Reset the internal model state and rolling audio context."""
        self.dll.silero_vad_model_reset(self.model)
        self._context = np.zeros(0, dtype=np.float32)

    def forward(
        self,
        chunk: np.ndarray,
        *,
        has_context: bool = True,
        context_length: int = 64,
    ) -> float:
        """Run one model step and return the speech probability.

        Args:
            chunk: Audio samples for one inference step. If `has_context` is
                true, this must already include the left context and typically
                have length `input_samples`. If `has_context` is false, the
                method prepends the internal rolling context and expects the
                fresh chunk length to be `input_samples - context_length`.
            has_context: Whether `chunk` already includes the left context.
            context_length: Number of context samples to prepend and track when
                `has_context` is false.

        Returns:
            The speech probability for this chunk.
        """
        chunk_array = np.ascontiguousarray(chunk, dtype=np.float32).reshape(-1)
        if has_context:
            if chunk_array.shape[0] != self.input_samples:
                raise ValueError(
                    f"Expected {self.input_samples} samples with context, got {chunk_array.shape[0]}"
                )
        else:
            if context_length < 0 or context_length > self.input_samples:
                raise ValueError(
                    f"context_length must be between 0 and {self.input_samples}, got {context_length}"
                )
            expected_chunk = self.input_samples - context_length
            if chunk_array.shape[0] != expected_chunk:
                raise ValueError(
                    f"Expected {expected_chunk} samples without context, got {chunk_array.shape[0]}"
                )
            if self._context.shape[0] != context_length:
                self._context = np.zeros(context_length, dtype=np.float32)
            chunk_array = np.concatenate(
                [self._context, chunk_array]
            )
            self._context = chunk_array[-context_length:].copy()

        output = ctypes.c_float()
        status = self.dll.silero_vad_model_forward(
            self.model,
            chunk_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(output),
        )
        if status != 0:
            raise RuntimeError(
                f"silero_vad_model_forward failed with status {status}")
        return float(output.value)

    def forward_audio(self, audio: np.ndarray) -> np.ndarray:
        """Run full-audio inference and return the probability sequence."""
        audio_array = np.ascontiguousarray(audio, dtype=np.float32)
        probs_count = self.dll.silero_vad_model_audio_prob_count(
            audio_array.shape[0])
        probs = np.empty(probs_count, dtype=np.float32)
        written = ctypes.c_size_t()
        status = self.dll.silero_vad_model_forward_audio(
            self.model,
            audio_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            audio_array.shape[0],
            probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            probs.shape[0],
            ctypes.byref(written),
        )
        if status != 0:
            raise RuntimeError(
                f"silero_vad_model_forward_audio failed with status {status}")
        return probs[: written.value]

    def close(self) -> None:
        """Destroy the native model and release the shared-library resources."""
        if self.model:
            self.dll.silero_vad_model_destroy(self.model)
            self.model = None

    def __enter__(self) -> "SileroVadClib":
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the model when leaving a context-manager scope."""
        self.close()


def get_speech_probs_c(audio: np.ndarray, model: SileroVadClib, sampling_rate: int = 16000) -> list[float]:
    """Return per-chunk speech probabilities from the native C model."""
    if sampling_rate != 16000:
        raise ValueError("This C port currently targets the 16 kHz model only")

    return model.forward_audio(audio).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, default=None)
    parser.add_argument("--audio", type=Path,
                        default=Path("LJ001-0001_16k.wav"))
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()

    from src.silero_vad.utils_vad import read_audio

    dll_path = args.dll
    if dll_path is None:
        for candidate in (Path("build-msvc/silero_vad.dll"), Path("build/silero_vad.dll")):
            if candidate.exists():
                dll_path = candidate
                break
    if dll_path is None:
        raise FileNotFoundError(
            "Could not find silero_vad.dll. Pass --dll explicitly.")

    audio = read_audio(
        str(args.audio), sampling_rate=args.sampling_rate).numpy().astype(np.float32)

    with SileroVadClib(dll_path) as model:
        probs = get_speech_probs_c(
            audio, model, sampling_rate=args.sampling_rate)

    print(json.dumps(probs))


if __name__ == "__main__":
    main()
