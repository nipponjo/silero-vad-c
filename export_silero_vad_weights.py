from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


STATE_LAYOUT = [
    # ("stft_conv.weight", "silero_vad_stft_weight", (258, 1, 256)),
    ("conv1.weight", "silero_vad_conv1_weight", (128, 129, 3)),
    ("conv1.bias", "silero_vad_conv1_bias", (128,)),
    ("conv2.weight", "silero_vad_conv2_weight", (64, 128, 3)),
    ("conv2.bias", "silero_vad_conv2_bias", (64,)),
    ("conv3.weight", "silero_vad_conv3_weight", (64, 64, 3)),
    ("conv3.bias", "silero_vad_conv3_bias", (64,)),
    ("conv4.weight", "silero_vad_conv4_weight", (128, 64, 3)),
    ("conv4.bias", "silero_vad_conv4_bias", (128,)),
    ("lstm_cell.weight_ih", "silero_vad_lstm_weight_ih", (512, 128)),
    ("lstm_cell.weight_hh", "silero_vad_lstm_weight_hh", (512, 128)),
    ("lstm_cell.bias_ih", "silero_vad_lstm_bias_ih", (512,)),
    ("lstm_cell.bias_hh", "silero_vad_lstm_bias_hh", (512,)),
    ("final_conv.weight", "silero_vad_final_conv_weight", (1, 128, 1)),
    ("final_conv.bias", "silero_vad_final_conv_bias", (1,)),
]


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file

        state_dict = load_file(str(path))
        return {key: np.asarray(value, dtype=np.float32) for key, value in state_dict.items()}
    except ImportError:
        os.environ.setdefault("DEBUG", "0")
        from tinygrad.nn.state import safe_load

        state_dict = safe_load(str(path))
        return {key: np.asarray(value.numpy(), dtype=np.float32) for key, value in state_dict.items()}


def format_array(values: np.ndarray) -> str:
    flat = values.reshape(-1)
    lines: list[str] = []
    line_parts: list[str] = []

    for idx, value in enumerate(flat, start=1):
        literal = f"{float(value):.9g}"
        if "e" not in literal.lower() and "." not in literal:
            literal += ".0"
        line_parts.append(f"{literal}f")
        if idx % 8 == 0:
            lines.append("  " + ", ".join(line_parts))
            line_parts = []

    if line_parts:
        lines.append("  " + ", ".join(line_parts))

    return ",\n".join(lines)


def write_header(path: Path) -> None:
    content = """#ifndef SILERO_VAD_WEIGHTS_H
#define SILERO_VAD_WEIGHTS_H

#include "silero_vad.h"

#ifdef __cplusplus
extern "C" {
#endif

SILERO_VAD_API const SileroVadWeights *silero_vad_get_embedded_weights(void);

#ifdef __cplusplus
}
#endif

#endif
"""
    path.write_text(content, encoding="ascii")


def write_source(path: Path, state_dict: dict[str, np.ndarray]) -> None:
    lines = [
        '#include "silero_vad_weights.h"',
        "",
    ]

    for state_key, c_name, expected_shape in STATE_LAYOUT:
        value = state_dict.get(state_key)
        if value is None:
            raise KeyError(f"Missing tensor: {state_key}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Unexpected shape for {state_key}: got {tuple(value.shape)}, expected {expected_shape}"
            )

        lines.append(f"static const float {c_name}[] = {{")
        lines.append(format_array(value))
        lines.append("};")
        lines.append("")

    lines.extend(
        [
            "static const SileroVadWeights kSileroVadWeights = {",
            # "  silero_vad_stft_weight,",
            "  silero_vad_conv1_weight,",
            "  silero_vad_conv1_bias,",
            "  silero_vad_conv2_weight,",
            "  silero_vad_conv2_bias,",
            "  silero_vad_conv3_weight,",
            "  silero_vad_conv3_bias,",
            "  silero_vad_conv4_weight,",
            "  silero_vad_conv4_bias,",
            "  silero_vad_lstm_weight_ih,",
            "  silero_vad_lstm_weight_hh,",
            "  silero_vad_lstm_bias_ih,",
            "  silero_vad_lstm_bias_hh,",
            "  silero_vad_final_conv_weight,",
            "  silero_vad_final_conv_bias",
            "};",
            "",
            "const SileroVadWeights *silero_vad_get_embedded_weights(void) {",
            "  return &kSileroVadWeights;",
            "}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/silero_vad/data/silero_vad_16k.safetensors"),
        help="Path to the 16 kHz safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where silero_vad_weights.c/.h will be written.",
    )
    args = parser.parse_args()

    state_dict = load_state_dict(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_header(args.output_dir / "silero_vad_weights.h")
    write_source(args.output_dir / "silero_vad_weights.c", state_dict)


if __name__ == "__main__":
    main()
