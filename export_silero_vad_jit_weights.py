from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from export_silero_vad_weights import STATE_LAYOUT, format_array, write_header


JIT_TO_EXPORT_NAME = {
    # "stft.forward_basis_buffer": "stft_conv.weight",
    "encoder.0.reparam_conv.weight": "conv1.weight",
    "encoder.0.reparam_conv.bias": "conv1.bias",
    "encoder.1.reparam_conv.weight": "conv2.weight",
    "encoder.1.reparam_conv.bias": "conv2.bias",
    "encoder.2.reparam_conv.weight": "conv3.weight",
    "encoder.2.reparam_conv.bias": "conv3.bias",
    "encoder.3.reparam_conv.weight": "conv4.weight",
    "encoder.3.reparam_conv.bias": "conv4.bias",
    "decoder.rnn.weight_ih": "lstm_cell.weight_ih",
    "decoder.rnn.weight_hh": "lstm_cell.weight_hh",
    "decoder.rnn.bias_ih": "lstm_cell.bias_ih",
    "decoder.rnn.bias_hh": "lstm_cell.bias_hh",
    "decoder.decoder.2.weight": "final_conv.weight",
    "decoder.decoder.2.bias": "final_conv.bias",
}


def load_jit_state_dict(force_reload: bool) -> dict[str, np.ndarray]:
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=force_reload,
    )
    state_dict = model._model.state_dict()
    return {
        JIT_TO_EXPORT_NAME[key]: value.detach().cpu().numpy().astype(np.float32)
        for key, value in state_dict.items()
        if key in JIT_TO_EXPORT_NAME
    }


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
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where silero_vad_weights.c/.h will be written.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force torch.hub to refresh the downloaded model.",
    )
    args = parser.parse_args()

    state_dict = load_jit_state_dict(force_reload=args.force_reload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_header(args.output_dir / "silero_vad_weights.h")
    write_source(args.output_dir / "silero_vad_weights.c", state_dict)


if __name__ == "__main__":
    main()
