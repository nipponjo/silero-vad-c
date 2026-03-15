#ifndef SILERO_VAD_H
#define SILERO_VAD_H

#include <stddef.h>

/*
 * Standalone C port of the 16 kHz Silero VAD model.
 *
 * This API exposes both low-level reusable building blocks (Conv1d, LSTMCell)
 * and a full-model interface for chunked or full-audio inference.
 *
 * Current scope:
 * - sample rate: 16000 Hz
 * - model topology: fixed 16 kHz Silero VAD path
 * - streaming chunk shape: typically 576 samples
 *   (64 samples of left context + 512 new samples)
 *
 * Weight layout:
 * - weights are supplied through SileroVadWeights
 * - tensor shapes and ordering are expected to match the exporter scripts in
 *   this repository
 * - for parity with the Torch hub JIT model, use the JIT weight exporter
 */

#if defined(_WIN32)
#  if defined(SILERO_VAD_BUILD_DLL)
#    define SILERO_VAD_API __declspec(dllexport)
#  elif defined(SILERO_VAD_USE_DLL)
#    define SILERO_VAD_API __declspec(dllimport)
#  else
#    define SILERO_VAD_API
#  endif
#else
#  define SILERO_VAD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  SILERO_VAD_STFT_FFT_SIZE = 256,
  SILERO_VAD_STFT_HOP_SIZE = 128,
  SILERO_VAD_STFT_RIGHT_PAD = 64,
  SILERO_VAD_STFT_BINS = 129,
  SILERO_VAD_HIDDEN_SIZE = 128
};

typedef enum SileroVadStatus {
  SILERO_VAD_STATUS_OK = 0,
  SILERO_VAD_STATUS_INVALID_ARGUMENT = 1,
  SILERO_VAD_STATUS_ALLOCATION_FAILED = 2,
  SILERO_VAD_STATUS_INVALID_SHAPE = 3
} SileroVadStatus;

typedef struct SileroVadWeights {
  // const float *stft_weight;

  const float *conv1_weight;
  const float *conv1_bias;
  const float *conv2_weight;
  const float *conv2_bias;
  const float *conv3_weight;
  const float *conv3_bias;
  const float *conv4_weight;
  const float *conv4_bias;

  const float *lstm_weight_ih;
  const float *lstm_weight_hh;
  const float *lstm_bias_ih;
  const float *lstm_bias_hh;

  const float *final_conv_weight;
  const float *final_conv_bias;
} SileroVadWeights;

typedef struct SileroVadConv1d {
  size_t input_channels;
  size_t output_channels;
  size_t kernel_size;
  size_t stride;
  size_t padding;

  const float *weight;
  const float *bias;
  float *packed_weight_t0;
  float *packed_weight_t1;
  float *packed_weight_t2;

  float *padded_input;
  size_t padded_capacity;
  size_t max_input_frames;
} SileroVadConv1d;

typedef struct SileroVadLstmCell {
  size_t input_size;
  size_t hidden_size;

  const float *weight_ih;
  const float *weight_hh;
  const float *bias_ih;
  const float *bias_hh;

  float *packed_weight_ih;
  float *packed_weight_hh;
  float *hidden_state;
  float *cell_state;
  float *gates;
} SileroVadLstmCell;

typedef struct SileroVadModel {
  size_t input_samples;
  size_t stft_frames;
  size_t conv2_frames;
  size_t conv3_frames;
  size_t conv4_frames;

  SileroVadConv1d conv1;
  SileroVadConv1d conv2;
  SileroVadConv1d conv3;
  SileroVadConv1d conv4;
  SileroVadConv1d final_conv;
  SileroVadLstmCell lstm;

  float *padded_audio;
  float *fft_real;
  float *fft_imag;
  float *stft_mag;
  float *conv1_out;
  float *conv2_out;
  float *conv3_out;
  float *conv4_out;
  float *lstm_out;
  float *final_out;
  float *fft_twiddle_cos;
  float *fft_twiddle_sin;
  unsigned short *fft_bitrev;
} SileroVadModel;

/* Computes the output frame count for a 1D convolution with symmetric padding. */
SILERO_VAD_API size_t silero_vad_conv1d_output_frames(size_t input_frames,
                                                      size_t kernel_size,
                                                      size_t stride,
                                                      size_t padding);

/*
 * Initializes a reusable Conv1d context.
 *
 * This is primarily exposed for low-level use. The full model API below is the
 * intended entry point for normal inference.
 */
SILERO_VAD_API SileroVadStatus silero_vad_conv1d_init(SileroVadConv1d *conv,
                                                      size_t input_channels,
                                                      size_t output_channels,
                                                      size_t kernel_size,
                                                      size_t stride,
                                                      size_t padding,
                                                      size_t max_input_frames,
                                                      const float *weight,
                                                      const float *bias);

/* Clears any reusable scratch buffers owned by the convolution context. */
SILERO_VAD_API void silero_vad_conv1d_reset(SileroVadConv1d *conv);
/* Releases memory owned by the convolution context. */
SILERO_VAD_API void silero_vad_conv1d_free(SileroVadConv1d *conv);

/* Runs a single Conv1d forward pass using the initialized context. */
SILERO_VAD_API SileroVadStatus silero_vad_conv1d_forward(SileroVadConv1d *conv,
                                                         const float *input,
                                                         size_t input_frames,
                                                         float *output);

/* Initializes a reusable LSTMCell context with persistent hidden/cell state. */
SILERO_VAD_API SileroVadStatus silero_vad_lstm_cell_init(SileroVadLstmCell *cell,
                                                         size_t input_size,
                                                         size_t hidden_size,
                                                         const float *weight_ih,
                                                         const float *weight_hh,
                                                         const float *bias_ih,
                                                         const float *bias_hh);

/* Resets the hidden state, cell state, and internal gate scratch buffers. */
SILERO_VAD_API void silero_vad_lstm_cell_reset(SileroVadLstmCell *cell);
/* Releases memory owned by the LSTM context. */
SILERO_VAD_API void silero_vad_lstm_cell_free(SileroVadLstmCell *cell);

/* Runs one LSTMCell step and writes the new hidden state to hidden_out. */
SILERO_VAD_API SileroVadStatus silero_vad_lstm_cell_forward(SileroVadLstmCell *cell,
                                                            const float *input,
                                                            float *hidden_out);

/*
 * Initializes a full 16 kHz Silero VAD model instance.
 *
 * The supplied weights must match the layouts expected by SileroVadWeights.
 * For the current streaming-style model path, input_samples is typically 576:
 * 64 samples of left context + 512 new samples.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_init(SileroVadModel *model,
                                                     const SileroVadWeights *weights,
                                                     size_t input_samples);
/* Allocates and initializes a model instance. Returns NULL on failure. */
SILERO_VAD_API SileroVadModel *silero_vad_model_create(const SileroVadWeights *weights,
                                                       size_t input_samples);

/* Resets model state for a new utterance or stream. */
SILERO_VAD_API void silero_vad_model_reset(SileroVadModel *model);
/* Releases model-owned buffers but does not free the model pointer itself. */
SILERO_VAD_API void silero_vad_model_free(SileroVadModel *model);
/* Fully destroys a model created with silero_vad_model_create. */
SILERO_VAD_API void silero_vad_model_destroy(SileroVadModel *model);

/*
 * Runs one model step on a single chunk with left context already included.
 *
 * input must point to input_samples floats configured at init time. The output
 * speech probability is written to speech_probability.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_forward(SileroVadModel *model,
                                                        const float *input,
                                                        float *speech_probability);
/* Returns the number of 512-sample probability steps produced for audio_samples. */
SILERO_VAD_API size_t silero_vad_model_audio_prob_count(size_t audio_samples);
/*
 * Runs full-audio inference using the internal 64-sample context logic.
 *
 * The model is reset at the start of the call. speech_probabilities must have
 * capacity for at least silero_vad_model_audio_prob_count(audio_samples)
 * floats. The number of values written is returned through
 * speech_probabilities_written.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_forward_audio(SileroVadModel *model,
                                                              const float *audio,
                                                              size_t audio_samples,
                                                              float *speech_probabilities,
                                                              size_t speech_probabilities_capacity,
                                                              size_t *speech_probabilities_written);

#ifdef __cplusplus
}
#endif

#endif
