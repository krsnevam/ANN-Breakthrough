// kws_mfcc.h – MFCC feature extractor, a line-by-line port of mfcc() in kws_notebook (Section 1).
// Every setting comes from kws_config.h (written by the notebook), so retraining with different MFCC
// settings only means copying in the new header. Plain C++ with no Arduino calls, so it compiles on a PC for testing.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "kws_config.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace kws {

static const int kBins = KWS_NFFT / 2 + 1;

class Mfcc {
 public:
  // Precompute window, mel filterbank, DCT and FFT tables (done once, in double precision like numpy)
  void begin() {
    for (int n = 0; n < KWS_FRAME_LEN; n++)
      window_[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / KWS_FRAME_LEN));

    double mlo = hz2mel(KWS_F_LO), mhi = hz2mel(KWS_F_HI), hz[KWS_N_MEL + 2];
    for (int i = 0; i < KWS_N_MEL + 2; i++) hz[i] = mel2hz(mlo + (mhi - mlo) * i / (KWS_N_MEL + 1));
    int w = 0;
    for (int m = 0; m < KWS_N_MEL; m++) {
      double lo = hz[m], c = hz[m + 1], hi = hz[m + 2];
      fb_start_[m] = -1; fb_len_[m] = 0; fb_off_[m] = w;
      for (int b = 0; b < kBins; b++) {
        double f = (double)b * KWS_SR / KWS_NFFT;
        double up = (f - lo) / (c - lo), down = (hi - f) / (hi - c);
        double v = fmax(0.0, fmin(up, down));
        if (v > 0) {
          if (fb_start_[m] < 0) fb_start_[m] = b;
          fb_w_[w++] = (float)v; fb_len_[m]++;
        } else if (fb_start_[m] >= 0) break;
      }
      if (fb_start_[m] < 0) fb_start_[m] = 0;
    }
    for (int k = 0; k < KWS_N_MFCC; k++)
      for (int n = 0; n < KWS_N_MEL; n++)
        dct_[k][n] = (float)(cos(M_PI * k * (2 * n + 1) / (2.0 * KWS_N_MEL)) *
                             sqrt((k == 0 ? 1.0 : 2.0) / KWS_N_MEL));
    for (int i = 0; i < KWS_NFFT / 2; i++) {
      cos_[i] = (float)cos(2.0 * M_PI * i / KWS_NFFT);
      sin_[i] = (float)-sin(2.0 * M_PI * i / KWS_NFFT);
    }
    int bits = 0; while ((1 << bits) < KWS_NFFT) bits++;
    for (int i = 0; i < KWS_NFFT; i++) {
      int r = 0; for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      rev_[i] = (uint16_t)r;
    }
  }

  // frame: KWS_FRAME_LEN samples scaled to [-1, 1]. out: KWS_N_MFCC coefficients.
  void compute(const float* frame, float* out) {
    for (int i = 0; i < KWS_NFFT; i++) { re_[i] = 0; im_[i] = 0; }
    for (int n = 0; n < KWS_FRAME_LEN; n++) re_[rev_[n]] = frame[n] * window_[n];   // bit-reversed load
    fft();
    float power[kBins];
    for (int b = 0; b < kBins; b++) power[b] = re_[b] * re_[b] + im_[b] * im_[b];
    float logmel[KWS_N_MEL];
    for (int m = 0; m < KWS_N_MEL; m++) {
      float s = 0;
      for (int j = 0; j < fb_len_[m]; j++) s += fb_w_[fb_off_[m] + j] * power[fb_start_[m] + j];
      logmel[m] = logf(s + 1e-6f);
    }
    for (int k = 0; k < KWS_N_MFCC; k++) {
      float s = 0;
      for (int n = 0; n < KWS_N_MEL; n++) s += dct_[k][n] * logmel[n];
      out[k] = s;
    }
  }

 private:
  static double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
  static double mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

  // In-place iterative radix-2 FFT (input already in bit-reversed order)
  void fft() {
    for (int len = 2; len <= KWS_NFFT; len <<= 1) {
      int half = len >> 1, step = KWS_NFFT / len;
      for (int i = 0; i < KWS_NFFT; i += len)
        for (int j = 0; j < half; j++) {
          float wr = cos_[j * step], wi = sin_[j * step];
          int a = i + j, b = a + half;
          float tr = re_[b] * wr - im_[b] * wi, ti = re_[b] * wi + im_[b] * wr;
          re_[b] = re_[a] - tr; im_[b] = im_[a] - ti;
          re_[a] += tr; im_[a] += ti;
        }
    }
  }

  float window_[KWS_FRAME_LEN];
  float fb_w_[kBins * 2];           // each FFT bin is in at most two triangles
  int16_t fb_start_[KWS_N_MEL], fb_len_[KWS_N_MEL], fb_off_[KWS_N_MEL];
  float dct_[KWS_N_MFCC][KWS_N_MEL];
  float cos_[KWS_NFFT / 2], sin_[KWS_NFFT / 2];
  uint16_t rev_[KWS_NFFT];
  float re_[KWS_NFFT], im_[KWS_NFFT];
};

// Builds the same chirp as chirp() in the notebook, runs the MFCC on it and returns the largest difference
// from the notebook's values. Under ~0.05 means the board's features match training.
inline float self_test(Mfcc& m, float* worst_coeff_out = nullptr) {
#ifdef KWS_HAS_GOLDEN
  static float frame[KWS_FRAME_LEN];
  float out[KWS_N_MFCC], worst = 0;
  const double Tc = (double)KWS_CLIP / KWS_SR;
  for (int g = 0; g < 3; g++) {
    int start = KWS_GOLD_FRAMES[g] * KWS_HOP;
    for (int n = 0; n < KWS_FRAME_LEN; n++) {
      double t = (double)(start + n) / KWS_SR;
      frame[n] = (float)(0.3 * sin(2 * M_PI * (KWS_CHIRP_F0 * t + (KWS_CHIRP_F1 - KWS_CHIRP_F0) / (2 * Tc) * t * t)));
    }
    m.compute(frame, out);
    for (int k = 0; k < KWS_N_MFCC; k++) worst = fmaxf(worst, fabsf(out[k] - KWS_GOLD[g][k]));
  }
  if (worst_coeff_out) *worst_coeff_out = worst;
  return worst;
#else
  (void)m; (void)worst_coeff_out; return -1;
#endif
}

}  // namespace kws
