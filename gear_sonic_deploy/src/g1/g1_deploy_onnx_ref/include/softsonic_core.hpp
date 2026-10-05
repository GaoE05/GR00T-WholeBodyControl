#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace softsonic {
constexpr std::size_t kToken = 64;
constexpr std::size_t kHistory = 930;
constexpr std::size_t kPolicy = kToken + kHistory;
enum class Mode { Off, Shadow, On };

inline Mode ParseMode(const std::string& mode) {
  if (mode == "off") return Mode::Off;
  if (mode == "shadow") return Mode::Shadow;
  if (mode == "on") return Mode::On;
  throw std::invalid_argument("SoftSONIC mode must be off, shadow or on");
}

template <typename Vector>
void RequireFinite(const Vector& values, const char* what) {
  for (const auto value : values) {
    if (!std::isfinite(static_cast<double>(value))) {
      throw std::runtime_error(std::string("nonfinite ") + what);
    }
  }
}

// Inspect names, dimensions AND offsets: 994 dimensions alone is insufficient.
template <typename ObservationList>
void ValidateObservationLayout(const ObservationList& observations) {
  const std::array<const char*, 6> names = {
    "token_state", "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"};
  const std::array<std::size_t, 6> dimensions = {64, 30, 290, 290, 290, 30};
  if (observations.size() != names.size()) {
    throw std::runtime_error("SoftSONIC requires the exact six-term observation layout");
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto& term = observations[i];
    if (term.name != names[i] || term.dimension != dimensions[i] || term.offset != offset) {
      throw std::runtime_error("SoftSONIC observation name/order/dimension/offset mismatch");
    }
    offset += dimensions[i];
  }
}

// Only the successfully gathered CONTROL ticks count. No old history is used
// until ten new physical state samples replace native SONIC's zero padding.
class ContextGate {
 public:
  bool Observe(std::uint64_t tick, std::uintptr_t motion, int frame, bool play, int encoder_mode,
               bool explicit_reset = false) {
    if (encoder_mode != 0 || motion == 0 || frame < 0) {
      throw std::runtime_error("this bundle only supports local G1 reference encoder mode 0");
    }
    if (seen_ && tick <= last_tick_) throw std::runtime_error("duplicate/out-of-order control tick");
    const bool discontinuity = !seen_ || explicit_reset || (seen_ && tick != last_tick_ + 1)
      || motion != last_motion_ || encoder_mode != last_mode_
      || play != last_play_ || frame < last_frame_ || frame > last_frame_ + 1;
    samples_ = discontinuity ? 1 : samples_ + 1;
    seen_ = true;
    last_tick_ = tick; last_motion_ = motion; last_frame_ = frame;
    last_play_ = play; last_mode_ = encoder_mode;
    return samples_ >= 10;
  }
  std::size_t Samples() const { return samples_; }
 private:
  bool seen_ = false, last_play_ = false;
  std::uint64_t last_tick_ = 0;
  std::uintptr_t last_motion_ = 0;
  int last_frame_ = 0, last_mode_ = 0;
  std::size_t samples_ = 0;
};

inline void ValidatePolicyInput(const std::vector<double>& observation) {
  if (observation.size() != kPolicy) throw std::runtime_error("expected token64 + state history930");
  RequireFinite(observation, "SONIC observation");
  for (std::size_t i = 0; i < kToken; ++i) {
    if (std::abs(observation[i]) > 1.25) throw std::runtime_error("invalid nominal motion token");
  }
}

inline void PackActorInput(const std::vector<double>& observation, std::vector<float>& actor) {
  ValidatePolicyInput(observation);
  if (actor.size() != kHistory && actor.size() != kPolicy) {
    throw std::runtime_error("actor input must be 930 or history930+nominal64");
  }
  for (std::size_t i = 0; i < kHistory; ++i) actor[i] = static_cast<float>(observation[kToken + i]);
  if (actor.size() == kPolicy) {
    for (std::size_t i = 0; i < kToken; ++i) actor[kHistory + i] = static_cast<float>(observation[i]);
  }
  RequireFinite(actor, "float32 actor input");
}

inline void Fuse(std::vector<double>& observation, const std::array<float, kToken>& delta,
                 Mode mode, double gain) {
  ValidatePolicyInput(observation);
  if (!std::isfinite(gain) || gain < 0 || gain > 1) throw std::runtime_error("gain must be in [0,1]");
  RequireFinite(delta, "residual delta");
  for (const float value : delta) {
    if (std::abs(value) > 0.50001f) throw std::runtime_error("model delta exceeds trained +/-0.5 bound");
  }
  // Shadow and gain=0 preserve the original double values exactly (no cast).
  if (mode == Mode::On && gain != 0) {
    for (std::size_t i = 0; i < kToken; ++i) observation[i] += gain * static_cast<double>(delta[i]);
  }
  // Do NOT bound fused tokens to +/-1.25, requantize, or modify state/hands.
}
}  // namespace softsonic
