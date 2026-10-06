#pragma once
#include "softsonic_core.hpp"
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <fstream>
#include <memory>

namespace softsonic {
struct Config {
  std::string joint_limits;
  std::string target_environment = "hardware";
  std::string target_trace;
  std::string actor;
  std::string mode = "off";
  std::string trace;
  double gain = 1.0;
  double deadline_ms = 10.0;
};

class ResidualEngine {
 public:
  explicit ResidualEngine(const Config& config)
    : config_(config), mode_(ParseMode(config.mode)), env_(ORT_LOGGING_LEVEL_WARNING, "SoftSONIC") {
    if (!std::isfinite(config.gain) || config.gain < 0 || config.gain > 1 ||
        !std::isfinite(config.deadline_ms) || config.deadline_ms <= 0 || config.deadline_ms >= 20) {
      throw std::invalid_argument("invalid SoftSONIC gain/deadline (deadline must be below 20ms)");
    }
    if (mode_ != Mode::Off) {
      if (config.actor.empty()) throw std::invalid_argument("--softsonic-model required for shadow/on");
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      session_ = std::make_unique<Ort::Session>(env_, config.actor.c_str(), options);
      Ort::AllocatorWithDefaultOptions allocator;
      if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
        throw std::runtime_error("residual ONNX requires one input and one output");
      }
      input_name_ = session_->GetInputNameAllocated(0, allocator).get();
      output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
      const auto input_type = session_->GetInputTypeInfo(0);
      const auto output_type = session_->GetOutputTypeInfo(0);
      const auto input_info = input_type.GetTensorTypeAndShapeInfo();
      const auto output_info = output_type.GetTensorTypeAndShapeInfo();
      const auto in_shape = input_info.GetShape(); const auto out_shape = output_info.GetShape();
      if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          in_shape.size() != 2 || in_shape[0] != 1 ||
          (in_shape[1] != 930 && in_shape[1] != 994) || out_shape != std::vector<int64_t>({1,64})) {
        throw std::runtime_error("residual ONNX needs static float32 [1,930/994] -> [1,64]");
      }
      actor_.resize(static_cast<std::size_t>(in_shape[1]), 0.0f);
      input_shape_ = {1, in_shape[1]};
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      input_ = Ort::Value::CreateTensor<float>(memory, actor_.data(), actor_.size(), input_shape_.data(), 2);
      output_ = Ort::Value::CreateTensor<float>(memory, delta_.data(), delta_.size(), output_shape_.data(), 2);
      // Allocate/optimize outside the physical control loop; these are not histories.
      for (int i = 0; i < 5; ++i) Run();
    }
    if (!config.trace.empty()) {
      trace_.open(config.trace, std::ios::out | std::ios::app);
      if (!trace_) throw std::runtime_error("cannot open SoftSONIC diagnostic trace");
      trace_ << "tick,frame,warmup_samples,mode,gain,inference_ms,delta_l2,delta_max\n";
    }
  }

  void Apply(std::vector<double>& observation, std::uint64_t tick,
             std::uintptr_t motion, int frame, bool play, int encoder_mode, bool explicit_reset = false) {
    ValidatePolicyInput(observation);
    const bool ready = gate_.Observe(tick, motion, frame, play, encoder_mode, explicit_reset);
    double elapsed = 0.0;
    delta_.fill(0.0f);
    if (ready && mode_ != Mode::Off) {
      PackActorInput(observation, actor_);
      const auto start = std::chrono::steady_clock::now();
      Run();
      elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if (elapsed > config_.deadline_ms) throw std::runtime_error("residual inference exceeded time budget");
      Fuse(observation, delta_, mode_, config_.gain);
    }
    // Buffered std::ofstream: no per-tick flush; SONIC logs full physical states.
    if (trace_ && (tick % 50 == 0 || gate_.Samples() == 10)) {
      double l2 = 0, maximum = 0;
      for (const auto d : delta_) { l2 += d*d; maximum = std::max(maximum, std::abs(static_cast<double>(d))); }
      trace_ << tick << ',' << frame << ',' << gate_.Samples() << ',' << config_.mode << ','
             << config_.gain << ',' << elapsed << ',' << std::sqrt(l2) << ',' << maximum << '\n';
    }
  }
 private:
  void Run() {
    const char* inputs[] = {input_name_.c_str()}; const char* outputs[] = {output_name_.c_str()};
    session_->Run(Ort::RunOptions{nullptr}, inputs, &input_, 1, outputs, &output_, 1);
  }
  Config config_; Mode mode_; ContextGate gate_;
  Ort::Env env_; std::unique_ptr<Ort::Session> session_;
  std::string input_name_, output_name_;
  std::vector<float> actor_;
  std::array<float, kToken> delta_{};
  std::array<int64_t, 2> input_shape_{};
  std::array<int64_t, 2> output_shape_{1,64};
  Ort::Value input_{nullptr}, output_{nullptr};
  std::ofstream trace_;
};
}  // namespace softsonic
