// Offline only: no G1Deploy, DDS, MotionSwitcher, publisher or robot commands.
#include "../include/encoder.hpp"
#include "../include/control_policy.hpp"
#include "../include/motion_data_reader.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<float> Read(const std::filesystem::path& path, std::size_t values) {
  if (std::filesystem::file_size(path) != values*sizeof(float)) throw std::runtime_error("fixture size mismatch");
  std::vector<float> result(values);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(result.data()), values*sizeof(float))) throw std::runtime_error("fixture read failed");
  for (const auto v:result) if (!std::isfinite(v)) throw std::runtime_error("nonfinite fixture");
  return result;
}

template<class Engine, class Input, class Output, class Run>
std::array<double,3> Check(Engine& engine, const std::vector<float>& inputs,
  const std::vector<float>& expected, std::size_t rows, std::size_t in_size,
  std::size_t out_size, Input input, Output output, Run run, double tolerance) {
  double maximum_error=0, maximum_ms=0, total_ms=0;
  for (std::size_t row=0;row<rows;++row) {
    auto& destination=input(engine);
    if(destination.size()!=in_size) throw std::runtime_error("engine input dimension mismatch");
    std::copy_n(inputs.data()+row*in_size,in_size,destination.begin());
    const auto start=std::chrono::steady_clock::now();
    if(!run(engine)) throw std::runtime_error("native inference failed");
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    total_ms+=ms;maximum_ms=std::max(maximum_ms,ms);
    const auto& values=output(engine);
    if(values.size()!=out_size) throw std::runtime_error("engine output dimension mismatch");
    for(std::size_t i=0;i<out_size;++i) {
      if(!std::isfinite(values[i])) throw std::runtime_error("nonfinite native output");
      maximum_error=std::max(maximum_error,std::abs(static_cast<double>(values[i])-expected[row*out_size+i]));
    }
  }
  if(maximum_error>tolerance) throw std::runtime_error("TensorRT versus source golden mismatch: "+std::to_string(maximum_error));
  return {maximum_error,total_ms/rows,maximum_ms};
}

int main(int argc,char** argv) {
  if(argc!=2) {std::cerr<<"Usage: g1_softsonic_check <assets-directory> (offline; no DDS)\n";return 2;}
  try {
    const std::filesystem::path assets(argv[1]), fixtures=assets/"golden_native";
    for(const auto name : {"stand", "walking"}) {
      MotionDataReader reader;
      if(!reader.ReadFromCSV((assets/"motion_sets"/name).string()) || reader.motions.size()!=1) {
        throw std::runtime_error("reference directory must contain exactly one readable motion");
      }
      const auto& motion=reader.motions.front();
      const int expected_frames=std::string(name)=="stand"?847:600;
      if(motion->timesteps!=expected_frames || motion->GetNumJoints()!=29) {
        throw std::runtime_error("native reference frame/joint dimensions mismatch");
      }
      for(int frame=0;frame<motion->timesteps;++frame) {
        for(int j=0;j<29;++j) {
          if(!std::isfinite(motion->JointPositions(frame)[j]) || !std::isfinite(motion->JointVelocities(frame)[j])) {
            throw std::runtime_error("nonfinite reference joint data");
          }
        }
      }
    }
    EncoderEngine encoder; PolicyEngine decoder;
    if(!encoder.Initialize((assets/"sonic_original_g1_encoder.onnx").string(),false) ||
       !decoder.Initialize((assets/"sonic_original_decoder.onnx").string(),false)) {
      throw std::runtime_error("native engine initialization failed");
    }
    if(!encoder.CaptureGraph() || !decoder.CaptureGraph()) throw std::runtime_error("CUDA graph capture failed");
    auto enc=Check(encoder,Read(fixtures/"encoder_input.bin",64*640),Read(fixtures/"encoder_expected.bin",64*64),
      64,640,64,[](auto& e)->auto&{return e.GetInputBuffer();},
      [](auto& e)->auto&{return e.GetTokenBuffer();},[](auto& e){return e.Encode();},2e-5);
    auto dec=Check(decoder,Read(fixtures/"decoder_input.bin",192*994),Read(fixtures/"decoder_expected.bin",192*29),
      192,994,29,[](auto& e)->auto&{return e.GetInputBuffer();},
      [](auto& e)->auto&{return e.GetActionBuffer();},[](auto& e){return e.Infer();},1e-4);
    if(enc[2]+dec[2]>=20.0) throw std::runtime_error("native encoder+decoder worst observed cost exceeds 20ms before residual");
    std::cout << "SOFTSONIC_NATIVE_CHECK_PASS encoder_maxerr="<<enc[0]<<" decoder_maxerr="<<dec[0]
      <<" encoder_mean_ms="<<enc[1]<<" encoder_max_ms="<<enc[2]
      <<" decoder_mean_ms="<<dec[1]<<" decoder_max_ms="<<dec[2]<<std::endl;
    return 0;
  } catch(const std::exception& e) {std::cerr<<"SOFTSONIC_NATIVE_CHECK_FAIL: "<<e.what()<<std::endl;return 1;}
}
