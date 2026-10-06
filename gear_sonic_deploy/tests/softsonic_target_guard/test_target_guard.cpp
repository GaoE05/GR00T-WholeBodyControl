#include "../../src/g1/g1_deploy_onnx_ref/include/softsonic_target_guard.hpp"
#include <chrono>
#include <filesystem>
#include <cassert>
#include <future>
#include <iostream>
#include <limits>
#include <memory>

int main() {
  using namespace softsonic;
  JointLimits limits;
  limits.lower.fill(-2.0); limits.upper.fill(2.0);
  limits.lower[13] = -.52; limits.upper[13] = .52;
  limits.lower[14] = -.52; limits.upper[14] = .52;
  GuardCommand active;
  active.kp.fill(20.f); active.kd.fill(1.f);
  assert(ValidateTarget(active, limits, false).index == -1);
  auto unsafe = active; unsafe.q_target[13] = 1.3157319f;
  assert(ValidateTarget(unsafe, limits, false).index == 13);
  assert(unsafe.q_target[13] == 1.3157319f);
  auto edge = active;
  edge.q_target[13] = static_cast<float>(.52);
  assert(ValidateTarget(edge, limits, false).index == -1);
  edge.q_target[13] = std::nextafter(edge.q_target[13], std::numeric_limits<float>::infinity());
  assert(ValidateTarget(edge, limits, false).index == 13);
  for (int field=0; field<5; ++field) {
    auto nonfinite=active;
    std::array<float,29>* fields[]={&nonfinite.q_target,&nonfinite.dq_target,&nonfinite.tau_ff,&nonfinite.kp,&nonfinite.kd};
    (*fields[field])[7]=std::numeric_limits<float>::quiet_NaN();
    assert(ValidateTarget(nonfinite, limits, true).index==7);
  }
  auto knee_limits=limits; knee_limits.lower[3]=.1;
  assert(ValidateTarget(GuardDamping(), knee_limits, true).index==-1);
  assert(ValidateTarget(GuardDamping(), knee_limits, false).index==3);
  auto forged=GuardDamping(); forged.tau_ff[3]=1.f;
  assert(ValidateTarget(forged, knee_limits, true).index==3);

  const auto trace_path=std::filesystem::temp_directory_path() / ("softsonic_guard_review_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".csv");
  const auto rejected_path=trace_path.string()+".rejected";
  {
    TargetGuard guard(limits,rejected_path);bool accepted=false,damped=false;
    assert(!guard.Commit(unsafe,GuardStage::Init,[&]{accepted=true;},[&]{damped=true;}));
    assert(!accepted && damped && guard.Faulted());
    assert(!guard.Commit(active,GuardStage::ControlPlay,[&]{accepted=true;},[&]{damped=true;}));
    assert(!accepted);
  }
  std::filesystem::remove(rejected_path);
  std::atomic<bool> committed_after_fault{false};
  std::atomic<int> damping_commits{0};
  std::vector<GuardCommand> sent;
  {
    TargetGuard guard(limits, trace_path.string());
    std::shared_ptr<GuardCommand> buffer;
    auto damp=[&]{buffer=std::make_shared<GuardCommand>(GuardDamping()); ++damping_commits;};
    assert(guard.Commit(active, GuardStage::Init, [&]{buffer=std::make_shared<GuardCommand>(active);}, damp));
    buffer=std::make_shared<GuardCommand>(unsafe);
    std::promise<void> writer_entered, allow_writer, control_started;
    auto release=allow_writer.get_future();
    std::thread writer([&]{
      guard.PublishLatest([&]{writer_entered.set_value(); release.wait(); return buffer;},
                          [&](const GuardCommand& command){sent.push_back(command);}, damp);
    });
    writer_entered.get_future().wait();
    std::thread control([&]{
      control_started.set_value();
      committed_after_fault=guard.Commit(active, GuardStage::ControlPlay,
        [&]{buffer=std::make_shared<GuardCommand>(active);}, damp);
    });
    control_started.get_future().wait();
    allow_writer.set_value();
    writer.join(); control.join();
    assert(guard.Faulted()); assert(!committed_after_fault);
    assert(damping_commits==1);
    assert(buffer->kp[13]==0.f && buffer->kd[13]==8.f);
    for(int i=0;i<100;++i) {
      assert(!guard.Commit(active, GuardStage::ControlPlay,
        [&]{buffer=std::make_shared<GuardCommand>(active);}, damp));
      guard.PublishLatest([&]{return buffer;}, [&](const GuardCommand& command){sent.push_back(command);}, damp);
    }
    guard.Damp(damp);
    assert(guard.Faulted());
  }
  assert(sent.size()==101);
  for(const auto& command:sent) for(int i=0;i<29;++i) {
    assert(command.q_target[i]==0.f && command.dq_target[i]==0.f && command.tau_ff[i]==0.f);
    assert(command.kp[i]==0.f && command.kd[i]==8.f);
  }
  std::ifstream trace(trace_path.string());
  std::string line; bool recorded_fault=false;
  while(std::getline(trace,line)) {
    const auto fields=GuardSplit(line);
    if(fields.size()==16 && fields[13]=="q_target_out_of_range") {
      recorded_fault=true;
      assert(fields[4]=="13" && fields[5]=="waist_roll_joint");
      assert(std::stod(fields[6])==static_cast<double>(unsafe.q_target[13]));
      assert(fields[2]=="0" && fields[3]=="1");
    }
  }
  assert(recorded_fault);
  trace.close();
  std::filesystem::remove(trace_path);
  std::cout << "Independent target guard checks passed: range boundaries, five finite fields, exact damping bypass, writer/control race, irreversible latch, exact rejected target telemetry.\n";
}
