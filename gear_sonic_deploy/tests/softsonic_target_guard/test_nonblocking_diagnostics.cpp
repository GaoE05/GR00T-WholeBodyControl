#include "../../src/g1/g1_deploy_onnx_ref/include/softsonic_target_guard.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <memory>
int main() {
 using namespace softsonic;
 JointLimits limits; limits.lower.fill(-2);limits.upper.fill(2);
 GuardCommand valid;valid.kp.fill(20);valid.kd.fill(1);
 auto exercise=[&](const std::string& path){
  TargetGuard guard(limits,path);
  std::shared_ptr<GuardCommand> buffer;int accepted=0,active_sent=0,damping_sent=0;
  auto damp=[&]{buffer=std::make_shared<GuardCommand>(GuardDamping());};
  assert(guard.Commit(valid,GuardStage::Init,[&]{++accepted;buffer=std::make_shared<GuardCommand>(valid);},damp));
  assert(!guard.Faulted());
  guard.PublishLatest([&]{return buffer;},[&](const GuardCommand& c){assert(c.kp[0]==20);++active_sent;},damp);
  auto invalid=valid;invalid.q_target[14]=3;
  assert(!guard.Commit(invalid,GuardStage::ControlPlay,[&]{++accepted;},damp));
  assert(guard.Faulted());assert(accepted==1 && active_sent==1);
  guard.PublishLatest([&]{return buffer;},[&](const GuardCommand& c){assert(c.kp[0]==0 && c.kd[0]==8);++damping_sent;},damp);
  assert(damping_sent==1);
 };
 auto missing=std::filesystem::temp_directory_path()/("softsonic_missing_log_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))/"target.csv";
 exercise(missing.string());
 if(std::filesystem::exists("/dev/full")) exercise("/dev/full");
 std::cout<<"Diagnostic IO failure does not block valid commands; invalid motor targets still latch damping.\n";
}
