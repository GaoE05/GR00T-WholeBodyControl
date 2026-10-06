#pragma once
// Position-command boundary guard. Does not certify the robot or limit torque.
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace softsonic {
inline constexpr const char* kGuardVersion = "SOFTSONIC_TARGET_GUARD_V1";
inline constexpr std::array<const char*,29> kHardwareJointNames = {
 "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
 "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
 "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
 "waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint",
 "left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint",
 "left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint",
 "right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"};
inline std::vector<std::string> GuardSplit(const std::string& s) {
 std::vector<std::string> result; std::istringstream input(s); std::string part;
 while (std::getline(input,part,',')) result.push_back(part);
 if (!s.empty() && s.back()==',') result.emplace_back();
 return result;
}
struct JointLimits {
 std::array<double,29> lower{},upper{};
 std::map<std::string,std::string> metadata;
 static JointLimits Load(const std::string& path, const std::string& environment,
                         const std::string& interface, bool disable_crc) {
  if (environment!="hardware" && environment!="simulation") throw std::runtime_error("invalid target environment");
  const bool loopback=interface=="lo" || interface=="lo0" || interface=="127.0.0.1";
  if ((environment=="simulation" && (!loopback || !disable_crc)) ||
      (environment=="hardware" && (loopback || disable_crc)))
   throw std::runtime_error("target environment/interface/CRC mismatch before DDS");
  std::ifstream stream(path); if (!stream) throw std::runtime_error("--joint-limits required: cannot read limits");
  JointLimits result; std::string line; bool rows=false; std::size_t count=0;
  while (std::getline(stream,line)) {
   if (!line.empty() && line.back()=='\r') line.pop_back();
   if (line.empty()) throw std::runtime_error("blank limit line rejected");
   auto fields=GuardSplit(line);
   if (!rows) {
    if (line=="index,name,lower_rad,upper_rad") {rows=true;continue;}
    if (fields.size()!=2 || fields[0].empty() || fields[1].empty() ||
        !result.metadata.emplace(fields[0],fields[1]).second) throw std::runtime_error("invalid limits metadata");
   } else {
    if (count>=29 || fields.size()!=4 || fields[0]!=std::to_string(count) || fields[1]!=kHardwareJointNames[count])
     throw std::runtime_error("limits require exactly 29 hardware-order PR joints");
    std::size_t used=0; result.lower[count]=std::stod(fields[2],&used);
    if (used!=fields[2].size()) throw std::runtime_error("malformed lower limit");
    result.upper[count]=std::stod(fields[3],&used);
    if (used!=fields[3].size() || !std::isfinite(result.lower[count]) || !std::isfinite(result.upper[count]) ||
        result.lower[count]>=result.upper[count]) throw std::runtime_error("invalid joint limit interval");
    ++count;
   }
  }
  const std::array<std::string,8> keys={"schema","purpose","units","coordinates","source","source_sha256","verified_by","hardware_verified"};
  if (count!=29 || result.metadata.size()!=keys.size()) throw std::runtime_error("incomplete limit configuration");
  for (const auto& key:keys) if (!result.metadata.count(key)) throw std::runtime_error("missing limits metadata: "+key);
  if (result.metadata.at("schema")!="softsonic_joint_limits_v1" || result.metadata.at("purpose")!=environment ||
      result.metadata.at("units")!="rad" || result.metadata.at("coordinates")!="hardware_PR")
   throw std::runtime_error("limit schema/purpose/coordinate mismatch");
  const auto& digest=result.metadata.at("source_sha256");
  if (digest.size()!=64 || digest.find_first_not_of("0123456789abcdef")!=std::string::npos)
   throw std::runtime_error("limits require source SHA256");
  if ((environment=="hardware" && (result.metadata.at("hardware_verified")!="true" || result.metadata.at("verified_by")=="UNVERIFIED")) ||
      (environment=="simulation" && result.metadata.at("hardware_verified")!="false"))
   throw std::runtime_error("hardware limits need explicit onsite verification; simulation is not hardware approval");
  return result;
 }
};
struct GuardCommand {
 std::array<float,29> q_target{},dq_target{},tau_ff{},kp{},kd{};
};
inline GuardCommand GuardDamping() {
 GuardCommand c; c.kd.fill(8.0f); return c;
}
enum class GuardStage { Init, ControlPreplay, ControlPlay, ControlHold, Damping };
inline const char* GuardStageName(GuardStage stage) {
 switch(stage) {
 case GuardStage::Init:return "INIT";
 case GuardStage::ControlPreplay:return "CONTROL_PREPLAY";
 case GuardStage::ControlPlay:return "CONTROL_PLAY";
 case GuardStage::ControlHold:return "CONTROL_HOLD";
 default:return "DAMPING";
 }
}
struct GuardViolation { int index=-1; const char* reason=""; };
template<class Command> GuardCommand GuardCopy(const Command& c) {
 GuardCommand out;
 for(std::size_t i=0;i<29;++i) {out.q_target[i]=c.q_target[i];out.dq_target[i]=c.dq_target[i];
  out.tau_ff[i]=c.tau_ff[i];out.kp[i]=c.kp[i];out.kd[i]=c.kd[i];}
 return out;
}
inline GuardViolation ValidateTarget(const GuardCommand& c,const JointLimits& bounds,bool allow_damping) {
 bool damping=allow_damping;
 for(std::size_t i=0;i<29;++i) {
  if (!std::isfinite(c.q_target[i]) || !std::isfinite(c.dq_target[i]) || !std::isfinite(c.tau_ff[i]) ||
      !std::isfinite(c.kp[i]) || !std::isfinite(c.kd[i])) return {static_cast<int>(i),"nonfinite_command"};
  damping=damping && c.q_target[i]==0 && c.dq_target[i]==0 && c.tau_ff[i]==0 && c.kp[i]==0 && c.kd[i]==8;
 }
 if(damping) return {};
 for(std::size_t i=0;i<29;++i) {
  if(c.q_target[i]<bounds.lower[i] || c.q_target[i]>bounds.upper[i]) return {static_cast<int>(i),"q_target_out_of_range"};
  if(c.kp[i]<0 || c.kd[i]<0) return {static_cast<int>(i),"negative_gain"};
 }
 return {};
}
// Queue mutex never covers disk I/O. A full/busy queue drops normal telemetry;
// the first fault has its own reserved slot and is flushed by the logger thread.
class GuardTelemetry {
 public:
 struct Row {GuardCommand command; GuardStage stage; std::uint64_t sequence; bool sent; GuardViolation violation;
  std::int64_t steady_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();};
 GuardTelemetry(const std::string& path,const JointLimits& limits):limits_(limits),file_(path) {
  if(!file_) { WarnUnavailable(); return; }
  file_<<"sequence,stage,sent,fault,index,name,q_target_rad,dq_target_rad_s,tau_ff,kp,kd,lower_rad,upper_rad,reason,dropped_rows,steady_ns\n";
  file_.flush(); if(!file_) { WarnUnavailable(); return; }
  worker_=std::thread([this]{Run();});
 }
 ~GuardTelemetry(){ {std::lock_guard<std::mutex> lock(queue_mutex_); stop_=true;} cv_.notify_one(); if(worker_.joinable())worker_.join(); }
 void Push(const Row& row) {
  if(Failed()) { dropped_.fetch_add(29); return; }
  std::unique_lock<std::mutex> lock(queue_mutex_,std::try_to_lock);
  if(!lock || queue_.size()>=256) {dropped_.fetch_add(29);return;}
  queue_.push_back(row); lock.unlock(); cv_.notify_one();
 }
 // Called once under command transaction lock; this is memory-only, no disk I/O.
 void Fault(const Row& row) {
  if(Failed()) {
   std::cerr<<"[SoftSONIC TARGET FAULT] trace unavailable; reason="<<row.violation.reason
            <<" index="<<row.violation.index<<std::endl;
   return;
  }
  std::lock_guard<std::mutex> lock(queue_mutex_); fault_=row; has_fault_=true; cv_.notify_one();
 }
 std::uint64_t Dropped() const{return dropped_.load();}
 bool Failed() const{return io_failed_.load();}
 private:
 void WarnUnavailable() {
  if(!io_failed_.exchange(true))
   std::cerr<<"[SoftSONIC WARNING] target telemetry unavailable; motor command validation remains active"<<std::endl;
 }
 void Write(const Row& row) {
  if(Failed()) return;
  file_<<std::setprecision(17);
  for(std::size_t i=0;i<29;++i) file_<<row.sequence<<','<<GuardStageName(row.stage)<<','<<row.sent<<','
   <<(row.violation.index>=0)<<','<<i<<','<<kHardwareJointNames[i]<<','<<row.command.q_target[i]<<','
   <<row.command.dq_target[i]<<','<<row.command.tau_ff[i]<<','<<row.command.kp[i]<<','<<row.command.kd[i]<<','
   <<limits_.lower[i]<<','<<limits_.upper[i]<<','<<(row.violation.index==static_cast<int>(i)?row.violation.reason:"")<<','<<dropped_.load()<<','<<row.steady_ns<<'\n';
  if(row.violation.index>=0) file_.flush();
  if(!file_) WarnUnavailable();
 }
 void Run() {
  for(;;) {
   Row row; {std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_.wait(lock,[this]{return stop_||has_fault_||!queue_.empty();});
    if(has_fault_) {row=fault_;has_fault_=false;}
    else if(!queue_.empty()) {row=queue_.front();queue_.pop_front();}
    else if(stop_) break; else continue;
   } Write(row);
  } file_.flush();
 }
 JointLimits limits_; std::ofstream file_; std::thread worker_; std::mutex queue_mutex_; std::condition_variable cv_;
 std::deque<Row> queue_; Row fault_{}; bool has_fault_=false,stop_=false;
 std::atomic<std::uint64_t> dropped_{0}; std::atomic<bool> io_failed_{false};
};
class TargetGuard {
 public:
 TargetGuard(JointLimits limits,const std::string& trace):bounds_(limits),telemetry_(trace,limits){}
 bool Faulted() const{return faulted_.load();}
 void CheckStartup(const GuardCommand& defaults) const {
  const auto fault=ValidateTarget(defaults,bounds_,false);
  if(fault.index>=0) throw std::runtime_error(std::string("default target invalid before DDS: ")+kHardwareJointNames[fault.index]+" "+fault.reason);
 }
 template<class Command,class Accepted,class Damped>
 bool Commit(const Command& command,GuardStage stage,Accepted accepted,Damped damped) {
  std::lock_guard<std::mutex> lock(transaction_);
  if(faulted_.load()) return false;
  ++sequence_;
  const auto c=GuardCopy(command);
  const auto fault=ValidateTarget(c,bounds_,false);
  if(fault.index>=0) {faulted_.store(true); damped(); telemetry_.Fault({c,stage,sequence_,false,fault});return false;}
  stage_=stage;accepted();telemetry_.Push({c,stage,sequence_,false,{}});return true;
 }
 template<class Get,class Publish,class Damped>
 void PublishLatest(Get get,Publish publish,Damped damped) {
  std::lock_guard<std::mutex> lock(transaction_);
  auto command=get(); if(!command && !faulted_.load()) return;
  GuardCommand c=command?GuardCopy(*command):GuardDamping();
  auto fault=ValidateTarget(c,bounds_,true);
  if(fault.index>=0 && !faulted_.load()) {
   faulted_.store(true);damped();
   // Log the rejected exact command; publish damping before requesting disk I/O.
   publish(GuardDamping());telemetry_.Fault({c,stage_,sequence_,false,fault});
   telemetry_.Push({GuardDamping(),GuardStage::Damping,sequence_,true,{}});return;
  }
  if(faulted_.load()) c=GuardDamping();
  publish(c);telemetry_.Push({c,faulted_.load()?GuardStage::Damping:stage_,sequence_,true,{}});
 }
 template<class Damped> void Damp(Damped damped) {
  std::lock_guard<std::mutex> lock(transaction_);faulted_.store(true);stage_=GuardStage::Damping;damped();
 }
 private:
 JointLimits bounds_; GuardTelemetry telemetry_; std::atomic<bool> faulted_{false}; std::mutex transaction_;
 std::uint64_t sequence_=0; GuardStage stage_=GuardStage::Init;
};
} // namespace softsonic
