#include "../../src/g1/g1_deploy_onnx_ref/include/softsonic_target_guard.hpp"
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>

int main() {
 using namespace softsonic;
 const auto path=std::filesystem::temp_directory_path()/("guard_limits_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".csv");
 const auto meta=std::string("schema,softsonic_joint_limits_v1\npurpose,simulation\nunits,rad\ncoordinates,hardware_PR\nsource,SYNTHETIC_TEST_ONLY\nsource_sha256,")+std::string(64,'0')+"\nverified_by,UNVERIFIED\nhardware_verified,false\nindex,name,lower_rad,upper_rad\n";
 std::string valid=meta;
 for(int i=0;i<29;++i) valid+=std::to_string(i)+","+kHardwareJointNames[i]+",-2,2\n";
 auto write=[&](std::string text){std::ofstream file(path);file<<text;};
 auto rejects=[&](std::string text,std::string env="simulation",std::string nic="lo",bool crc=true) {
  write(text);bool threw=false;try{JointLimits::Load(path.string(),env,nic,crc);}catch(const std::exception&){threw=true;}assert(threw);
 };
 write(valid);auto bounds=JointLimits::Load(path.string(),"simulation","lo",true);
 assert(bounds.lower[28]==-2 && bounds.upper[28]==2);
 rejects(valid,"hardware","eth0",false); // simulation cannot authorize hardware
 rejects(valid,"simulation","eth0",true); // CRC flag alone doesn't authorize sim
 rejects(valid,"simulation","lo",false);
 rejects(valid,"unknown","lo",true);
 rejects(valid.substr(0,valid.rfind("28,"))); // missing joint
 rejects(valid+"29,extra_joint,-2,2\n");
 rejects("schema,softsonic_joint_limits_v1\n"+valid); // duplicate metadata
 auto wrong=valid;wrong.replace(wrong.find("0,left_hip_pitch_joint"),1,"1");rejects(wrong);
 wrong=valid;wrong.replace(wrong.find("-2,2"),4,"nan,2");rejects(wrong);
 wrong=valid;wrong.replace(wrong.find("-2,2"),4,"2,2");rejects(wrong);
 wrong=valid;wrong.replace(wrong.find("-2,2"),4,"-2,2oops");rejects(wrong);
 wrong=valid;wrong.replace(wrong.find("hardware_PR"),11,"policy_order");rejects(wrong);
 // Only a separate, explicit, provenance-confirmed hardware config is accepted.
 auto hardware=valid;hardware.replace(hardware.find("purpose,simulation"),18,"purpose,hardware");
 hardware.replace(hardware.find("hardware_verified,false"),23,"hardware_verified,true");
 rejects(hardware,"hardware","eth0",false); // still UNVERIFIED
 hardware.replace(hardware.find("UNVERIFIED"),10,"SYNTHETIC_TEST_REVIEWER");
 write(hardware);JointLimits::Load(path.string(),"hardware","eth0",false);
 rejects(hardware,"hardware","lo",false);rejects(hardware,"hardware","eth0",true);
 GuardCommand defaults;defaults.kp.fill(10.f);defaults.kd.fill(1.f);defaults.q_target[13]=3.f;
 const auto trace=path.string()+".trace";
 {TargetGuard guard(bounds,trace);bool threw=false;try{guard.CheckStartup(defaults);}catch(const std::exception&){threw=true;}assert(threw);}
 std::filesystem::remove(path);std::filesystem::remove(trace);
 std::cout<<"Limits configuration tests passed: identity/order/count/finite intervals/provenance/environment/CRC/startup.\n";
}
