// VVRL Stage 1: lightweight, header-only picture-QP experiment bridge.
// Target: Fraunhofer HHI VVenC v1.14.0 commit 9428ea8636ae7f443ecde89999d16b2dfc421524.
// Research instrumentation. NOT a VVenC upstream public API.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vvrl {
struct Observation {
  int poc = 0;
  int native_qp = 0;
  int qp_min = 0;
  int qp_max = 63;
  int frame_type = 0;       // 0=I,1=P,2=B, derived using SliceType constants
  int temporal_layer = 0;
  int gop_position = 0;     // poc % configured GOP size; check irregular GOP separately
  int gop_size = 0;
  int picture_target_bits = 0;
  int ctu_columns = 0;
  int ctu_rows = 0;
  int source_width = 0;
  int source_height = 0;
  int prev_qp = 0;
  int prev_feedback_poc = -1;
  bool prev_valid = false;
  double target_bpp = 0.;
  double budget_error = 0.;
  double prev_bit_ratio = 0.;
  double spatial_activity = 0.;
  double temporal_activity = 0.;
  bool spatial_valid = false;
  bool temporal_valid = false;
};

inline void require_finite(double v, const char* field) {
  if (!std::isfinite(v)) throw std::runtime_error(std::string("VVRL non-finite ") + field);
}
inline int parse_int(const std::string& text, const char* field) {
  if (text.empty()) throw std::runtime_error(std::string("VVRL empty ") + field);
  size_t n = 0;
  const long long value = std::stoll(text, &n, 10);
  if (n != text.size() || value < -100000000 || value > 100000000)
    throw std::runtime_error(std::string("VVRL invalid integer ") + field);
  return static_cast<int>(value);
}
inline int clip_int(int x, int a, int b) { return std::max(a, std::min(x,b)); }

class Controller {
  std::string mode_;
  std::string log_path_;
  int constant_delta_ = 0;
  int prev_qp_ = 0;
  int prev_feedback_poc_ = -1;
  double prev_ratio_ = 0.;
  bool prev_valid_ = false;
  std::map<int,std::map<int,int>> map_file_; // poc -> (CTU raster index -> delta)
  std::map<int,std::vector<int>> map_stdin_; // poc -> raster-ordered deltas

  static std::string env(const char* k) {
    const char* val=std::getenv(k); return val ? val : "";
  }
  void append(const std::string& line) const {
    if (log_path_.empty()) return;
    std::ofstream f(log_path_, std::ios::app);
    if (!f) throw std::runtime_error("VVRL cannot append VVCRL_LOG");
    f << line << '\n';
    if (!f) throw std::runtime_error("VVRL failed to write VVCRL_LOG");
  }
  static std::vector<int> parse_map_values(const std::string& csv, int expected) {
    std::vector<int> out; std::istringstream ss(csv); std::string token;
    while (std::getline(ss,token,',')) {
      int d=parse_int(token,"map delta");
      if(d < -2 || d > 2) throw std::runtime_error("VVRL map delta outside [-2,2]");
      out.push_back(d);
    }
    if (static_cast<int>(out.size()) != expected)
      throw std::runtime_error("VVRL map shape mismatch");
    return out;
  }
  std::string json_obs(const Observation& o) const {
    require_finite(o.target_bpp,"target_bpp");
    require_finite(o.budget_error,"budget_error");
    require_finite(o.prev_bit_ratio,"prev_bit_ratio");
    if(o.spatial_valid) require_finite(o.spatial_activity,"spatial_activity");
    if(o.temporal_valid) require_finite(o.temporal_activity,"temporal_activity");
    std::ostringstream ss; ss << std::setprecision(12);
    ss << "{\"schema\":\"vvrl.v1\",\"kind\":\"observation\",\"poc\":"<<o.poc
       <<",\"native_qp\":"<<o.native_qp <<",\"qp_min\":"<<o.qp_min
       <<",\"qp_max\":"<<o.qp_max
       <<",\"frame_type\":"<<o.frame_type <<",\"temporal_layer\":"<<o.temporal_layer
       <<",\"gop_position\":"<<o.gop_position <<",\"gop_size\":"<<o.gop_size
       <<",\"target_bpp\":"<<o.target_bpp <<",\"budget_error\":"<<o.budget_error
       <<",\"prev_qp\":"<<o.prev_qp <<",\"prev_bit_ratio\":"<<o.prev_bit_ratio
       <<",\"prev_valid\":"<<(o.prev_valid?"true":"false")
       <<",\"prev_feedback_poc\":"<<o.prev_feedback_poc
       <<",\"picture_target_bits\":"<<o.picture_target_bits
       <<",\"ctu_columns\":"<<o.ctu_columns <<",\"ctu_rows\":"<<o.ctu_rows
       <<",\"source_width\":"<<o.source_width <<",\"source_height\":"<<o.source_height
       <<",\"spatial_activity\":"<<(o.spatial_valid?o.spatial_activity:0.)
       <<",\"temporal_activity\":"<<(o.temporal_valid?o.temporal_activity:0.)
       <<",\"spatial_valid\":"<<(o.spatial_valid?"true":"false")
       <<",\"temporal_valid\":"<<(o.temporal_valid?"true":"false") <<"}";
    return ss.str();
  }
  void read_action( const Observation& o, int& delta ) {
    std::string msg = "VVCRL_OBS\t"+json_obs(o);
    std::puts(msg.c_str()); std::fflush(stdout);
    char line[65536];
    if (!std::fgets(line,sizeof(line),stdin)) throw std::runtime_error("VVRL stdin EOF before action");
    std::istringstream in(line); std::string cmd,extra; int poc=0;
    if (mode_=="stdin") {
      if (!(in>>cmd>>poc>>delta) || cmd!="ACT" || poc!=o.poc || (in>>extra))
        throw std::runtime_error("VVRL expected ACT <poc> <delta>");
      if (delta < -2 || delta > 2) throw std::runtime_error("VVRL action outside [-2,2]");
    } else {
      if (!(in>>cmd>>poc>>extra) || cmd!="MAP" || poc!=o.poc)
        throw std::runtime_error("VVRL expected MAP <poc> <csv>");
      std::string junk; if(in>>junk) throw std::runtime_error("VVRL map extra tokens");
      map_stdin_[o.poc]=parse_map_values(extra,o.ctu_columns*o.ctu_rows);
      delta=0;
    }
  }
public:
  Controller(): mode_(env("VVCRL_MODE")),log_path_(env("VVCRL_LOG")) {
    if (mode_.empty()) mode_="off";
    if (mode_!="off" && mode_!="zero" && mode_!="constant" && mode_!="stdin" &&
        mode_!="map-checker" && mode_!="map-file" && mode_!="stdin-map")
      throw std::runtime_error("VVRL invalid VVCRL_MODE");
    if(mode_=="constant" || mode_=="map-checker") {
      constant_delta_=parse_int(env("VVCRL_DELTA"),"VVCRL_DELTA");
      if(constant_delta_<-2 || constant_delta_>2) throw std::runtime_error("VVRL DELTA outside [-2,2]");
    }
    if(mode_=="map-file") {
      std::ifstream f(env("VVCRL_MAP"));
      if(!f) throw std::runtime_error("VVRL cannot open VVCRL_MAP");
      std::string line; int line_n=0;
      while(std::getline(f,line)) {
        ++line_n; if(line.empty() || line[0]=='#') continue;
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream in(line); int poc,ctu,delta; std::string extra;
        if(!(in>>poc>>ctu>>delta) || (in>>extra) || poc<0 || ctu<0 || delta<-2 || delta>2)
          throw std::runtime_error("VVRL map-file format/limit error on line "+std::to_string(line_n));
        auto& row=map_file_[poc];
        if(row.find(ctu)!=row.end()) throw std::runtime_error("VVRL duplicate CTU map index");
        row[ctu]=delta;
      }
    }
  }
  bool enabled() const { return mode_!="off"; }
  bool map_mode() const { return mode_=="map-checker" || mode_=="map-file" || mode_=="stdin-map"; }
  void fill_history(Observation& o) const {o.prev_qp=prev_qp_;o.prev_bit_ratio=prev_ratio_;o.prev_valid=prev_valid_;o.prev_feedback_poc=prev_valid_?prev_feedback_poc_:-1;}
  int action(const Observation& o) {
    if(!enabled()) return 0;
    if(o.qp_min>o.qp_max || o.native_qp<o.qp_min || o.native_qp>o.qp_max)
      throw std::runtime_error("VVRL invalid native QP bounds");
    if(o.ctu_columns<=0 || o.ctu_rows<=0) throw std::runtime_error("VVRL invalid CTU grid");
    if(mode_=="map-file") {
      const int count=o.ctu_columns*o.ctu_rows;
      const auto it=map_file_.find(o.poc);
      if(it!=map_file_.end())
        for(const auto& v : it->second)
          if(v.first>=count) throw std::runtime_error("VVRL map-file CTU outside grid");
    }
    const std::string s=json_obs(o); append(s);
    int d=0;
    if(mode_=="constant") d=constant_delta_;
    if(mode_=="stdin" || mode_=="stdin-map") read_action(o,d);
    return d;
  }
  int map_delta(int poc,int ctu,int count) const {
    if(!map_mode()) return 0;
    if(ctu<0 || ctu>=count) throw std::runtime_error("VVRL CTU index out of range");
    if(mode_=="map-checker") return (ctu&1)?-constant_delta_:constant_delta_;
    if(mode_=="stdin-map") {
      auto it=map_stdin_.find(poc);
      if(it==map_stdin_.end()) throw std::runtime_error("VVRL missing stdin map for POC");
      return it->second.at(static_cast<size_t>(ctu));
    }
    auto it=map_file_.find(poc);
    if(it==map_file_.end()) return 0;
    auto j=it->second.find(ctu);
    return j==it->second.end()?0:j->second;
  }
  void applied(int poc,int native,int request,int accepted,double native_lambda,double accepted_lambda) const {
    if(!enabled())return;
    require_finite(native_lambda,"native_lambda");require_finite(accepted_lambda,"accepted_lambda");
    std::ostringstream s; s<<std::setprecision(12)
      <<"{\"kind\":\"picture_action\",\"poc\":"<<poc
      <<",\"qp_native\":"<<native<<",\"qp_requested\":"<<request
      <<",\"qp_accepted\":"<<accepted
      <<",\"lambda_native\":"<<native_lambda<<",\"lambda_accepted\":"<<accepted_lambda<<"}";
    append(s.str());
  }
  void slice_qp(int poc,int qp)const {
    if(enabled())append("{\"kind\":\"slice_qp\",\"poc\":"+std::to_string(poc)+",\"qp\":"+std::to_string(qp)+"}");
  }
  void ctu_map_applied(int poc,int ctu,int old_qp,int request,int actual)const {
    if(!enabled())return;
    std::ostringstream s;s<<"{\"kind\":\"ctu_map_action\",\"poc\":"<<poc
      <<",\"ctu\":"<<ctu<<",\"qp_before\":"<<old_qp
      <<",\"qp_requested\":"<<request<<",\"qp_accepted\":"<<actual<<"}";
    append(s.str());
  }
  void coding_qp_summary(int poc,int count,int minimum,int maximum,double average)const {
    if(!enabled())return;
    require_finite(average,"coding_mean_qp");
    std::ostringstream s;s<<std::setprecision(12)<<"{\"kind\":\"coding_qp\",\"poc\":"<<poc
      <<",\"cu_count\":"<<count<<",\"qp_min\":"<<minimum<<",\"qp_max\":"<<maximum
      <<",\"qp_mean_unweighted\":"<<average<<"}";
    append(s.str());
  }
  void ctu_coding_summary(int poc,int ctu,int count,double mean) const {
    if(!enabled() || count<=0)return;
    require_finite(mean,"ctu_mean_qp");
    std::ostringstream s;s<<std::setprecision(12)<<"{\"kind\":\"coding_ctu_qp\",\"poc\":"<<poc
      <<",\"ctu\":"<<ctu<<",\"cu_count\":"<<count<<",\"qp_mean_unweighted\":"<<mean<<"}";
    append(s.str());
  }
  void feedback(int poc,int qp,int bits,int target_bits) {
    if(!enabled())return;
    prev_valid_=target_bits>0;
    prev_feedback_poc_=prev_valid_?poc:-1;
    prev_qp_=prev_valid_?qp:0;
    prev_ratio_=prev_valid_?double(bits)/target_bits:0.;
    std::ostringstream s;s<<std::setprecision(12)<<"{\"kind\":\"feedback\",\"poc\":"<<poc
      <<",\"qp\":"<<qp<<",\"actual_bits\":"<<bits<<",\"target_bits\":"<<target_bits
      <<",\"prev_valid\":"<<(prev_valid_?"true":"false")<<"}";
    append(s.str());
  }
};
inline Controller& controller(){static Controller instance;return instance;}
} // namespace vvrl
