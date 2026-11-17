#include "../src/VVRLControl.h"
#include <iostream>
int main() {
  vvrl::Observation o;
  o.poc=7; o.native_qp=30; o.qp_min=0; o.qp_max=63;
  o.frame_type=2; o.temporal_layer=3; o.gop_position=7; o.gop_size=32;
  o.target_bpp=0.125; o.budget_error=0.25; o.ctu_columns=4; o.ctu_rows=3;
  o.picture_target_bits=1300; o.source_width=480; o.source_height=320;
  auto& c=vvrl::controller(); c.fill_history(o);
  const int a=c.action(o);
  std::cout << "TEST_DELTA=" << a << '\n';
  c.applied(o.poc,o.native_qp,o.native_qp+a,vvrl::clip_int(o.native_qp+a,0,63),1200.,1200.*std::pow(2.,a/3.));
  if(c.map_mode()) std::cout<<"TEST_MAP_0="<<c.map_delta(o.poc,0,12)<<" TEST_MAP_1="<<c.map_delta(o.poc,1,12)<<'\n';
  c.slice_qp(o.poc,o.native_qp+a);
  c.coding_qp_summary(o.poc,10,26,32,29.2);
  c.feedback(o.poc,o.native_qp+a,1500,o.picture_target_bits);
}
