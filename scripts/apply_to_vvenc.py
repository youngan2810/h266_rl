#!/usr/bin/env python3
"""Fail-closed overlay for *stock* VVenC v1.14.0 (exact commit). Not a generic patch.

The overlay changes only QP-facing paths: EncGOP.cpp, EncSlice.cpp, RateCtrl.cpp,
EncCu.cpp and adds VVRLControl.h. It never modifies partition/MTS algorithms.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys

PIN = '9428ea8636ae7f443ecde89999d16b2dfc421524'
TARGET = 'source/Lib/EncoderLib'


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected exactly one exact anchor, saw {n}')
    return src.replace(old, new, 1)


def sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


def patch(root: Path, header: Path):
    root=root.resolve()
    if not (root / '.git').exists():
        raise RuntimeError('source root must be a local git checkout')
    result=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if result != PIN:
        raise RuntimeError(f'Wrong VVenC commit: {result}; expected {PIN}')
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
    if dirty.strip():
        raise RuntimeError('Refusing dirty checkout: save your changes or use a fresh clone')
    d=root/TARGET
    old={p:(d/p).read_text(encoding='utf-8') for p in ['EncGOP.cpp','EncSlice.cpp','RateCtrl.cpp','EncCu.cpp']}
    updated=dict(old)

    updated['EncGOP.cpp']=replace_once(updated['EncGOP.cpp'],
        '#include "GOPCfg.h"', '#include "GOPCfg.h"\n#include "VVRLControl.h"', 'EncGOP include')
    observation=r'''  // VVRL_BEGIN PICTURE_OBSERVATION: run after native RC and GOP QP adjustment.
  if( vvrl::controller().enabled() && m_pcRateCtrl->rcIsFinalPass && !m_isPreAnalysis )
  {
    if( m_pcEncCfg->m_numThreads != 0 || m_pcEncCfg->m_maxParallelFrames > 1 )
      throw std::runtime_error("VVRL pilot requires serial encoding: numThreads=0 and maxParallelFrames<=1");
    if( m_pcEncCfg->m_RCTargetBitrate <= 0 || !m_pcRateCtrl->encRCSeq || !pic->encRCPic )
      throw std::runtime_error("VVRL pilot requires positive bitrate-based native Rate Control");
    if( !std::isfinite(pic->picInitialLambda) || pic->picInitialLambda <= 0 )
      throw std::runtime_error("VVRL native lambda not available");
    if( vvrl::controller().map_mode() && (!m_pcEncCfg->m_usePerceptQPA || !pic->slices[0]->pps->useDQP) )
      throw std::runtime_error("VVRL map experiment requires QPA + PPS useDQP");

    vvrl::Observation ob;
    const auto* rcseq = m_pcRateCtrl->encRCSeq;
    const double fps = double(m_pcEncCfg->m_FrameRate) / m_pcEncCfg->m_FrameScale;
    if( fps <= 0 || m_pcEncCfg->m_SourceWidth<=0 || m_pcEncCfg->m_SourceHeight<=0 )
      throw std::runtime_error("VVRL invalid FPS or frame dimensions");
    ob.poc=pic->poc;
    ob.native_qp=pic->picInitialQP;
    ob.qp_min=-pic->slices[0]->sps->qpBDOffset[CH_L];
    ob.qp_max=MAX_QP;
    ob.frame_type=pic->slices[0]->sliceType==I_SLICE?0:(pic->slices[0]->sliceType==P_SLICE?1:2);
    ob.temporal_layer=pic->slices[0]->TLayer;
    ob.gop_size=m_pcEncCfg->m_GOPSize;
    ob.gop_position=ob.gop_size>0?((ob.poc%ob.gop_size)+ob.gop_size)%ob.gop_size:0;
    ob.target_bpp=double(m_pcEncCfg->m_RCTargetBitrate)/(fps*m_pcEncCfg->m_SourceWidth*m_pcEncCfg->m_SourceHeight);
    ob.budget_error=double(rcseq->bitsUsed-rcseq->estimatedBitUsage)/std::max<int64_t>(1,rcseq->estimatedBitUsage);
    ob.picture_target_bits=pic->encRCPic->targetBits;
    ob.source_width=m_pcEncCfg->m_SourceWidth;
    ob.source_height=m_pcEncCfg->m_SourceHeight;
    ob.ctu_columns=pic->cs->pcv->widthInCtus;
    ob.ctu_rows=pic->cs->pcv->heightInCtus;
    // S1 content features are NOT enabled until availability audit; do not leak guessed values.
    ob.spatial_valid=false;
    ob.temporal_valid=false;
    vvrl::controller().fill_history(ob);

    const int action=vvrl::controller().action(ob);
    const int requested=ob.native_qp+action;
    const int accepted=vvrl::clip_int(requested,ob.qp_min,ob.qp_max);
    const double oldLambda=pic->picInitialLambda;
    if( accepted!=ob.native_qp )
    {
      pic->picInitialQP=accepted;
      pic->picInitialLambda=oldLambda*std::pow(2.0,double(accepted-ob.native_qp)/3.0);
    }
    vvrl::controller().applied(ob.poc,ob.native_qp,requested,accepted,oldLambda,pic->picInitialLambda);
  }
  // VVRL_END PICTURE_OBSERVATION

'''
    updated['EncGOP.cpp']=replace_once(updated['EncGOP.cpp'],
       '  // compress next picture\n  picEncoder->compressPicture( *pic, *this );',
       observation + '  // compress next picture\n  picEncoder->compressPicture( *pic, *this );' + r'''

  // VVRL_BEGIN POST_ENCODE_CU_QP: finalized (NOT pre-action) coding QP summary.
  if( vvrl::controller().enabled() && m_pcRateCtrl->rcIsFinalPass && !m_isPreAnalysis )
  {
    int n=0, lo=MAX_QP, hi=-999;
    double sum=0.;
    const int numCtu=pic->cs->pcv->sizeInCtus;
    const int cols=pic->cs->pcv->widthInCtus;
    const int ctuSize=pic->slices[0]->sps->CTUSize;
    std::vector<int> ctuCnt(numCtu,0);
    std::vector<double> ctuQpSum(numCtu,0.);
    for( const auto* cu : pic->cs->cus )
    {
      if( cu && cu->chType == CH_L )
      {
        const int qp=cu->qp;
        ++n; sum+=qp; lo=std::min(lo,qp); hi=std::max(hi,qp);
        const int index=(cu->lumaPos().y/ctuSize)*cols+(cu->lumaPos().x/ctuSize);
        if(index>=0 && index<numCtu) { ++ctuCnt[index]; ctuQpSum[index]+=qp; }
      }
    }
    if(n>0) vvrl::controller().coding_qp_summary(pic->poc,n,lo,hi,sum/n);
    for(int i=0;i<numCtu;i++)
      if(ctuCnt[i]>0) vvrl::controller().ctu_coding_summary(pic->poc,i,ctuCnt[i],ctuQpSum[i]/ctuCnt[i]);
  }
  // VVRL_END POST_ENCODE_CU_QP''', 'EncGOP picture hook')

    updated['EncSlice.cpp']=replace_once(updated['EncSlice.cpp'],
        '#include "EncSlice.h"', '#include "EncSlice.h"\n#include "VVRLControl.h"', 'EncSlice include')
    updated['EncSlice.cpp']=replace_once(updated['EncSlice.cpp'],
        '  slice->sliceQp            = iQP;\n  slice->chromaQpAdjEnabled',
        '  slice->sliceQp            = iQP;\n  // VVRL_BEGIN FINAL_SLICE_QP\n  if( vvrl::controller().enabled() && !slice->pic->isPreAnalysis )\n    vvrl::controller().slice_qp(slice->poc,iQP);\n  // VVRL_END FINAL_SLICE_QP\n  slice->chromaQpAdjEnabled', 'EncSlice QP')

    # Historical values are updated only after picture rate-control feedback (not by frame POC).
    updated['RateCtrl.cpp']= '#include "VVRLControl.h"\n'+updated['RateCtrl.cpp']
    updated['RateCtrl.cpp']=replace_once(updated['RateCtrl.cpp'],
        '  encRCSeq->updateAfterPic( pic->actualTotalBits, encRCPic->tmpTargetBits );',
        '  encRCSeq->updateAfterPic( pic->actualTotalBits, encRCPic->tmpTargetBits );\n'
        '  // VVRL_BEGIN FEEDBACK: only completed pictures, not pre-action statistics.\n'
        '  if( rcIsFinalPass && !pic->isPreAnalysis )\n'
        '    vvrl::controller().feedback(pic->poc,pic->slices[0]->sliceQp,pic->actualTotalBits,encRCPic->targetBits);\n'
        '  // VVRL_END FEEDBACK', 'RateCtrl feedback')

    updated['EncCu.cpp']=replace_once(updated['EncCu.cpp'],
        '#include "EncCu.h"', '#include "EncCu.h"\n#include "VVRLControl.h"', 'EncCu include')
    ctu = r'''  // VVRL_BEGIN EXPERIMENTAL_QG_MAP: optional; do not use for PPO before validation.
  // Native QPA has already selected the QP/lambda for this QG. The RL offset
  // is applied afterwards to the current luma QG, respecting the VVC bit-depth QP range.
  if( vvrl::controller().map_mode() && !slice.pic->isPreAnalysis &&
      isLuma(partitioner.chType) && pps.useDQP && partitioner.currQgEnable() )
  {
    const int ctuIdx=getCtuAddr(partitioner.currQgPos,*pps.pcv);
    const int nCTU=pps.pcv->sizeInCtus;
    const int d=vvrl::controller().map_delta(slice.poc,ctuIdx,nCTU);
    if(d!=0)
    {
      const int q0=tempCS->baseQP;
      const int req=q0+d;
      const int qmin=-sps.qpBDOffset[CH_L];
      const int qmax=MAX_QP;
      if(qmin>qmax) throw std::runtime_error("VVRL invalid QG QP bounds");
      const int qa=vvrl::clip_int(req,qmin,qmax);
      if(qa!=q0)
      {
        const double oldLambda=m_cRdCost.getLambda();
        tempCS->currQP[partitioner.chType]=tempCS->baseQP=qa;
        bestCS->currQP[partitioner.chType]=bestCS->baseQP=qa;
        updateLambda(slice,oldLambda,q0,qa,true);
      }
      vvrl::controller().ctu_map_applied(slice.poc,ctuIdx,q0,req,qa);
    }
  }
  // VVRL_END EXPERIMENTAL_QG_MAP

'''
    updated['EncCu.cpp']=replace_once(updated['EncCu.cpp'],
       '  if (partitioner.currQtDepth == 0)\n  {\n    m_MergeSimpleFlag = 0;',
       ctu+'  if (partitioner.currQtDepth == 0)\n  {\n    m_MergeSimpleFlag = 0;',
       'EncCu post-QPA QG hook')

    if (d/'VVRLControl.h').exists():
        raise RuntimeError('VVRLControl.h already exists')
    content=header.read_bytes()
    for name,source in updated.items():
        if source==old[name]:raise RuntimeError(f'No patch produced for {name}')
    # All anchor + consistency checks above succeed: now write overlay.
    try:
        for name,source in updated.items():
            (d/name).write_text(source,encoding='utf-8')
        (d/'VVRLControl.h').write_bytes(content)
    except Exception:
        for name,source in old.items():
            (d/name).write_text(source,encoding='utf-8')
        (d/'VVRLControl.h').unlink(missing_ok=True)
        raise
    print('Patched VVenC v1.14.0 source:', root)
    for name in updated:
        print('  MODIFIED',f'{TARGET}/{name}', 'sha256=',sha(updated[name].encode()))
    print('  ADDED',f'{TARGET}/VVRLControl.h', 'sha256=',sha(content))
    print('NOT YET VERIFIED: actual VVenC build, encode, VVdeC decode, QG signaling')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--vvenc',required=True,type=Path)
    ap.add_argument('--header',type=Path,default=Path(__file__).resolve().parents[1]/'src'/'VVRLControl.h')
    args=ap.parse_args()
    patch(args.vvenc,args.header)
