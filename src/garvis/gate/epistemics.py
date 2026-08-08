from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Tuple

class ClaimState(str,Enum):
 HYPOTHESIS="HYPOTHESIS"
 SUPPORTED="SUPPORTED"
 NOT_SUPPORTED="NOT_SUPPORTED"
 UNRESOLVED="UNRESOLVED"
 RETRACTED="RETRACTED"

@dataclass(frozen=True)
class EvidenceRecord:
 evidence_id:str
 source:str
 support:float
 verified:bool=True

@dataclass(frozen=True)
class ClaimRecord:
 claim_id:str
 text:str
 state:ClaimState
 confidence:float
 evidence_ids:Tuple[str,...]=()

@dataclass(frozen=True)
class RetractionEntry:
 claim_id:str
 reason:str
 superseded_by:str=""

@dataclass(frozen=True)
class RetractionLedger:
 entries:Tuple[RetractionEntry,...]=()
 def append(self,entry:RetractionEntry)->"RetractionLedger":
  if any(x.claim_id==entry.claim_id for x in self.entries): raise ValueError("retraction already recorded")
  return RetractionLedger(self.entries+(entry,))
 def replace(self,*args,**kwargs):
  raise RuntimeError("retractions are append-only")

def normalize_weights(values):
 v=tuple(float(x) for x in values)
 if not v or any(x<0 for x in v) or sum(v)<=0: raise ValueError("invalid candidate weights")
 s=sum(v)
 return tuple(x/s for x in v)

def entropy(values):
 p=normalize_weights(values)
 return -sum(x*log(x) for x in p if x>0)

def assess_claim(claim_id:str,text:str,evidence:Tuple[EvidenceRecord,...]=(),contradicted:bool=False)->ClaimRecord:
 verified=tuple(e for e in evidence if e.verified)
 ids=tuple(e.evidence_id for e in verified)
 if contradicted: return ClaimRecord(claim_id,text,ClaimState.NOT_SUPPORTED,0.0,ids)
 if not verified: return ClaimRecord(claim_id,text,ClaimState.HYPOTHESIS,0.0,())
 score=sum(max(-1.0,min(1.0,e.support)) for e in verified)/len(verified)
 confidence=min(1.0,abs(score))
 state=ClaimState.SUPPORTED if score>=0.6 else ClaimState.NOT_SUPPORTED if score<=-0.6 else ClaimState.UNRESOLVED
 return ClaimRecord(claim_id,text,state,confidence,ids)

def probability_is_proof(probability:float)->bool:
 return False

def physical_quantum_claim_allowed(hardware_independently_verified:bool)->bool:
 return bool(hardware_independently_verified)
