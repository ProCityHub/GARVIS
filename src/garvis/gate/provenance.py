from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re

_SHA40=re.compile(r"^[0-9a-fA-F]{40}$")

class ArtifactStatus(str,Enum):
 ACTIVE="ACTIVE"
 HISTORICAL="HISTORICAL"
 SUPERSEDED="SUPERSEDED"
 MISSING="MISSING"
 UNRESOLVED="UNRESOLVED"

@dataclass(frozen=True)
class ProvenanceRecord:
 name:str
 status:ArtifactStatus
 historical_ref:str=""
 current_ref:str=""
 commit:str=""
 note:str=""
 def immutable_commit_valid(self)->bool:
  return bool(self.commit and _SHA40.fullmatch(self.commit))
 def can_be_live_dependency(self)->bool:
  return self.status is ArtifactStatus.ACTIVE and bool(self.current_ref) and self.immutable_commit_valid()
 def citation_resolved(self)->bool:
  if self.status is ArtifactStatus.ACTIVE:return bool(self.current_ref) and self.immutable_commit_valid()
  if self.status is ArtifactStatus.SUPERSEDED:return bool(self.historical_ref and self.current_ref) and self.immutable_commit_valid()
  if self.status is ArtifactStatus.HISTORICAL:return bool(self.historical_ref)
  return False

def validate_record(record:ProvenanceRecord)->bool:
 if record.commit and not record.immutable_commit_valid():raise ValueError("commit must be a 40-character Git SHA")
 if record.status is ArtifactStatus.ACTIVE and not record.current_ref:raise ValueError("active provenance requires current_ref")
 if record.status is ArtifactStatus.SUPERSEDED and (not record.historical_ref or not record.current_ref):raise ValueError("superseded provenance requires historical and current refs")
 if record.status is ArtifactStatus.MISSING and record.current_ref:raise ValueError("missing artifact cannot claim a current source")
 return True

def dead_link_is_evidence()->bool:
 return False

def historical_reference_authorizes_execution()->bool:
 return False

def repository_name_authenticates_identity()->bool:
 return False
