"""Typed, fakeable Windows UIA workflow core with checked native delivery."""
from __future__ import annotations
import json, os, threading, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

def source_span(source: str, start: int, end: int) -> dict:
    """Validate Unicode codepoint offsets and export both Python and UTF-16 ranges."""
    if not (0 <= start <= end <= len(source)): raise ValueError("source span out of bounds")
    return {"start":start,"end":end,"utf16_start":len(source[:start].encode('utf-16-le'))//2,
            "utf16_end":len(source[:end].encode('utf-16-le'))//2,"text":source[start:end]}

@dataclass
class Candidate:
    id:str; operation:str; role:str=""; label:str=""; value:str=""; enabled:bool=True
    focused:bool=False; box:dict|None=None; ref:str=""; window_id:str=""; process_id:int|None=None
    actions:list[str]=field(default_factory=list)

@dataclass
class State:
    goal:str; current_instruction:str; source_text:str; candidates:list[Candidate]
    observation_id:str; coverage:dict=field(default_factory=dict); verified_progress:list[str]=field(default_factory=list)
    pending_effect:dict|None=None; image_path:str|None=None; geometry:dict|None=None
    apps:list[dict]=field(default_factory=list); windows:list[dict]=field(default_factory=list)
    def selector_dict(self):
        # Only observation-safe fields cross the model boundary.
        return {"goal":self.goal,"current_instruction":self.current_instruction,"source_text":self.source_text,
          "candidates":[{k:getattr(c,k) for k in ("id","operation","role","label","value","enabled","focused","box","actions")} for c in self.candidates],
          "observation_id":self.observation_id,"coverage":self.coverage,"verified_progress":self.verified_progress,
          "pending_effect":self.pending_effect,"image_path":self.image_path,"geometry":self.geometry}

class Backend(Protocol):
    def observe(self)->dict: ...
    def dispatch(self, candidate:Candidate, action:dict)->dict: ...

class Selector(Protocol):
    def select(self,state:dict)->dict: ...

_mutex=threading.Lock()

class Runtime:
    """No implicit retries: uncertain delivery/effect requires fresh observation and new choice."""
    def __init__(self, backend:Backend, selector:Selector, log_path=None, dry_run=True, cancel=None):
        self.backend,self.selector=backend,selector; self.log_path=Path(log_path) if log_path else None
        self.dry_run=dry_run; self.cancel=cancel or threading.Event(); self.checkpoint={}; self.registry={}

    def _log(self, event):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False)+'\n')

    def run(self, goal:str, source:str, max_steps=12):
        cursor=0; verified=[]; history=[]
        for _ in range(max_steps):
            if self.cancel.is_set(): return self._stop("cancelled",source,cursor,verified)
            obs=self.backend.observe(); oid=str(obs.get('observation_id') or uuid.uuid4())
            self.registry={c['id']:(oid,c) for c in obs.get('candidates',[])}
            instruction=obs.get('current_instruction') or source[cursor:]
            state=State(goal,instruction,source,[Candidate(**c) for c in obs.get('candidates',[])],oid,
                        obs.get('coverage',{}),verified,obs.get('pending_effect'),obs.get('image_path'),obs.get('geometry'),
                        obs.get('apps',[]),obs.get('windows',[]))
            decision=self.selector.select(state.selector_dict()); self._log({"event":"choice","observation_id":oid,"decision":decision})
            if decision.get('choice_id') in (None,'none_here'):
                return self._stop("no_action_unknown",source,cursor,verified)
            entry=self.registry.get(decision['choice_id'])
            if not entry or entry[0]!=oid: return self._stop("stale_or_unknown_candidate",source,cursor,verified)
            cand=Candidate(**entry[1]); span=decision.get('source_span')
            if span:
                checked=source_span(source,int(span['start']),int(span['end']))
                if span.get('text',checked['text'])!=checked['text']: return self._stop("invalid_source_span",source,cursor,verified)
            if not cand.enabled: return self._stop("disabled_candidate",source,cursor,verified)
            operation=decision.get('action',decision.get('operation',cand.operation))
            if operation != cand.operation: return self._stop("operation_mismatch",source,cursor,verified)
            action={"action":operation,"observation_id":oid}
            if span:
                action["text"]=checked["text"]
                action["source_span"]={"start":checked["start"],"end":checked["end"]}
            if decision.get('arguments'):
                return self._stop("unchecked_arguments",source,cursor,verified)
            if self.dry_run: self._log({"event":"dry_run","action":action}); return self._stop("dry_run",source,cursor,verified)
            if os.name != 'nt': return self._stop("execution_mutex_unavailable",source,cursor,verified)
            from windows_input import ExecutionLock, emergency_stop
            try: execution_lock=ExecutionLock()
            except Exception: return self._stop("execution_mutex_busy",source,cursor,verified)
            try:
                if self.cancel.is_set() or emergency_stop(): return self._stop("cancelled_before_dispatch",source,cursor,verified)
                report=self.backend.dispatch(cand,action)
            finally: execution_lock.close()
            self.checkpoint={"source":source,"cursor":cursor,"verified_progress":verified,"pending_effect":{"action":action,"delivery":report.get('delivery','unknown')}}
            self._log({"event":"dispatch","report":report})
            if report.get('delivery') not in ('delivered','delivered_accessibility','delivered_input'):
                return self._stop("delivery_unknown_or_failed",source,cursor,verified)
            # Verification is explicitly from newly observed state and same-object evidence.
            after=self.backend.observe(); effect=self.backend.verify(obs,after,cand,action) if hasattr(self.backend,'verify') else {"verified":False}
            self._log({"event":"verification","evidence":effect})
            if effect.get('verified') is not True or effect.get('same_object') is not True:
                return self._stop("effect_unverified",source,cursor,verified)
            phase=effect.get('phase')
            if phase=="progress":
                verified.append(effect.get('summary',instruction))
                self.checkpoint={"source":source,"cursor":cursor,"verified_progress":verified,"pending_effect":None}
                continue
            if phase!="complete":
                self.checkpoint["pending_effect"]["verification"]=effect
                return self._stop("effect_not_complete",source,cursor,verified)
            next_cursor=effect.get('next_cursor')
            if type(next_cursor) is not int or not cursor < next_cursor <= len(source):
                self.checkpoint["pending_effect"]["verification"]=effect
                return self._stop("invalid_completion_evidence",source,cursor,verified)
            verified.append(effect.get('summary',instruction))
            cursor=next_cursor
            if cursor>=len(source): return {"status":"completed","verified_progress":verified,"cursor":cursor,"pending_effect":None}
        return self._stop("step_limit",source,cursor,verified)

    @staticmethod
    def _stop(reason,source,cursor,verified):
        return {"status":"stopped","reason":reason,"source":source,"cursor":cursor,"verified_progress":verified,
                "pending_effect":{"outcome":"unknown until observed","action_must_not_repeat":True} if reason in ('effect_unverified','effect_not_complete','invalid_completion_evidence','delivery_unknown_or_failed') else None}
