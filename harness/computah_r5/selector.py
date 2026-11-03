"""Adapters from frozen LAYS/r5 decisions to Computah candidate IDs."""
from __future__ import annotations
import re, time

FAMILY_OP = {"click":"invoke", "double_click":"invoke", "right_click":"invoke", "type":"type_text",
             "key":"press_return", "scroll":"scroll_down", "drag":"invoke", "done":"already_satisfied",
             "fail":"none_here", "ask":"reobserve", "wait":"reobserve"}

def snap_point(point, candidates):
    """Return smallest containing candidate and Euclidean distance; ties/absence abstain."""
    if not point: return None, None, "no_point"
    x,y=point; found=[]
    for c in candidates:
        b=c.get("box")
        if isinstance(b, dict):
            x0,y0,x1,y1=(b.get(k) for k in ("x0","y0","x1","y1"))
        elif isinstance(b,(tuple,list)) and len(b)==4: x0,y0,x1,y1=b
        else: continue
        if None not in (x0,y0,x1,y1) and x0<=x<=x1 and y0<=y<=y1:
            found.append(((x1-x0)*(y1-y0),c))
    if not found: return None,None,"outside_candidates"
    found.sort(key=lambda z:z[0]); smallest=found[0][0]
    tied=[c for area,c in found if area==smallest]
    if len(tied)!=1:return None,None,"ambiguous_containment"
    c=tied[0]; b=c["box"]
    if isinstance(b,dict): x0,y0,x1,y1=(b[k] for k in ("x0","y0","x1","y1"))
    else:x0,y0,x1,y1=b
    cx,cy=(x0+x1)/2,(y0+y1)/2
    return c,((x-cx)**2+(y-cy)**2)**.5,None

def exact_substring(goal, generated):
    """Find the longest exact nonempty goal substring represented in generated output."""
    s=generated.strip(); best=None
    for n in range(len(s),0,-1):
        for i in range(len(s)-n+1):
            piece=s[i:i+n]; start=goal.find(piece)
            if start>=0 and (best is None or n>len(best["text"])):
                best={"start":start,"end":start+len(piece),"text":piece,"source":"goal"}
        if best is not None and (n >= 2 or not s.isascii()): return best
    return {"text":generated,"source":"generated"}

class R5Selector:
    def __init__(self, policy, route="element", done_threshold=.9):
        if route not in ("pixel","element"): raise ValueError(route)
        self.policy,self.route,self.done_threshold=policy,route,done_threshold
    def select(self,state):
        started=time.perf_counter(); candidates=state.get("candidates",[]); goal=state.get("goal","")
        # A callable may be injected by tests. Production wrapper delegates to existing policy heads.
        if hasattr(self.policy,"select_computah"):
            raw=self.policy.select_computah(state,self.route)
        else:
            raw={"family":"fail","route":"none"}
        family=raw.get("family","fail"); route=self.route; chosen=None; distance=None; reason=None
        if route=="element":
            idx=raw.get("element_index")
            if type(idx) is int and 0<=idx<len(candidates): chosen=candidates[idx]
            else:
                route="pixel"; raw["pixel_fallback_used"]=True
        if route=="pixel" and chosen is None:
            chosen,distance,reason=snap_point(raw.get("point"),candidates)
        if family in ("done","fail","ask","wait"):
            cid={"done":"already_satisfied","fail":"none_here","ask":"reobserve","wait":"reobserve"}[family]
        elif chosen is None: cid="none_here"
        else: cid=chosen["id"]
        op=chosen.get("operation") if chosen else FAMILY_OP.get(family)
        result={"choice_id":cid,"action":op,"source_span":None,"probabilities":raw.get("probabilities",{}),
                "route":route,"pixel_fallback_used":bool(raw.get("pixel_fallback_used",False)),
                "diagnostics":{"family":family,"snap_distance":distance,"abstain_reason":reason,
                               "latency_ms":(time.perf_counter()-started)*1000,"p_done":raw.get("p_done"),"p_feasible":raw.get("p_feasible")}}
        if family=="type":
            span=raw.get("span")
            if span and 0<=span[0]<=span[1]<=len(goal):
                result["source_span"]={"start":span[0],"end":span[1],"text":goal[span[0]:span[1]]}
            else: result["diagnostics"]["text_source"]="generated"
        return result
