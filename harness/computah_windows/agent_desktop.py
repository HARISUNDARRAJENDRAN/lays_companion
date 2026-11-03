"""agent-desktop CLI backend. Native refs are held in the runtime only."""
import json, os, shutil, subprocess, uuid
from pathlib import Path
from .runtime import Candidate

class AgentDesktop:
    def __init__(self,binary=None,app=None,window_id=None,timeout=15):
        path=binary or shutil.which('agent-desktop') or str(Path.home()/'lays-run/desktop/node_modules/agent-desktop/bin/agent-desktop-win32-x64.exe')
        if not Path(path).is_file(): raise RuntimeError('agent-desktop executable unavailable')
        self.binary=str(Path(path).resolve()); self.app=app; self.window_id=window_id; self.timeout=timeout; self.last={}
    def call(self,*args):
        p=subprocess.run([self.binary,*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=self.timeout)
        try: e=json.loads(p.stdout)
        except json.JSONDecodeError: raise RuntimeError('agent-desktop returned invalid JSON') from None
        if p.returncode or not e.get('ok'): raise RuntimeError('agent-desktop operation failed: '+str(e.get('error',{}).get('code','CLI_ERROR')))
        return e['data']
    def observe(self):
        apps=self.call('list-apps'); windows=self.call('list-windows')
        args=['snapshot','-i','--include-bounds']
        if self.app: args += ['--app',self.app]
        if self.window_id: args += ['--window-id',self.window_id]
        raw=self.call(*args); oid=str(raw.get('snapshot_id') or uuid.uuid4()); rows=[]
        def walk(node):
            if not isinstance(node,dict): return
            ref=node.get('ref_id')
            if ref:
                rows.append({"id":"c"+str(len(rows)),"operation":"activate","role":node.get('role',''),"label":node.get('name',''),
                  "value":node.get('value',''),"enabled":"disabled" not in node.get('states',[]),"focused":"focused" in node.get('states',[]),
                  "box":node.get('bounds'),"ref":ref,"window_id":(raw.get('window') or {}).get('id','') if isinstance(raw.get('window'),dict) else '',
                  "process_id":(raw.get('app') or {}).get('pid') if isinstance(raw.get('app'),dict) else None,"actions":node.get('available_actions',[])})
            for ch in node.get('children',[]): walk(ch)
        walk(raw.get('tree',{})); self.last={"observation_id":oid,"raw":raw,"candidates":rows}
        return {"observation_id":oid,"candidates":rows,"coverage":{"complete":raw.get('complete'),"ref_count":raw.get('ref_count')},
                "apps":apps,"windows":windows,"current_instruction":""}
    def dispatch(self,candidate,action):
        if action.get('observation_id')!=self.last.get('observation_id'): raise RuntimeError('stale observation')
        if candidate.ref not in {x['ref'] for x in self.last.get('candidates',[])}: raise RuntimeError('candidate ref not observed')
        op=action.get('action') or action.get('operation') or 'click'; ref=candidate.ref
        commands={'activate':'click','click':'click','press':'click','focus':'focus','select':'select','toggle':'toggle','expand':'expand','collapse':'collapse','check':'check','uncheck':'uncheck','clear':'clear'}
        if op in commands: self.call(commands[op],ref)
        elif op=='type': self.call('type',ref,'--text',str(action.get('text','')))
        elif op=='set-value': self.call('set-value',ref,'--value',str(action.get('value','')))
        else: raise ValueError('unsupported semantic operation')
        return {"delivery":"delivered_accessibility","operation":op}
    def verify(self,before,after,candidate,action):
        # A generic runtime cannot infer intent: require an explicit adapter-specific evidence rule.
        return {"verified":False,"same_object":False,"summary":"verification rule not configured"}
