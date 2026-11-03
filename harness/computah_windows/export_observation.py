"""Capture one scoped native observation and export a model-safe record."""
import argparse, base64, json
from pathlib import Path
from .agent_desktop import AgentDesktop

def export(window_id, output, binary=None):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    desktop=AgentDesktop(binary=binary,window_id=window_id)
    state=desktop.observe(); raw=desktop.last['raw']
    if not raw.get('complete'): raise RuntimeError('incomplete UIA snapshot')
    snap=desktop.call('snapshot','--window-id',window_id,'--include-bounds','-i','--session','lays-native-eval')
    shot=desktop.call('screenshot','--window-id',window_id,'--session','lays-native-eval')
    image=out/'observation.png'; image.write_bytes(base64.b64decode(shot['data']))
    state['image_path']=image.name
    state['coverage']={**state['coverage'],'candidate_count':len(state['candidates']),'scope':'one owned Notepad window','candidate_source':'UIA'}
    (out/'observation.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    editor=next((c for c in state['candidates'] if c['role']=='textfield' and c['label']=='Text editor'),None)
    (out/'oracle.json').write_text(json.dumps({'intended_control_id':editor['id'] if editor else None,'criterion':'editable document field in the owned Notepad window','task_success':False,'note':'Capture only; no task completion is claimed.'},indent=2),encoding='utf8')
    (out/'capture.json').write_text(json.dumps({'source':'agent-desktop snapshot and screenshot CLI','window_id':window_id,'capture_geometry':shot.get('width'), 'height':shot.get('height'),'scale_factor':shot.get('scale_factor'),'snapshot_complete':snap.get('complete'),'ref_count':snap.get('ref_count'),'native_refs_retained_private':True},indent=2),encoding='utf8')
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--window-id',required=True); p.add_argument('--output',required=True); p.add_argument('--binary'); a=p.parse_args(); export(a.window_id,a.output,a.binary)
