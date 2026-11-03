"""Check the installed standalone bundle without downloading anything."""
import importlib.metadata as md
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def main():
    import numpy
    import torch
    import transformers
    import sounddevice
    import safetensors
    from packaging.requirements import Requirement
    errors=[]
    for line in (ROOT/'requirements.lock.txt').read_text().splitlines():
        name, version=line.split('==',1)
        if md.version(name) != version:
            errors.append('Installed version mismatch: '+name)
    # Validate each active dependency's declared version range.
    for dist in md.distributions():
        for entry in dist.requires or []:
            req=Requirement(entry)
            if req.marker and not req.marker.evaluate({'extra':''}): continue
            try:
                version=md.version(req.name)
                if req.specifier and version not in req.specifier:
                    errors.append(f'{dist.metadata["Name"]} requires {req}; installed {version}')
            except md.PackageNotFoundError:
                errors.append(f'Missing dependency: {req}')
    for path in ['models/base/config.json','models/base/model.safetensors-00001-of-00001.safetensors','models/round2/molab-round2-fast/trainable.safetensors','bin/agent-desktop.exe','harness/live_local.py']:
        if not (ROOT/path).is_file(): errors.append('Missing asset: '+path)
    if not torch.cuda.is_available(): errors.append('CUDA GPU unavailable; install an NVIDIA driver for the CUDA runtime bundled with PyTorch.')
    # The status overlay needs a working Tcl/Tk runtime; report it cleanly.
    tkinter_available=True
    try:
        import tkinter
        window=tkinter.Tk()
        window.withdraw()
        window.destroy()
    except Exception as exc:
        tkinter_available=False
        errors.append(f'Tcl/Tk initialization failed: {type(exc).__name__}: {exc}')
    inputs=[d['name'] for d in sounddevice.query_devices() if d['max_input_channels']>0]
    result={'root':str(ROOT),'python':sys.version.split()[0],'prefix':sys.prefix,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'tkinter':tkinter_available,'microphone_devices':len(inputs),'errors':sorted(set(errors)),'ready':not errors}
    print(json.dumps(result,indent=2))
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data/dependency-check.json').write_text(json.dumps(result,indent=2))
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
