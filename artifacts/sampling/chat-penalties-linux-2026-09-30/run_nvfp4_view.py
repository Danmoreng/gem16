"""Verify installed locked bytes, isolate root component, run the existing gate."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import fetch_model
import hf_cache
import validate_chat_penalties as gate

lock_path = ROOT / 'models/gemma4-26b-gem16-target.lock.json'
lock = json.loads(lock_path.read_text())
entries = fetch_model.validate_lock(lock)
source = hf_cache.locked_snapshot_path(lock_path)
parent = hf_cache.hub_cache_root() / '.gem16/snapshots'
evidence = Path(__file__).parent
with tempfile.TemporaryDirectory(prefix='nvfp4-penalty-check-', dir=parent) as tmp:
    view = Path(tmp)
    manifest = {'source': str(source), 'lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(), 'files': []}
    for entry in entries:
        path = source / entry['path']
        assert stat.S_ISREG(path.lstat().st_mode), path
        assert fetch_model.verify(path, entry), path
        if entry['path'] == 'gem16_components.json':
            manifest['excluded_hub_metadata'] = entry
            continue
        target = view / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        assert path.samefile(target)
        manifest['files'].append({'path': entry['path'], 'size': entry['size'], 'sha256': entry['sha256'], 'same_inode': True})
    (evidence / 'nvfp4-view.json').write_text(json.dumps(manifest, indent=2) + '\n')
    original = gate.profile_arguments
    def arguments(profile, d2):
        result = original(profile, d2)
        assert profile == 'nvfp4'
        result[1] = str(view)
        return result
    gate.profile_arguments = arguments
    sys.argv = [str(ROOT / 'tools/validate_chat_penalties.py'), '--server', str(ROOT / 'build/Linux/blackwell-release/bin/gem16-server'), '--output-dir', str(evidence / 'nvfp4-models-clean-view'), '--lifecycle', '--warmups', '1', '--samples', '1', '--profiles', 'nvfp4']
    raise SystemExit(gate.main())
