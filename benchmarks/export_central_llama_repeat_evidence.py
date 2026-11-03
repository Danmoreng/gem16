"""Export a privacy-safe derivative; retain the original local measurement record."""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/results/2026-11-03/6b71ee3/central-llama-7071ba3/comparison.json'
DESTINATION = ROOT / 'artifacts/llama_cpp/central-repeat-2026-11-03.json'


def export():
    raw = SOURCE.read_bytes()
    original = json.loads(raw)
    home = Path.home()
    central = home / 'Development/_engines/llama.cpp'
    hostnames = {os.uname().nodename}
    kernel = original.get('environment', {}).get('kernel', {}).get('stdout', '').split()
    if len(kernel) > 1 and kernel[0] == 'Linux':
        hostnames.add(kernel[1])
    prefixes = sorted([(str(ROOT), '${REPO_ROOT}'), (str(central), '${LLAMA_CPP_CENTRAL_ROOT}'),
                       (str(home), '${HOME}')], key=lambda pair: len(pair[0]), reverse=True)
    local_user = home.name

    def redact_text(text):
        for source, placeholder in prefixes:
            text = text.replace(source, placeholder)
        for hostname in hostnames:
            if hostname:
                text = text.replace(hostname, '<HOSTNAME>')
        text = re.sub(r'/home/[^/\s"\x27\\]+', '${HOME}', text)
        text = re.sub(r'/Users/[^/\s"\x27\\]+', '${HOME}', text)
        text = re.sub(r'GPU-[0-9a-fA-F-]{16,}', '<GPU_UUID>', text)
        text = re.sub(r'\b' + re.escape(local_user) + r'\b', '<USER>', text, flags=re.IGNORECASE)
        return text

    def redact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                public_key = redact_text(key)
                assert public_key not in result, 'Path redaction must not merge record keys'
                if key.lower() in {'pid', 'hostname', 'host_name', 'machine_id', 'machine-id',
                                   'gpu_uuid', 'uuid', 'serial_number', 'username', 'user_name'}:
                    result[public_key] = '<REDACTED>'
                else:
                    result[public_key] = redact(item)
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        return redact_text(value) if isinstance(value, str) else value

    public = redact(original)
    for field in ['comparisons', 'historical_samples', 'actual_dispatch']:
        assert public[field] == original[field], field
    for key, cohort in original['cohorts'].items():
        for field in ['summary', 'configuration', 'output_token_ids', 'server_facts']:
            assert public['cohorts'][key][field] == cohort[field], (key, field)
        assert {k:v for k,v in public['cohorts'][key]['workload'].items() if k != 'path'} == {
            k:v for k,v in cohort['workload'].items() if k != 'path'}, key
    public['publication'] = {
        'kind': 'privacy-safe derivative of the immutable local measurement record',
        'source_record_sha256': hashlib.sha256(raw).hexdigest(),
        'path_placeholders': ['${REPO_ROOT}', '${HOME}', '${LLAMA_CPP_CENTRAL_ROOT}'],
        'removed': ['local account paths/names', 'hostname', 'device UUIDs', 'process identifiers'],
        'retained': ['benchmark measurements', 'output token IDs', 'model/executable/library hashes',
                     'hardware models', 'driver/compiler versions', 'technical dispatch evidence'],
        'hash_scope': 'File hashes identify original local bytes, not the redacted strings in this export. Commands/scripts with placeholders require local substitution.',
    }
    text = json.dumps(public, indent=2) + '\n'
    assert str(home) not in text and not re.search(r'/home/|/Users/|GPU-[0-9a-fA-F-]{16,}', text)
    assert not any(hostname and hostname in text for hostname in hostnames)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(text)
    print('Exported privacy-safe benchmark record; measured values and original local record unchanged')


if __name__ == '__main__':
    export()
