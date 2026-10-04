"""Preserve measured byte hashes and verify text across Git line-ending changes."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources(report, results):
    record = results / 'source_provenance.json'
    portable = json.loads(record.read_text(encoding='utf-8')) if record.exists() else {}
    for name, expected in report['source_sha256'].items():
        data = (ROOT / name.replace('\\', '/')).read_bytes()
        if digest(data) == expected:
            continue
        entry = portable.get('sources', {}).get(name)
        assert entry and entry['measured_sha256'] == expected, name
        assert digest(data.replace(b'\r\n', b'\n')) == entry['lf_normalized_sha256'], name


def main():
    results = HERE / 'results'
    destination = results / 'source_provenance.json'
    assert not destination.exists(), 'Retained provenance is not overwritten'
    report = json.loads((results / 'timings.json').read_text(encoding='utf-8'))
    sources = {}
    for name, expected in report['source_sha256'].items():
        data = (ROOT / name.replace('\\', '/')).read_bytes()
        assert digest(data) == expected, ('Need the original measured bytes', name)
        sources[name] = dict(measured_sha256=expected,
                             lf_normalized_sha256=digest(data.replace(b'\r\n', b'\n')))
    record = dict(description='The original raw byte hashes remain in timings.json. These additional '
                  'hashes normalize CRLF to LF only, permitting verification across Git checkout settings.',
                  sources=sources)
    destination.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    verify_sources(report, results)
    print('Recorded portable source digests for', len(sources), 'files')


if __name__ == '__main__':
    main()
