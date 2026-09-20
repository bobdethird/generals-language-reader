"""Download a completed 3800 interpreter and one real audit shard from Modal.

Run using .venv-modal/bin/python. Backbone weights are downloaded separately by
scripts/download_reader.py, pinned to reader-model.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import modal


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("runs/interpreter-3800"))
    p.add_argument("--what-reader",default="iter3800-decision-readout-v1")
    p.add_argument("--why-reader",default="iter3800-why-reader-v4")
    p.add_argument("--evidence",default="iter3800-counterfactual-data-v2")
    args=p.parse_args()
    for name in (args.what_reader,args.why_reader,args.evidence):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name):raise ValueError("Invalid experiment name")
    volume=modal.Volume.from_name("generals-reader-experiments")
    def read_json(path):return json.loads(b"".join(volume.read_file(path)))
    # Reports are only emitted after weight selection and held-out evaluation.
    what=read_json(args.what_reader+"/report.json")
    why=read_json(args.why_reader+"/report.json")
    data=read_json(args.evidence+"/manifest.json")
    provenance=read_json("iter3800-reader-transfer-v1/provenance.json")
    sha=provenance["player_sha256"]
    if any(x!=sha for x in (what["configuration"]["player_sha256"],why["configuration"]["player_sha256"],data["player_sha256"])):
        raise ValueError("Player/reader/evidence identities disagree")
    if not (what["frozen_player_unchanged"] and what["frozen_language_unchanged"] and why["language_unchanged"]):
        raise ValueError("Frozen-weight verification did not pass")
    shard=data["test"][0]
    args.output.mkdir(parents=True,exist_ok=True)
    files={"player-3800.eqx":"iter3800-reader-transfer-v1/L_7d_gae90_ema_3800.eqx",
           "what.pt":args.what_reader+"/best.pt","why.pt":args.why_reader+"/best.pt",
           "what-report.json":args.what_reader+"/report.json","why-report.json":args.why_reader+"/report.json",
           "evidence-manifest.json":args.evidence+"/manifest.json",
           "transfer-result.json":"iter3800-reader-transfer-v1/result.json"}
    files.update({"sample/"+name:args.evidence+"/"+shard+"/"+name for name in ("evidence.npz","examples.jsonl","report.json")})
    records={}
    for local,remote in files.items():
        target=args.output/local;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+".partial")
        with temporary.open("wb") as stream:
            for chunk in volume.read_file(remote):stream.write(chunk)
        temporary.replace(target)
        records[local]={"volume_path":remote,"sha256":digest(target),"bytes":target.stat().st_size}
        print(json.dumps({"downloaded":local,**records[local]}),flush=True)
    if records["player-3800.eqx"]["sha256"]!=sha:raise ValueError("Downloaded player hash mismatch")
    # Preserve the measurement records for every final-test explanation. They
    # permit independent claim verification without downloading every tensor.
    target=args.output/"why-test-evidence.jsonl"
    with target.with_suffix(".partial").open("wb") as stream:
        for test_shard in data["test"]:
            for chunk in volume.read_file(args.evidence+"/"+test_shard+"/examples.jsonl"):
                stream.write(chunk)
    target.with_suffix(".partial").replace(target)
    records[target.name]={"volume_path":args.evidence+"/<test shards>/examples.jsonl",
                          "sha256":digest(target),"bytes":target.stat().st_size}
    manifest={"player_iteration":3800,"player_sha256":sha,"what_reader":args.what_reader,
              "why_reader":args.why_reader,"files":records,
              "scope":"Post-hoc move reporting and measured local policy sensitivity; text cannot change game actions."}
    (args.output/"bundle.json").write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=="__main__":main()
