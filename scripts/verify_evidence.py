
"""Verify stored evidence only: no torch, training, inference or dataset access."""
import hashlib,json,pathlib
def require(value,message):
 if not value:raise RuntimeError(message)
def main():
 r=pathlib.Path(__file__).resolve().parents[1]
 manifest=json.loads((r/"MANIFEST.sha256.json").read_text())
 for name,expected in manifest.items():
  p=r/name
  require(p.is_file(),"Missing file: "+name)
  require(hashlib.sha256(p.read_bytes()).hexdigest()==expected,"Changed file: "+name)
 h=json.loads((r/"evidence/hierarchy/COMPLETION.json").read_text())
 require(len(h["results"])==9,"Expected nine hierarchy freezes")
 for row in h["results"]:
  require(row["duplicate_exact"] is True,"Hierarchy duplicate not exact")
  require(row["first"]==row["second"],"Saved hierarchy duplicate drift")
  c=row["candidate"];p=r/"checkpoints/hierarchy"/pathlib.Path(c["checkpoint"]).name
  require(hashlib.sha256(p.read_bytes()).hexdigest()==c["checkpoint_sha256"],"Hierarchy checkpoint drift")
 for row in h["paired"]:
  require(row["ce"]["cross"]<row["ce"]["original"],"Cross-scale paired win missing")
 p=json.loads((r/"evidence/pair_energy/confirmation/RESULT.json").read_text())
 require(len(p)==12,"Expected 12 pair-energy confirmations")
 for row in p:
  require(row["duplicate_drift"]==0,"Saved pair-energy duplicate drift")
  matches=list((r/"checkpoints/pair_energy").glob(str(row["seed"])+"-"+row["arm"]+"-*.pt"))
  require(len(matches)==1,"Missing/ambiguous selected checkpoint")
  require(hashlib.sha256(matches[0].read_bytes()).hexdigest()==row["sha256"],"Pair-energy checkpoint drift")
 for seed in {x["seed"] for x in p}:
  ce={x["arm"]:x["validation"]["ce"] for x in p if x["seed"]==seed}
  require(all(ce["pair_energy"]<v for a,v in ce.items() if a!="pair_energy"),"Pair-energy paired win missing")
 w=json.loads((r/"data/incomplete_word_sweep.json").read_text())
 require(w["completed_trials"]==6 and w["terminal_receipt_present"] is False,"Reassess incomplete word-sweep status")
 hist=json.loads((r/"data/historical_10m.json").read_text())
 require(hashlib.sha256((r/hist["source"]).read_bytes()).hexdigest()==hist["source_sha256"],"Historical PDF drift")
 require(hist["nanogpt"]["ce"][0]<hist["nanogpt"]["hierarchy_ce"][0],"Retain nanoGPT's 512-token win")
 print("PASS:",len(manifest),"file hashes; 21 selected checkpoints; saved duplicate receipts and paired wins.")
 print("Historical 120M results remain report-backed, not raw checkpoint revalidated. No new evaluation.")
if __name__=="__main__":main()
