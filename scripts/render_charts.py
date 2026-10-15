
"""Render evidence-only SVG charts. No model imports, data downloads or evaluation."""
import html,json,math,pathlib
from statistics import mean
COLORS=["#d66b35","#187c69","#336eb5","#8d59ab"]
def esc(s):return html.escape(str(s))
class Canvas:
 def __init__(self,title,subtitle,foot,height=620):
  self.height=height;self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}"><rect width="1100" height="{height}" fill="#fbfcfe"/>']
  self.text(40,44,title,26,weight="bold");self.text(40,74,subtitle,15,color="#46546b")
  for i,t in enumerate(foot):self.text(40,height-36+18*i,t,13,color="#46546b")
 def text(self,x,y,t,size=14,color="#16223a",anchor="start",weight="normal"):
  self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{esc(t)}</text>')
 def line(self,x1,y1,x2,y2,color="#d9e1eb",width=1):
  self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
 def legend(self,labels):
  for i,label in enumerate(labels):
   x=55+i*258;self.parts.append(f'<rect x="{x}" y="99" width="15" height="12" fill="{COLORS[i]}"/>');self.text(x+22,110,label,14)
 def finish(self):return ("\n".join(self.parts)+"</svg>\n").encode()
def plot(title,subtitle,foot,categories,series,ymax,unit,log=False):
 c=Canvas(title,subtitle,foot);c.legend([s[0] for s in series]);left,right,top,bottom=90,1045,150,510
 def yy(v):return bottom-(math.log10(v)-3)/(8-3)*(bottom-top) if log else bottom-v/ymax*(bottom-top)
 ticks=[1e3,1e4,1e5,1e6,1e7,1e8] if log else [ymax*i/5 for i in range(6)]
 for t in ticks:
  y=yy(t);c.line(left,y,right,y);c.text(left-14,y+5,f'{t:g}',13,anchor="end",color="#64748b")
 c.text(45,137,unit,14);step=(right-left)/len(categories)
 for i,cat in enumerate(categories):c.text(left+(i+.5)*step,540,cat,15,anchor="middle")
 for j,(name,vals) in enumerate(series):
  w=min(64,step*.8/len(series))
  for i,v in enumerate(vals):
   x=left+(i+.5)*step+(j-(len(series)-1)/2)*(w+7);y=yy(v)
   base=yy(1e3) if log else bottom
   c.parts.append(f'<rect x="{x-w/2}" y="{y}" width="{w}" height="{base-y}" rx="3" fill="{COLORS[j]}"/>')
   label=f'{v/1e6:.2f} MB' if log and v>1e6 else f'{v:,.0f} B' if log else f'{v:.3f}'
   c.text(x,y-9,label,13,anchor="middle")
 return c.finish()
def build_charts(h,b):
 out={}
 out["historical_context.svg"]=plot("10M-token study: context-length extrapolation",
 "Historical report values | trained at 512 tokens | one seed | lower CE is better",
 ["Small 1K/2K samples; absolute-position extrapolation confounds architecture.",
  "Report-backed, not raw checkpoint revalidation. CE is not accuracy; no SOTA claim."],
 [str(x) for x in h["context_tokens"]],
 [("In-house Transformer",h["in_house"]["transformer_ce"]),("Pure hierarchy",h["in_house"]["hierarchy_ce"]),("Local + function weights",h["in_house"]["local_fw_ce"]),("Upstream nanoGPT control",h["nanogpt"]["ce"])],12,"Token CE")
 out["persistent_state.svg"]=plot("Constant recurrent state versus growing KV state",
 "Analytic bf16 bytes per sequence | log scale | historical configuration accounting",
 ["Excludes model parameters, activations, allocator overhead and latency; NOT total-memory savings.",
  "In-house KV sizes rounded in source; nanoGPT 73.5 MiB at 2K is reported separately in the README."],
 [str(x) for x in h["context_tokens"]],
 [("In-house KV",h["state"]["in_house_kv_bytes_rounded"]),("Hierarchy",[h["state"]["hierarchy_bytes"]]*3)],1e8,"Bytes (log10)",True)
 tasks=h["benchmarks"]
 out["task_accuracy.svg"]=plot("Reported task benchmarks: wins, and the zero-score task",
 "10M-token study | one seed | limit 1,000 per task | higher is better",
 ["Historical report transcription; no confidence intervals. These are local checkpoint scores, NOT task SOTA.",
  "LAMBADA is included at zero for every arm. Absolute accuracy changes are percentage points."],
 tasks["tasks"],[("In-house Transformer",tasks["in_house_transformer_accuracy"]),("Pure hierarchy",tasks["pure_hierarchy_accuracy"]),("Local + function weights",tasks["local_fw_accuracy"])],.6,"Accuracy")
 # Show every seed; the two panels are separate experiments, not a joint leaderboard.
 c=Canvas("Confirmed byte pilots: every seed and control",
 "Completed saved FP32 duplicate evaluations | lower CE is better | distinct experiments",
 ["128-byte context, 3.072M byte exposures/trial; not FLOP/walltime matched or official word-level evaluation.",
  "Lines connect seeds for readability, not training time. No novelty, measured speed, or SOTA claim."],670)
 panels=[(70,510,"Pair-energy versus local controls",b["pair_energy"]["rows"],
          ["transformer","geometric_overlapping","relative_phase_key","pair_energy"],[3361,3407,3457]),
         (640,1060,"Cross-scale versus original hierarchy",b["hierarchy"]["paired"],
          ["original","local","cross"],[73,191,307])]
 for left,right,title,rows,arms,seeds in panels:
  c.text(left,120,title,17,weight="bold");top,bottom=235,550;lo,hi=1.55,1.85
  def y(v):return bottom-(v-lo)/(hi-lo)*(bottom-top)
  for k in range(7):
   v=lo+.05*k;c.line(left,y(v),right,y(v));c.text(left-8,y(v)+4,f'{v:.2f}',12,anchor="end",color="#64748b")
  c.text(left,218,"Byte CE (nats/byte); axis starts at 1.55",12)
  for j,a in enumerate(arms):
   # Two rows of legend text, compact enough for each panel.
   x=left+(j%2)*220;ly=151+24*(j//2)
   c.line(x,ly-5,x+16,ly-5,COLORS[j],4);c.text(x+23,ly,a,12)
   vals=[next(r["validation"]["ce"] for r in rows if r["seed"]==s and r["arm"]==a) if "validation" in rows[0] else next(r["ce"][a] for r in rows if r["seed"]==s) for s in seeds]
   pts=[(left+50+i*(right-left-100)/2,y(v)) for i,v in enumerate(vals)]
   for (x1,y1),(x2,y2) in zip(pts,pts[1:]):c.line(x1,y1,x2,y2,COLORS[j],2)
   for x1,y1 in pts:c.parts.append(f'<circle cx="{x1}" cy="{y1}" r="5" fill="{COLORS[j]}"/>')
  for i,s in enumerate(seeds):c.text(left+50+i*(right-left-100)/2,579,str(s),13,anchor="middle")
 out["confirmed_byte_pilots.svg"]=c.finish()
 return out
def main():
 root=pathlib.Path(__file__).resolve().parents[1];h=json.loads((root/"data/historical_10m.json").read_text());b=json.loads((root/"data/confirmed_byte_pilots.json").read_text())
 out=root/"graphs";out.mkdir(exist_ok=True)
 for name,content in build_charts(h,b).items():(out/name).write_bytes(content)
 print("Rendered four charts from saved numbers only; no model evaluation.")
if __name__=="__main__":main()
