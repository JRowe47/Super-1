"use strict";
const $=id=>document.getElementById(id);
const tensor=$('tensor'),basis=$('basis'),component=$('component'),direction=$('direction');
const names=[...new Set(DATA.catalog.map(c=>c.tensor))];
for(const name of names){let o=document.createElement('option');o.value=name;o.textContent=name;tensor.appendChild(o);}
tensor.value='l1.weight';
function label(c){
 if(c.basis==='haar')return `${c.kind} · level ${c.level}, index ${c.index} · weights ${c.start}–${c.end-1}`;
 if(c.basis==='fourier')return `bins ${c.low}–${c.high-1} · depth ${c.depth}`;
 return `branch ${c.path||'root'} · depth ${c.depth}`;
}
function choices(){
 component.textContent='';
 for(const c of DATA.catalog.filter(c=>c.tensor===tensor.value&&c.basis===basis.value)){
  let o=document.createElement('option');o.value=c.id;o.textContent=label(c);component.appendChild(o);
 }
 render();
}
function lineChart(series){
 const canvas=$('curve'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,p=48;
 ctx.clearRect(0,0,w,h);let flat=series.flatMap(s=>s.v),lo=Math.min(...flat),hi=Math.max(...flat);
 if(hi-lo<1e-10){lo-=1;hi+=1;}let margin=.1*(hi-lo);lo-=margin;hi+=margin;
 const xx=(i,n)=>p+(w-2*p)*i/Math.max(1,n-1),yy=v=>h-p-(h-2*p)*(v-lo)/(hi-lo);
 ctx.font='13px system-ui';ctx.fillStyle='#526371';ctx.strokeStyle='#dce5e9';ctx.lineWidth=1;
 for(let k=0;k<5;k++){let v=lo+(hi-lo)*k/4,y=yy(v);ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(w-p,y);ctx.stroke();ctx.fillText(v.toFixed(2),3,y+4);}
 ctx.fillText('Native flat index',w/2-45,h-8);ctx.fillText('0',p,h-25);ctx.fillText(String(series[0].v.length-1),w-p,h-25);
 for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();s.v.forEach((v,i)=>i===0?ctx.moveTo(xx(i,s.v.length),yy(v)):ctx.lineTo(xx(i,s.v.length),yy(v)));ctx.stroke();
  if(s.v.length===1){ctx.beginPath();ctx.arc(xx(0,1),yy(s.v[0]),3,0,Math.PI*2);ctx.fillStyle=s.color;ctx.fill();}}
}
function render(){
 const id=Number(component.value),c=DATA.catalog[id],rb=direction.value==='rollback',before=rb?DATA.new[c.tid]:DATA.old[c.tid];
 const delta=DATA.curves[id],after=before.map((v,j)=>v+(rb?-1:1)*delta[j]);
 const logits=rb?DATA.rollback[id]:DATA.transplant[id],base=rb?DATA.after_logits:DATA.before_logits;
 $('description').textContent=`Component ${id}: ${c.tensor}, ${label(c)}. ${rb?'Subtracting':'Adding'} this component of the step 0→800 update.`;
 lineChart([{v:before,color:'#2563a6'},{v:after,color:'#cf6232'},{v:delta,color:'#168773'}]);
 const acc=a=>[0,1,2].map(o=>100*a.filter((r,j)=>(r[o]>0)===(DATA.targets[j][o]>0)).length/16);
 const orig=acc(base),changed=acc(logits),labels=['Readout','Parity','Context'];
 $('scores').innerHTML='<table><tr><th>Behavior</th><th>Before</th><th>After</th></tr>'+labels.map((t,j)=>`<tr><td>${t}</td><td>${orig[j]}%</td><td>${changed[j]}%</td></tr>`).join('')+'</table>';
 $('truth').innerHTML='<table><tr><th>Inputs a b c d</th><th>Readout q</th><th>Parity q</th><th>Context q</th></tr>'+DATA.inputs.map((x,j)=>'<tr><td>'+x.map(v=>v>0?'+':'−').join(' ')+'</td>'+[0,1,2].map(o=>{
  const old=Math.tanh(base[j][o]/2),q=Math.tanh(logits[j][o]/2),bad=(logits[j][o]>0)!==(DATA.targets[j][o]>0);
  return `<td style="background:${bad?'#ffe7dd':'#f0f7f5'}">${old.toFixed(4)} → ${q.toFixed(4)}${bad?' ✕':''}</td>`;
 }).join('')+'</tr>').join('')+'</table>';
}
tensor.addEventListener('change',choices);basis.addEventListener('change',choices);component.addEventListener('change',render);direction.addEventListener('change',render);
choices();component.value='197';render();
