"use strict";
const $ = id => document.getElementById(id);
let runtime = null, token = "", offset = 0;
const make = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
async function api(url, options={}) {
  const response = await fetch(url, { ...options, headers: {"Content-Type":"application/json", ...(token ? {Authorization:`Bearer ${token}`} : {}), ...options.headers} });
  const data = await response.json();
  if (!response.ok && !data.status) throw new Error(data.error?.detail || "Runtime unavailable");
  return data;
}
function showError(target, error) { target.replaceChildren(make("div", error.message || String(error), "error")); }
function evidence(message, spans) {
  const box = make("div", undefined, "evidence");
  // JS indexes UTF-16. Convert to code points to match Python Unicode offsets.
  const chars = Array.from(message); let start=0;
  const sorted = [...spans].sort((a,b)=>a.start-b.start);
  for(const span of sorted) { if(span.start<start) continue; box.append(document.createTextNode(chars.slice(start,span.start).join(""))); box.append(make("mark",chars.slice(span.start,span.end).join(""))); start=span.end; }
  box.append(document.createTextNode(chars.slice(start).join(""))); return box;
}
function renderRecord(target, response, message) {
  target.replaceChildren();
  if(!response.output) { target.append(make("div",`${response.status.replaceAll("_"," ")}: ${response.error || "No validated record"}`,"error")); return; }
  const record=response.output;
  target.append(make("h3", record.primary_issue?.replaceAll("_"," ") || "Issue unresolved"));
  target.append(make("span", record.triage_status.replaceAll("_"," "), "status-pill"));
  target.append(make("h4","SOURCE EVIDENCE"));
  target.append(evidence(message,[...record.entities.map(e=>e.span),...record.summary_claims.flatMap(c=>c.spans)]));
  target.append(make("h4","SUMMARY CLAIMS"));
  for(const claim of record.summary_claims) target.append(make("p",claim.text));
  target.append(make("h4","STATED ENTITIES"));
  if(!record.entities.length) target.append(make("p","No entities stated."));
  for(const entity of record.entities) target.append(make("span",`${entity.type}: ${entity.value} [${entity.span.start}:${entity.span.end}]`,"entity-chip"));
  target.append(make("h4","MISSING INFORMATION"));
  for(const item of record.missing_information) target.append(make("div",item.question,"missing-row"));
  if(!record.missing_information.length) target.append(make("p","No operational questions recorded."));
  const timing=make("div",undefined,"timing");
  timing.append(make("span",`RAW VALID: ${response.raw_valid ? "YES":"NO"}`));
  timing.append(make("span",`REPAIRS: ${response.repair_count}`));
  timing.append(make("span",`${(response.timings.total_seconds||0).toFixed(2)}s · ${response.mode}`));
  if(response.usage) timing.append(make("span",`${response.usage.output_tokens} output tokens`));
  target.append(timing);
}
document.querySelectorAll(".nav").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll(".nav").forEach(el=>el.classList.toggle("active",el===button));
  document.querySelectorAll(".view").forEach(el=>el.hidden=el.id!==button.dataset.view);
  if(button.dataset.view==="experiments" && token) loadReports();
}));
$("message").addEventListener("input",()=>$("character-count").textContent=`${Array.from($("message").value).length} / 3000`);
$("run").addEventListener("click",async()=>{
  const button=$("run");button.disabled=true;$("result-state").textContent="RUNNING";
  $("result").replaceChildren(make("div","Waiting for the configured runtime…","empty"));
  try { const message=$("message").value; const response=await api("/api/v1/triage",{method:"POST",body:JSON.stringify({message,mode:$("mode").value})});renderRecord($("result"),response,message);$("result-state").textContent=response.status.toUpperCase(); }
  catch(error) {showError($("result"),error);$("result-state").textContent="FAILED";}
  finally {button.disabled=false;}
});
$("compare-run").addEventListener("click",async()=>{
  $("compare-run").disabled=true;$("comparison").replaceChildren(make("p","Comparing configured models…"));
  try { const message=$("compare-message").value;const data=await api("/api/v1/compare",{method:"POST",body:JSON.stringify({message})});$("comparison").replaceChildren();
    for(const result of data.results) {const card=make("article",undefined,"panel");card.append(make("h2",result.mode.replaceAll("_"," ")));const output=make("div",undefined,"result");renderRecord(output,result,message);card.append(output);$("comparison").append(card);} }
  catch(error){showError($("comparison"),error);}finally{$("compare-run").disabled=false;}
});
async function loadDataset(){
  $("data-state").textContent="Loading…";
  try{const data=await api(`/api/v1/dataset?split=${$("split").value}&offset=${offset}`);$("data-state").textContent=`${data.total} examples · ${data.manifest.independent_scenario_families} total scenario families · human-unreviewed`;
    $("dataset-rows").replaceChildren();for(const row of data.rows){const details=make("details",undefined,"data-row");details.append(make("summary",`${row.id} · ${row.review_status}`));details.append(make("p",`${row.source} / ${row.family_id} / ${row.split}`));details.append(evidence(row.message,row.target.entities.map(e=>e.span)));details.append(make("pre",JSON.stringify(row.target,null,2)));$("dataset-rows").append(details);} }
  catch(error){$("data-state").textContent=error.message;}
}
$("unlock").addEventListener("click",()=>{token=$("admin-token").value;loadDataset();loadReports();});
$("split").addEventListener("change",()=>{offset=0;loadDataset();});
$("dataset-next").addEventListener("click",()=>{offset+=20;loadDataset();});
async function loadReports(){try{const data=await api("/api/v1/experiments");$("report-select").replaceChildren(make("option","Select a saved report"));$("report-select").firstChild.value="";for(const id of data.reports){const option=make("option",id);option.value=id;$("report-select").append(option);}$("report-state").textContent=`${data.reports.length} saved reports`;}catch(error){$("report-state").textContent=error.message;}}
$("report-select").addEventListener("change",async()=>{const id=$("report-select").value;if(!id)return;try{const data=await api(`/api/v1/experiments/${encodeURIComponent(id)}`);$("report-json").textContent=JSON.stringify(data,null,2);$("report-summary").replaceChildren();
  const metrics=data.metrics || (data.category_accuracy!==undefined ? {classifier:{category_accuracy_all_attempted:data.category_accuracy,category_macro_f1_all_attempted:data.category_macro_f1}} : {});
  for(const [mode,values] of Object.entries(metrics)){const grid=make("div",undefined,"metric-grid");grid.append(make("h3",mode.replaceAll("_"," ")));for(const key of ["category_accuracy_all_attempted","raw_schema_validity","complete_record_reference_success"]){if(values[key]!==undefined){const box=make("div",undefined,"metric-box");box.append(make("strong",values[key]===null?"unmeasured":`${(values[key]*100).toFixed(1)}%`));box.append(make("span",key.replaceAll("_"," ")));grid.append(box);}}$("report-summary").append(grid);}
  $("report-state").textContent=data.interpretation || "Saved configuration / measurements";}catch(error){$("report-state").textContent=error.message;}});
(async()=>{try{runtime=await api("/api/v1/models");$("profile-badge").textContent=runtime.profile==="fixture"?"FIXTURE DEMO":`LOCAL MODEL / ${runtime.device.toUpperCase()}`;$("schema-label").textContent=runtime.schema_version;
  $("notice").textContent=runtime.profile==="fixture"?"Fixture mode · These documented responses test the interface and contracts. They are not model inference or benchmark results.":`${runtime.base_model_id} · ${runtime.device} / ${runtime.dtype} · ${runtime.adapter_bundle_id || "Adapter unavailable"}`;
  $("mode").replaceChildren();for(const mode of runtime.available_modes){const option=make("option",mode.replaceAll("_"," "));option.value=mode;$("mode").append(option);}
  for(const [i,message] of runtime.demo_messages.entries()){const button=make("button",["Duplicate charge","Delivery problem","Account access"][i]||`Example ${i+1}`);button.addEventListener("click",()=>{$("message").value=message;$("message").dispatchEvent(new Event("input"));});$("samples").append(button);}
  const example=runtime.demo_messages[0]||"Charged twice for my subscription yesterday. I don't have the transaction ID.";$("message").value=example;$("compare-message").value=example;$("message").dispatchEvent(new Event("input"));
}catch(error){$("profile-badge").textContent="UNAVAILABLE";$("notice").textContent=error.message;$("run").disabled=true;}})();
