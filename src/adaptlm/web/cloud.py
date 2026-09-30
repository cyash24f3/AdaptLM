"""Escaped presentation of actual inference results for the public cloud UI."""

from html import escape


CSS = """
.gradio-container {max-width:1120px!important; margin:auto!important;}
.adapt-result {background:#fff;border:1px solid #dce3db;border-radius:14px;padding:24px;color:#243429;min-height:170px;}
.adapt-result h3 {font-size:21px;margin:12px 0;}
.adapt-result h4 {margin:20px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#637469;}
.adapt-result p {line-height:1.65;}
.adapt-result li {line-height:1.7;margin-bottom:8px;}
.adapt-result small {color:#637469;}
.adapt-tag {display:inline-block;padding:4px 10px;border-radius:30px;background:#ecf2e8;font-size:12px;font-weight:600;}
.adapt-error .adapt-tag {background:#fff0dc;color:#895822;}
.adapt-source {border-left:3px solid #a7bda0;padding:10px 14px;background:#f7f9f5;}
"""


def result_html(response):
    status = escape(response["status"].replace("_", " "))
    record = response.get("output")
    mode = escape(response["mode"].replace("_", " "))
    usage = response.get("usage")
    timing = response.get("timings", {}).get("total_seconds")
    measured = (
        f"{timing:.2f}s · {usage['output_tokens']} generated tokens"
        if timing is not None and usage
        else "No generation timing available"
    )
    if record is None:
        error = escape(response.get("error") or "The request could not produce a validated record.")
        return f'<div class="adapt-result adapt-error"><span class="adapt-tag">{status}</span><h3>No validated record</h3><p>{error}</p><p>The app keeps failed output separate from usable triage.</p><small>{mode} · {measured}</small></div>'
    issue = escape((record["primary_issue"] or "Unresolved issue").replace("_", " ").title())
    claims = "".join(
        f'<p class="adapt-source">{escape(c["text"])}</p>' for c in record["summary_claims"]
    )
    entities = "".join(
        f"<li><strong>{escape(e['type'].replace('_', ' '))}</strong>: {escape(e['value'])}</li>"
        for e in record["entities"]
    )
    missing = "".join(f"<li>{escape(m['question'])}</li>" for m in record["missing_information"])
    raw = "passed" if response["raw_valid"] else "failed; repaired output passed"
    return f'<div class="adapt-result"><span class="adapt-tag">{status}</span><h3>{issue}</h3><h4>Summary claims</h4>{claims}<h4>Stated entities</h4><ul>{entities or "<li>None stated in this record.</li>"}</ul><h4>Details to request</h4><ul>{missing or "<li>No additional fields requested.</li>"}</ul><small>{mode} · {measured}<br>Raw source/schema validation {raw}. Semantic correctness remains unreviewed.</small></div>'


def report_markdown(value):
    if "metrics" in value:
        lines = [
            "| Mode | Attempts | Raw validity | Category agreement | Complete reference agreement |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode, metrics in value["metrics"].items():

            def percentage(key):
                v = metrics.get(key)
                return f"{v:.1%}" if v is not None else "—"

            lines.append(
                f"| {mode.replace('_', ' ')} | {metrics['attempted']} | {percentage('raw_schema_validity')} | {percentage('category_accuracy_all_attempted')} | {percentage('complete_record_reference_success')} |"
            )
        return (
            "\n".join(lines)
            + "\n\nAgreement with fictional, unreviewed reference labels; invalid outputs count as failures. Complete reference agreement requires exact field/span matching and is not semantic accuracy."
        )
    if "category_accuracy" in value:
        return f"**Category-only classifier:** {value['category_accuracy']:.1%} agreement, macro-F1 {value['category_macro_f1']:.3f}, {value['attempted']} applicable test rows. Does not produce complete triage records."
    return f"**Actual training:** {value.get('status', 'unknown')}; {value.get('optimizer_steps', '—')} optimizer steps; {value.get('duration_seconds', 0) / 60:.1f} minutes on MPS. Training loss is not generated-output quality."
