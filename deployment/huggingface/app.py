"""Public, quota-limited cloud demo. No training, uploads, or paid inference calls."""

import asyncio
import json
import sys
from pathlib import Path

import spaces  # Must initialize ZeroGPU before importing PyTorch/model code.
import gradio as gr

# Spaces installs requirements before copying source; use the committed src layout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from adaptlm.config import Settings  # noqa: E402
from adaptlm.data.pipeline import read_rows  # noqa: E402
from adaptlm.inference.engine import TransformersEngine  # noqa: E402
from adaptlm.inference.service import InferenceService  # noqa: E402
from adaptlm.web.cloud import CSS, report_markdown, result_html  # noqa: E402

settings = Settings(
    profile="local",
    device="cuda",
    dtype="bfloat16",
    attention_implementation="sdpa",
    adapter_path=Path("bundle"),
    max_chars=3000,
    max_new_tokens=384,
    _env_file=None,
)
engine = TransformersEngine(settings, revalidate_torch_version="2.13.0")


@spaces.GPU(duration=60)
def triage(message, mode):
    engine.revalidate_runtime()
    response = asyncio.run(InferenceService(settings, engine).triage(message, mode))
    return result_html(response), response


@spaces.GPU(duration=60)
def compare(message):
    engine.revalidate_runtime()

    async def run():
        service = InferenceService(settings, engine)
        return {
            mode: await service.triage(message, mode)
            for mode in ("zero_shot", "few_shot", "adapted")
        }

    responses = asyncio.run(run())
    return *(result_html(r) for r in responses.values()), responses


def report(name):
    path = REPORTS[name]
    value = json.loads(path.read_text())
    return report_markdown(value), value


REPORTS = {
    str(p.relative_to("reports")): p
    for p in sorted(Path("reports").rglob("*.json"))
    if p.name in ("report.json", "classifier.json", "training-main-214-run.json")
}
rows = read_rows(Path("datasets/validation.jsonl"))
sample = "A corner of the new monitor is smashed."
initial_report = (
    "test-final/report.json" if "test-final/report.json" in REPORTS else next(iter(REPORTS))
)
quality_note = "The adapter often produces incorrect source offsets; invalid output is rejected. "
if "test-final/report.json" in REPORTS:
    measured = json.loads(REPORTS["test-final/report.json"].read_text())["metrics"]["adapted"]
    quality_note = (
        f"**Held-out adapter records passing structural validation: {measured['valid_outputs']}/{measured['attempted']}.** "
        "The displayed monitor message is a selected validation demo. Invalid output is rejected. "
    )
with gr.Blocks(
    title="AdaptLM · Support triage lab",
    analytics_enabled=False,
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        "# AdaptLM\nA support triage lab with a **real M5-trained LoRA adapter**. "
        "[Source & measured results](https://github.com/cyash24f3/AdaptLM)\n\n"
        "Live cloud generation uses free Hugging Face ZeroGPU and visitor quotas. "
        "[Sign in to Hugging Face](https://huggingface.co/login) for the free account quota "
        "if anonymous runs are exhausted. "
        "Labels are fictional and independently unreviewed. "
        + quality_note
        + "This is a research demo. "
        "Use fictional messages here: input is sent to Hugging Face's infrastructure. "
        "No training or customer actions occur in this app."
    )
    with gr.Tab("Triage"):
        message = gr.Textbox(
            label="Fictional support message", lines=5, max_length=3000, value=sample
        )
        mode = gr.Dropdown(
            ["zero_shot", "few_shot", "adapted"], value="adapted", label="Model mode"
        )
        button = gr.Button("Run triage", variant="primary")
        result = gr.HTML(
            '<div class="adapt-result"><h3>Turn a message into a triage record</h3><p>Choose a model and run a fictional example. Free GPU queues and daily quotas apply.</p></div>'
        )
        gr.Examples(
            [[sample], [rows[0]["message"]], ["Charged twice yesterday."]],
            inputs=message,
            label="Fictional examples · full failure rates are in Experiments",
        )
        with gr.Accordion("Response details and measured runtime", open=False):
            trace = gr.JSON(label="Actual response · source checks are not semantic review")
        button.click(
            triage,
            [message, mode],
            [result, trace],
            api_name="triage",
            concurrency_id="model",
            concurrency_limit=1,
        )
    with gr.Tab("Compare"):
        comparison_message = gr.Textbox(
            label="Same input for all three models", lines=5, max_length=3000, value=sample
        )
        compare_button = gr.Button("Compare all modes", variant="primary")
        with gr.Row():
            columns = []
            for title in ["Zero-shot base", "Few-shot base", "M5-trained adapter"]:
                with gr.Column():
                    gr.Markdown("### " + title)
                    columns.append(
                        gr.HTML('<div class="adapt-result"><p>Awaiting comparison.</p></div>')
                    )
        with gr.Accordion("Comparison response details", open=False):
            comparison = gr.JSON(label="Actual responses under identical input and budgets")
        compare_button.click(
            compare,
            comparison_message,
            [*columns, comparison],
            api_name="compare",
            concurrency_id="model",
            concurrency_limit=1,
        )
    with gr.Tab("Dataset"):
        gr.Markdown(
            "1,280 authored rows / 160 scenario families; frozen train 880, validation 160, test 240. No human-reviewed labels. This read-only view contains public fictional validation examples only."
        )
        gr.Dataframe(
            headers=["ID", "Family", "Message", "Issue", "Review"],
            value=[
                [
                    r["id"],
                    r["family_id"],
                    r["message"],
                    r["target"]["primary_issue"],
                    r["review_status"],
                ]
                for r in rows[:20]
            ],
            interactive=False,
            wrap=True,
        )
    with gr.Tab("Experiments"):
        gr.Markdown(
            "Actual local MPS experiments. Cloud CUDA timings are separate from these reports. Semantic summary support is unmeasured."
        )
        selection = gr.Dropdown(list(REPORTS), value=initial_report, label="Published report")
        initial_summary, initial_record = report(initial_report)
        summary = gr.Markdown(initial_summary)
        with gr.Accordion("Full evidence and provenance", open=False):
            saved = gr.JSON(value=initial_record, label="Measured evidence")
        selection.change(report, selection, [summary, saved], api_name=False, queue=False)
    with gr.Accordion("Runtime and reproducibility", open=False):
        gr.JSON(value=engine.metadata(), label="Pinned weights, adapter, prompt and runtime")

demo.queue(max_size=3, default_concurrency_limit=1).launch(
    # Core inference failures are already redacted; surface provider quota errors.
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
)
