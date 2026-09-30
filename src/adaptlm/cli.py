import argparse
import asyncio
import json
from pathlib import Path

from adaptlm.config import Settings


def main():
    parser = argparse.ArgumentParser(description="AdaptLM: local/free support triage experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data")
    data.add_argument("action", choices=["generate", "validate"])
    data.add_argument("--root", type=Path, default=Path("datasets"))
    classifier = sub.add_parser("classifier")
    classifier.add_argument("--output", type=Path, default=Path("reports/classifier.json"))
    trainer = sub.add_parser("train")
    trainer.add_argument("--config", type=Path, default=Path("configs/train-m5.json"))
    trainer.add_argument("--output", type=Path, default=Path("models/main-adapter"))
    trainer.add_argument("--resume", type=Path)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluator.add_argument(
        "--modes",
        nargs="+",
        choices=["zero_shot", "few_shot", "adapted"],
        default=["zero_shot", "few_shot", "adapted"],
    )
    evaluator.add_argument("--variant", type=int)
    evaluator.add_argument("--limit", type=int)
    evaluator.add_argument("--output", type=Path, required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--output", type=Path, default=Path("reports/serving.json"))
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--concurrency", type=int, default=1)
    server = sub.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "data":
        from adaptlm.data.pipeline import generate, validate

        result = generate(args.root) if args.action == "generate" else validate(args.root)
    elif args.command == "classifier":
        from adaptlm.baselines.classifier import run

        result = run(settings.data_dir, args.output)
    elif args.command == "train":
        from adaptlm.training.runner import train

        result = train(args.config, args.output, args.resume)
    elif args.command == "evaluate":
        from adaptlm.evaluation.runner import evaluate

        result = asyncio.run(
            evaluate(settings, args.split, args.modes, args.output, args.variant, args.limit)
        )
    elif args.command == "benchmark":
        from adaptlm.evaluation.runner import benchmark

        result = asyncio.run(benchmark(settings, args.output, args.repeats, args.concurrency))
    else:
        import uvicorn

        uvicorn.run("adaptlm.api.app:app", host=args.host, port=args.port)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
