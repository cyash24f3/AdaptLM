"""Publish an explicit free ZeroGPU Space from an allowlisted source bundle."""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

from adaptlm.artifacts.manifest import environment, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="cyash1204/AdaptLM")
    parser.add_argument("--adapter", type=Path, default=Path("models/main-adapter-214"))
    args = parser.parse_args()
    if not (args.adapter / "bundle.json").is_file():
        raise ValueError("exported genuine adapter required")
    api = HfApi()
    if args.space.split("/")[0] != api.whoami()["name"]:
        raise ValueError("publish only under the authenticated personal account")
    api.create_repo(
        args.space,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        space_hardware=SpaceHardware.ZERO_A10G,
        exist_ok=True,
    )
    runtime = api.get_space_runtime(args.space)
    if runtime.requested_hardware not in (None, SpaceHardware.ZERO_A10G):
        raise ValueError("refusing to alter a Space with a different hardware configuration")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for directory in ["src", "configs", "datasets", "reports"]:
            shutil.copytree(
                directory, root / directory, ignore=shutil.ignore_patterns("__pycache__")
            )
        for name in ["LICENSE", "NOTICE", "pyproject.toml", "uv.lock"]:
            shutil.copyfile(name, root / name)
        for name in ["app.py", "README.md", "requirements.txt"]:
            shutil.copyfile(Path("deployment/huggingface") / name, root / name)
        (root / "bundle").mkdir()
        for file in args.adapter.iterdir():
            if file.is_file() and file.name not in ["progress.json", "README.md"]:
                shutil.copyfile(file, root / "bundle" / file.name)
        for name in ["LICENSE", "NOTICE"]:
            shutil.copyfile(name, root / "bundle" / name)
        shutil.copyfile("docs/model-card.md", root / "bundle/README.md")
        result = api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=root,
            commit_message="Publish audited source, genuine adapter and public fictional reports on free ZeroGPU",
        )
    write_json(
        Path("reports/cloud-deployment.json"),
        environment()
        | {
            "provider": "Hugging Face",
            "space_id": args.space,
            "url": "https://huggingface.co/spaces/" + args.space,
            "requested_hardware": "zero-a10g",
            "paid_upgrade": False,
            "status": "uploaded; runtime verification required",
            "commit": result.oid,
        },
    )
    print(result.oid)


if __name__ == "__main__":
    main()
