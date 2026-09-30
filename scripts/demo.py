"""Deterministic credential-free HTTP walkthrough, explicitly fixture-only."""

import argparse

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()
with httpx.Client(base_url=args.url, timeout=60) as client:
    metadata = client.get("/api/v1/models").raise_for_status().json()
    if metadata["profile"] != "fixture":
        raise ValueError("this script is a fixture walkthrough, not a model quality test")
    print(metadata["warning"])
    for message in metadata["demo_messages"]:
        response = client.post("/api/v1/triage", json={"message": message, "mode": "fixture"})
        response.raise_for_status()
        print(message, "=>", response.json()["status"], "raw_valid:", response.json()["raw_valid"])
    comparison = client.post("/api/v1/compare", json={"message": metadata["demo_messages"][0]})
    print("Compare statuses:", [r["status"] for r in comparison.json()["results"]])
