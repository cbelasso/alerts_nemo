"""
Convert existing training data to simplified format
"""

import json
from pathlib import Path


def convert_to_simplified_format(input_jsonl: str, output_jsonl: str):
    """Convert full format JSONL to simplified format."""

    converted = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            # Parse the output JSON string
            output_data = json.loads(record["output"])

            # Extract simplified format
            if output_data.get("has_alerts"):
                alert_types = [alert["alert_type"] for alert in output_data.get("alerts", [])]
                simplified_output = {"alert_types": alert_types, "non_alert_types": []}
            else:
                non_alert = (
                    [output_data["non_alert_classification"]]
                    if output_data.get("non_alert_classification")
                    else []
                )
                simplified_output = {"alert_types": [], "non_alert_types": non_alert}

            converted.append(
                {
                    "input": record["input"],
                    "output": json.dumps(simplified_output, ensure_ascii=False),
                }
            )

    # Write simplified version
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(converted)} records")
    print(f"   From: {input_jsonl}")
    print(f"   To:   {output_jsonl}")


if __name__ == "__main__":
    input_file = "/home/clyde/workspace/alerts_detection_llama/scripts/finetuning/training_data/alerts_training_20251202_224001.jsonl"
    output_file = "/home/clyde/workspace/alerts_detection_llama/scripts/finetuning/training_data/alerts_training_20251202_224001_simplified.jsonl"

    convert_to_simplified_format(input_file, output_file)
