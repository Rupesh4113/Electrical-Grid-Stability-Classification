"""CLI prediction utility for Electrical Grid Stability Classification."""

import argparse
import json
import sys
from pathlib import Path

from config.config import INPUT_FEATURES
from src.prediction import predict_stability


def main():
    parser = argparse.ArgumentParser(
        description="Predict Electrical Grid Stability (0 = Stable, 1 = Unstable)."
    )
    parser.add_argument(
        "--json",
        type=str,
        help="JSON string containing all 12 input features (tau1..tau4, p1..p4, g1..g4).",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to JSON file containing sample inputs.",
    )

    # Individual arguments for manual testing
    for feat in INPUT_FEATURES:
        parser.add_argument(f"--{feat}", type=float, help=f"Value for {feat}")

    args = parser.parse_args()

    input_data = {}
    if args.json:
        try:
            input_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r") as f:
            input_data = json.load(f)
    else:
        # Check if all individual feature args provided
        provided = {f: getattr(args, f) for f in INPUT_FEATURES if getattr(args, f) is not None}
        if len(provided) == len(INPUT_FEATURES):
            input_data = provided
        else:
            # Default reference sample for demonstration if no args given
            print("No complete input provided. Running with sample reference grid parameters:")
            input_data = {
                "tau1": 2.959,
                "tau2": 3.079,
                "tau3": 8.381,
                "tau4": 9.780,
                "p1": 3.763,
                "p2": -1.527,
                "p3": -1.390,
                "p4": -0.845,
                "g1": 0.562,
                "g2": 0.413,
                "g3": 0.778,
                "g4": 0.958,
            }
            print(json.dumps(input_data, indent=2))

    try:
        result = predict_stability(input_data)
        print("\n" + "=" * 45)
        print(f"PREDICTION RESULT: {result['label'].upper()} ({result['prediction']})")
        print("=" * 45)
        print(f"Probability Stable:   {result['probability_stable'] * 100:.2f}%")
        print(f"Probability Unstable: {result['probability_unstable'] * 100:.2f}%")
        print(f"Model:                {result['model']}")
        print("=" * 45 + "\n")
    except Exception as e:
        print(f"Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
