import argparse

from experiments.experiment.parse_experiments import ExperimentParser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments defined in a CSV file."
    )
    parser.add_argument(
        "source_file",
        type=str,
        help="Path to the CSV file containing experiment configurations.",
    )
    args = parser.parse_args()

    experiment_parser = ExperimentParser(source_file=args.source_file)
    experiment_parser.perform_experiments()


if __name__ == "__main__":
    main()
