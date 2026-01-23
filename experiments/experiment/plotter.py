from dataclasses import dataclass

import pandas as pd
from matplotlib import pyplot as plt


@dataclass(frozen=True, slots=True)
class PlotterOptions:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    output_path: str | None = None


def plot_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    options: PlotterOptions | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], marker="o")

    if options is not None:
        if options.title:
            plt.title(options.title)
        if options.x_label:
            plt.xlabel(options.x_label)
        if options.y_label:
            plt.ylabel(options.y_label)

    plt.grid(True)

    if options and options.output_path:
        plt.savefig(options.output_path)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # TODO: generate plots from our csv result files
    pass
