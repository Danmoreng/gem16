"""Plot validated first-ID and post-first boundaries for selected cohorts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent
COLD = (("cold-4k", "4K"), ("cold-16k", "16K"), ("cold-64k", "64K"),
        ("cold-127.5k", "127.5K"))
WARM = (("warm-16k", "12K + 4K"), ("warm-127.5k", "120K + 7.5K"),
        ("warm-252k", "240K + 12K"))
ENGINES = {
    "GemSixteen": ("#006d61", "o"),
    "llama.cpp": ("#b34e19", "s"),
    "vLLM": ("#5842a4", "D"),
}


def make_figure(rows):
    by_engine = {(row["engine"], row["case"]): row for row in rows}
    assert len(by_engine) == len(rows), "each plotted case must select a single cohort per engine"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
        "svg.hashsalt": "gem16-12b-2026-11-17",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#aab2b8",
        "axes.labelcolor": "#273443", "xtick.color": "#344456", "ytick.color": "#344456",
        "text.color": "#202c38",
    })
    figure, panels = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex="col", sharey="row")
    figure.subplots_adjust(left=.10, right=.97, bottom=.13, top=.78, wspace=.10, hspace=.24)
    figure.text(.10, .94, "Gemma 4 12B IT on RTX 5080 Laptop", size=23, weight="bold")
    figure.text(.10, .895, "Batch 1 · 128 generated IDs · sampled D3 · median of retained requests",
                color="#4e6175", size=12)

    for column, (cohort, title, xlabel) in enumerate((
        (COLD, "Cold prefill", "Prompt IDs"),
        (WARM, "Warm append", "Requested cache prefix + appended IDs"),
    )):
        panels[0, column].set_title(title, loc="left", weight="bold")
        panels[1, column].set_xlabel(xlabel, labelpad=7)
        panels[1, column].set_xticks(range(len(cohort)), labels=[label for _, label in cohort])
        for row_index, (field, divisor) in enumerate((
            ("first_id_ms", 1000), ("ids_per_second_after_first_id", 1),
        )):
            axis = panels[row_index, column]
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=.28, linestyle="-", linewidth=.7)
            if row_index == 0:
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100, 300]))
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g} s"))
            for engine, (color, marker) in ENGINES.items():
                values = []
                for case, _ in cohort:
                    value = by_engine.get((engine, case), {}).get(field)
                    values.append(value / divisor if value is not None else float("nan"))
                axis.plot(range(len(cohort)), values, marker=marker, color=color,
                          linewidth=1.7, markersize=7, label=engine)
        panels[0, column].set_xlim(-.17, len(cohort) - .83)

    panels[0, 0].set_ylabel("Time to first ID (log scale)")
    panels[1, 0].set_ylabel("Post-first decode (IDs/s)")
    panels[1, 0].set_ylim(bottom=0)
    handles, labels = panels[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(.094, .863),
                  ncol=3, frameon=False, borderaxespad=0)
    figure.text(.10, .035, "Missing points lack a validated measurement for the timing boundary. "
                "Warm points exclude primer latency.", size=9.5, color="#4e6175")
    output = ROOT / "comparison.svg"
    figure.savefig(output, metadata={"Date": None}, facecolor="white")
    # Matplotlib emits whitespace at the end of SVG path data lines.
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8", newline="\n")
    plt.close(figure)


if __name__ == "__main__":
    result = json.loads((ROOT / "comparison.json").read_text())
    selected = [row for row in result["cells"] if row.get("primary_comparison")]
    make_figure(selected)
