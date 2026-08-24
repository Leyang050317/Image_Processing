from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================

DEEP_LEARNING_FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = DEEP_LEARNING_FOLDER / "results"


# =========================================================
# Experiment configuration
# =========================================================

EXPERIMENT_NAME = "baseline"


# =========================================================
# Result files
# =========================================================

CLASSIFICATION_REPORT_FILE = (
    RESULTS_FOLDER
    / f"{EXPERIMENT_NAME}_classification_report.csv"
)

OVERALL_METRICS_FILE = (
    RESULTS_FOLDER
    / f"{EXPERIMENT_NAME}_overall_metrics.csv"
)

FEATURE_HISTORY_FILE = (
    RESULTS_FOLDER
    / f"{EXPERIMENT_NAME}_feature_extraction_history.csv"
)

FINE_TUNE_HISTORY_FILE = (
    RESULTS_FOLDER
    / f"{EXPERIMENT_NAME}_fine_tuning_history.csv"
)


# =========================================================
# Validation
# =========================================================

def validate_files():

    required_files = [
        CLASSIFICATION_REPORT_FILE,
        OVERALL_METRICS_FILE,
        FEATURE_HISTORY_FILE,
        FINE_TUNE_HISTORY_FILE
    ]

    missing_files = [
        file
        for file in required_files
        if not file.exists()
    ]

    if missing_files:

        print(
            "\n[ERROR] Missing result file(s):"
        )

        for file in missing_files:

            print(
                f"  {file}"
            )

        raise FileNotFoundError(
            "Required result files are missing."
        )


# =========================================================
# 1. Per-class performance
# =========================================================

def plot_per_class_metrics():

    dataframe = pd.read_csv(
        CLASSIFICATION_REPORT_FILE
    )

    classes = dataframe[
        "class"
    ]

    precision = (
        dataframe[
            "precision"
        ]
        * 100
    )

    recall = (
        dataframe[
            "recall"
        ]
        * 100
    )

    f1_score = (
        dataframe[
            "f1_score"
        ]
        * 100
    )

    positions = list(
        range(
            len(classes)
        )
    )

    width = 0.25

    figure = plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        [
            position - width
            for position in positions
        ],
        precision,
        width=width,
        label="Precision"
    )

    plt.bar(
        positions,
        recall,
        width=width,
        label="Recall"
    )

    plt.bar(
        [
            position + width
            for position in positions
        ],
        f1_score,
        width=width,
        label="F1-score"
    )

    plt.xticks(
        positions,
        classes
    )

    plt.ylim(
        90,
        100
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Ripeness Class"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Per-Class Performance"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_per_class_metrics.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Generated: {output_file.name}"
    )


# =========================================================
# 2. Overall performance
# =========================================================

def plot_overall_metrics():

    dataframe = pd.read_csv(
        OVERALL_METRICS_FILE
    )

    metric_dictionary = dict(
        zip(
            dataframe["metric"],
            dataframe["value"]
        )
    )

    selected_metrics = {
        "Accuracy":
            metric_dictionary[
                "accuracy"
            ],

        "Macro Precision":
            metric_dictionary[
                "precision_macro"
            ],

        "Macro Recall":
            metric_dictionary[
                "recall_macro"
            ],

        "Macro F1":
            metric_dictionary[
                "f1_macro"
            ],

        "Weighted Precision":
            metric_dictionary[
                "precision_weighted"
            ],

        "Weighted Recall":
            metric_dictionary[
                "recall_weighted"
            ],

        "Weighted F1":
            metric_dictionary[
                "f1_weighted"
            ]
    }

    metric_names = list(
        selected_metrics.keys()
    )

    metric_values = [
        value * 100
        for value
        in selected_metrics.values()
    ]

    figure = plt.figure(
        figsize=(11, 6)
    )

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.ylim(
        90,
        100
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Metric"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Overall Model Performance"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_overall_performance.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Generated: {output_file.name}"
    )


# =========================================================
# Combine Stage 1 + Stage 2 histories
# =========================================================

def load_combined_history():

    stage_one = pd.read_csv(
        FEATURE_HISTORY_FILE
    )

    stage_two = pd.read_csv(
        FINE_TUNE_HISTORY_FILE
    )

    stage_one["global_epoch"] = (
        range(
            1,
            len(stage_one) + 1
        )
    )

    stage_two["global_epoch"] = (
        range(
            len(stage_one) + 1,
            len(stage_one)
            + len(stage_two)
            + 1
        )
    )

    return (
        stage_one,
        stage_two
    )


# =========================================================
# 3. Accuracy learning curve
# =========================================================

def plot_training_accuracy():

    (
        stage_one,
        stage_two
    ) = load_combined_history()

    figure = plt.figure(
        figsize=(11, 6)
    )

    # Stage 1
    plt.plot(
        stage_one[
            "global_epoch"
        ],
        stage_one[
            "accuracy"
        ] * 100,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        stage_one[
            "global_epoch"
        ],
        stage_one[
            "val_accuracy"
        ] * 100,
        marker="o",
        label="Validation Accuracy"
    )

    # Stage 2
    plt.plot(
        stage_two[
            "global_epoch"
        ],
        stage_two[
            "accuracy"
        ] * 100,
        marker="o"
    )

    plt.plot(
        stage_two[
            "global_epoch"
        ],
        stage_two[
            "val_accuracy"
        ] * 100,
        marker="o"
    )

    transition_epoch = len(
        stage_one
    )

    plt.axvline(
        x=transition_epoch + 0.5,
        linestyle="--",
        label="Fine-Tuning Begins"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_training_accuracy.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Generated: {output_file.name}"
    )


# =========================================================
# 4. Loss learning curve
# =========================================================

def plot_training_loss():

    (
        stage_one,
        stage_two
    ) = load_combined_history()

    figure = plt.figure(
        figsize=(11, 6)
    )

    # Stage 1
    plt.plot(
        stage_one[
            "global_epoch"
        ],
        stage_one[
            "loss"
        ],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        stage_one[
            "global_epoch"
        ],
        stage_one[
            "val_loss"
        ],
        marker="o",
        label="Validation Loss"
    )

    # Stage 2
    plt.plot(
        stage_two[
            "global_epoch"
        ],
        stage_two[
            "loss"
        ],
        marker="o"
    )

    plt.plot(
        stage_two[
            "global_epoch"
        ],
        stage_two[
            "val_loss"
        ],
        marker="o"
    )

    transition_epoch = len(
        stage_one
    )

    plt.axvline(
        x=transition_epoch + 0.5,
        linestyle="--",
        label="Fine-Tuning Begins"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_training_loss.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Generated: {output_file.name}"
    )


# =========================================================
# 5. Generalization gap
# =========================================================

def plot_generalization_gap():

    (
        stage_one,
        stage_two
    ) = load_combined_history()

    combined = pd.concat(
        [
            stage_one,
            stage_two
        ],
        ignore_index=True
    )

    gap = (
        combined[
            "accuracy"
        ]
        -
        combined[
            "val_accuracy"
        ]
    ) * 100

    figure = plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        combined[
            "global_epoch"
        ],
        gap,
        marker="o"
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    transition_epoch = len(
        stage_one
    )

    plt.axvline(
        x=transition_epoch + 0.5,
        linestyle="--"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Train - Validation Accuracy (%)"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Generalization Gap"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_generalization_gap.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Generated: {output_file.name}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATING PERFORMANCE VISUALIZATIONS"
    )

    print(
        "=" * 70
    )

    validate_files()

    plot_per_class_metrics()

    plot_overall_metrics()

    plot_training_accuracy()

    plot_training_loss()

    plot_generalization_gap()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VISUALIZATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nGraphs saved to:"
        f"\n{RESULTS_FOLDER}"
    )


if __name__ == "__main__":
    main()