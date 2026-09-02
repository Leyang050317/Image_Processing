from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Experiment configuration
# =========================================================
#
# Change ONLY this value when generating graphs
# for a different experiment.
#
# Available examples:
#
# "hy"
# "ly"
# "kw"
# "lw"
# "baseline"
#
# =========================================================

EXPERIMENT_NAME = "lw"


# =========================================================
# Paths
# =========================================================

DEEP_LEARNING_FOLDER = Path(__file__).resolve().parent

RESULTS_ROOT = (
    DEEP_LEARNING_FOLDER
    / "results"
)

RESULTS_FOLDER = (
    RESULTS_ROOT
    / f"{EXPERIMENT_NAME}_results"
)


# =========================================================
# Input result files
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
# Validate required files
# =========================================================

def validate_files():
    """
    Check that all files required for visualization
    exist before generating graphs.
    """

    required_files = [
        CLASSIFICATION_REPORT_FILE,
        OVERALL_METRICS_FILE,
        FEATURE_HISTORY_FILE,
        FINE_TUNE_HISTORY_FILE
    ]

    missing_files = []

    for file_path in required_files:

        if not file_path.exists():

            missing_files.append(
                file_path
            )

    if missing_files:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ERROR — REQUIRED RESULT FILES MISSING"
        )

        print(
            "=" * 70
        )

        for file_path in missing_files:

            print(
                f"\nMissing:\n{file_path}"
            )

        raise FileNotFoundError(
            "Some required evaluation files are missing."
        )


# =========================================================
# Load Stage 1 + Stage 2 history
# =========================================================

def load_combined_history():
    """
    Load feature-extraction and fine-tuning
    training histories.

    Global epoch numbers are created so both
    training stages can be shown on one graph.
    """

    stage_one = pd.read_csv(
        FEATURE_HISTORY_FILE
    )

    stage_two = pd.read_csv(
        FINE_TUNE_HISTORY_FILE
    )

    # -----------------------------------------------------
    # Stage 1 global epochs
    # -----------------------------------------------------

    stage_one[
        "global_epoch"
    ] = range(
        1,
        len(stage_one) + 1
    )

    # -----------------------------------------------------
    # Stage 2 continues after Stage 1
    # -----------------------------------------------------

    stage_two[
        "global_epoch"
    ] = range(
        len(stage_one) + 1,
        len(stage_one)
        + len(stage_two)
        + 1
    )

    return (
        stage_one,
        stage_two
    )


# =========================================================
# 1. Per-class Precision / Recall / F1
# =========================================================

def plot_per_class_metrics():
    """
    Generate grouped bars showing Precision,
    Recall and F1-score for every ripeness class.
    """

    dataframe = pd.read_csv(
        CLASSIFICATION_REPORT_FILE
    )

    classes = dataframe[
        "class"
    ].tolist()

    precision = (
        dataframe[
            "precision"
        ].to_numpy()
        * 100
    )

    recall = (
        dataframe[
            "recall"
        ].to_numpy()
        * 100
    )

    f1_scores = (
        dataframe[
            "f1_score"
        ].to_numpy()
        * 100
    )

    positions = list(
        range(
            len(classes)
        )
    )

    width = 0.25

    figure = plt.figure(
        figsize=(11, 7)
    )

    precision_bars = plt.bar(
        [
            position - width
            for position in positions
        ],
        precision,
        width=width,
        label="Precision"
    )

    recall_bars = plt.bar(
        positions,
        recall,
        width=width,
        label="Recall"
    )

    f1_bars = plt.bar(
        [
            position + width
            for position in positions
        ],
        f1_scores,
        width=width,
        label="F1-score"
    )

    # -----------------------------------------------------
    # Value labels
    # -----------------------------------------------------

    for bars in [
        precision_bars,
        recall_bars,
        f1_bars
    ]:

        for bar in bars:

            value = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.06,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(
        positions,
        classes
    )

    # Zoom into useful classification-performance range.
    # This makes small differences between high-performing
    # models easier to observe.
    plt.ylim(
        90,
        100
    )

    plt.xlabel(
        "Ripeness Class"
    )

    plt.ylabel(
        "Score (%)"
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
# 2. Overall Model Performance
# =========================================================

def plot_overall_metrics():
    """
    Generate overall model performance graph.

    Includes:
        Accuracy
        Macro Precision
        Macro Recall
        Macro F1
        Weighted Precision
        Weighted Recall
        Weighted F1
    """

    dataframe = pd.read_csv(
        OVERALL_METRICS_FILE
    )

    metric_dictionary = dict(
        zip(
            dataframe[
                "metric"
            ],
            dataframe[
                "value"
            ]
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
        figsize=(12, 7)
    )

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.ylim(
        90,
        100
    )

    plt.xlabel(
        "Metric"
    )

    plt.ylabel(
        "Score (%)"
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

    # -----------------------------------------------------
    # Add percentages above bars
    # -----------------------------------------------------

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
# 3. Training vs Validation Accuracy
# =========================================================

def plot_training_accuracy():
    """
    Plot training and validation accuracy across
    both transfer-learning stages.

    The vertical dashed line shows where
    MobileNetV2 fine-tuning begins.
    """

    (
        stage_one,
        stage_two
    ) = load_combined_history()

    figure = plt.figure(
        figsize=(12, 7)
    )

    # -----------------------------------------------------
    # Stage 1
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Stage 2
    #
    # Same labels are not repeated because they would
    # duplicate entries in the legend.
    # -----------------------------------------------------

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
# 4. Training vs Validation Loss
# =========================================================

def plot_training_loss():
    """
    Generate training-loss and validation-loss curves.

    Useful for identifying overfitting:
    training loss may continue decreasing while
    validation loss stabilizes or increases.
    """

    (
        stage_one,
        stage_two
    ) = load_combined_history()

    figure = plt.figure(
        figsize=(12, 7)
    )

    # -----------------------------------------------------
    # Stage 1
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Stage 2
    # -----------------------------------------------------

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
# 5. Generalization Gap
# =========================================================

def plot_generalization_gap():
    """
    Plot:

        training accuracy
        -
        validation accuracy

    A growing positive gap can indicate increasing
    overfitting.

    Negative values mean validation accuracy was
    temporarily higher than training accuracy.
    """

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

    generalization_gap = (
        combined[
            "accuracy"
        ]
        -
        combined[
            "val_accuracy"
        ]
    ) * 100

    figure = plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        combined[
            "global_epoch"
        ],
        generalization_gap,
        marker="o"
    )

    # -----------------------------------------------------
    # Zero-gap reference
    # -----------------------------------------------------

    plt.axhline(
        y=0,
        linestyle="--",
        label="No Accuracy Gap"
    )

    # -----------------------------------------------------
    # Fine-tuning transition
    # -----------------------------------------------------

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
        "Train - Validation Accuracy (%)"
    )

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Generalization Gap"
    )

    plt.legend()

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
        "PERFORMANCE VISUALIZATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nExperiment:"
        f" {EXPERIMENT_NAME.upper()}"
    )

    print(
        f"\nResults folder:"
        f"\n{RESULTS_FOLDER}"
    )

    print(
        "\nChecking required files..."
    )

    validate_files()

    print(
        "Required files found."
    )

    print(
        "\nGenerating graphs..."
    )

    # -----------------------------------------------------
    # Generate complete visualization package
    # -----------------------------------------------------

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
        f"\nAll graphs saved to:"
        f"\n{RESULTS_FOLDER}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()
