from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEEP_LEARNING_FOLDER = Path(__file__).resolve().parent

MANIFEST_FILE = (
    DEEP_LEARNING_FOLDER
    / "split_manifest.csv"
)

MODELS_FOLDER = (
    DEEP_LEARNING_FOLDER
    / "models"
)

RESULTS_FOLDER = (
    DEEP_LEARNING_FOLDER
    / "results"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Experiment configuration
# =========================================================

EXPERIMENT_NAME = "baseline"

SOURCE_FOLDER = (
    PROJECT_ROOT
    / "dataset"
)

MODEL_PATH = (
    MODELS_FOLDER
    / "baseline_final.keras"
)


CLASS_NAMES = [
    "overripe",
    "ripe",
    "rotten",
    "unripe"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
    in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 32


# =========================================================
# Load manifest
# =========================================================

def load_test_records():
    """
    Load only the fixed TEST records
    from split_manifest.csv.
    """

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_FILE}"
        )

    records = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            if row["split"] == "test":

                records.append(
                    row
                )

    return records


# =========================================================
# Resolve test paths
# =========================================================

def get_test_data(
    records
):
    """
    Convert manifest records into
    actual HY processed image paths
    and integer labels.
    """

    image_paths = []
    labels = []

    missing_files = []

    for record in records:

        relative_path = Path(
            record["relative_path"]
        )

        image_path = (
            SOURCE_FOLDER
            / relative_path
        )

        if not image_path.exists():

            missing_files.append(
                image_path
            )

            continue

        image_paths.append(
            str(image_path)
        )

        labels.append(
            CLASS_TO_INDEX[
                record["class"]
            ]
        )


    if missing_files:

        print(
            f"\n[WARNING] "
            f"{len(missing_files)} "
            f"test image(s) missing."
        )

        for path in missing_files[:10]:

            print(
                f"  {path}"
            )


    return (
        image_paths,
        labels
    )


# =========================================================
# Image loader
# =========================================================

def load_image(
    image_path,
    label
):
    """
    Load image for MobileNetV2.

    MobileNetV2 normalization is already
    built into the saved model.
    """

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return (
        image,
        label
    )


# =========================================================
# Build test dataset
# =========================================================

def build_test_dataset(
    image_paths,
    labels
):
    """
    Build the TensorFlow test dataset.

    No shuffling is applied because
    prediction order must remain aligned
    with the true labels.
    """

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                image_paths,
                labels
            )
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=
        tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# =========================================================
# Save overall metrics
# =========================================================

def save_overall_metrics(
    accuracy,
    precision_macro,
    recall_macro,
    f1_macro,
    precision_weighted,
    recall_weighted,
    f1_weighted
):
    """
    Save overall evaluation metrics.
    """

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_overall_metrics.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "metric",
                "value"
            ]
        )

        writer.writerow(
            [
                "accuracy",
                accuracy
            ]
        )

        writer.writerow(
            [
                "precision_macro",
                precision_macro
            ]
        )

        writer.writerow(
            [
                "recall_macro",
                recall_macro
            ]
        )

        writer.writerow(
            [
                "f1_macro",
                f1_macro
            ]
        )

        writer.writerow(
            [
                "precision_weighted",
                precision_weighted
            ]
        )

        writer.writerow(
            [
                "recall_weighted",
                recall_weighted
            ]
        )

        writer.writerow(
            [
                "f1_weighted",
                f1_weighted
            ]
        )

    print(
        f"\nOverall metrics saved:"
    )

    print(
        output_file
    )


# =========================================================
# Save classification report
# =========================================================

def save_classification_report(
    y_true,
    y_pred
):
    """
    Save class-level Precision,
    Recall, F1-score, and support.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_classification_report.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "class",
                "precision",
                "recall",
                "f1_score",
                "support"
            ]
        )

        for class_name in CLASS_NAMES:

            class_result = report[
                class_name
            ]

            writer.writerow(
                [
                    class_name,
                    class_result[
                        "precision"
                    ],
                    class_result[
                        "recall"
                    ],
                    class_result[
                        "f1-score"
                    ],
                    int(
                        class_result[
                            "support"
                        ]
                    )
                ]
            )

    return report


# =========================================================
# Save predictions
# =========================================================

def save_predictions(
    image_paths,
    y_true,
    y_pred,
    probabilities
):
    """
    Save every test prediction for later analysis.
    """

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_test_predictions.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "image",
                "true_class",
                "predicted_class",
                "confidence",
                "correct"
            ]
        )

        for (
            image_path,
            true_label,
            predicted_label,
            probability_vector
        ) in zip(
            image_paths,
            y_true,
            y_pred,
            probabilities
        ):

            confidence = float(
                np.max(
                    probability_vector
                )
            )

            writer.writerow(
                [
                    image_path,
                    CLASS_NAMES[
                        true_label
                    ],
                    CLASS_NAMES[
                        predicted_label
                    ],
                    confidence,
                    true_label
                    == predicted_label
                ]
            )

    print(
        f"\nPredictions saved:"
    )

    print(
        output_file
    )


# =========================================================
# Plot confusion matrix
# =========================================================

def save_confusion_matrix(
    y_true,
    y_pred
):
    """
    Generate and save confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    figure = plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.colorbar()

    plt.title(
        f"{EXPERIMENT_NAME.upper()} "
        f"Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        range(
            len(CLASS_NAMES)
        ),
        CLASS_NAMES,
        rotation=45
    )

    plt.yticks(
        range(
            len(CLASS_NAMES)
        ),
        CLASS_NAMES
    )


    # -----------------------------------------------------
    # Write values inside cells
    # -----------------------------------------------------

    for row in range(
        cm.shape[0]
    ):

        for column in range(
            cm.shape[1]
        ):

            plt.text(
                column,
                row,
                str(
                    cm[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )


    plt.tight_layout()

    output_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_confusion_matrix.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


    # -----------------------------------------------------
    # Save raw matrix as CSV too
    # -----------------------------------------------------

    csv_file = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_confusion_matrix.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "true/predicted",
                *CLASS_NAMES
            ]
        )

        for index, row in enumerate(
            cm
        ):

            writer.writerow(
                [
                    CLASS_NAMES[
                        index
                    ],
                    *row.tolist()
                ]
            )


    print(
        f"\nConfusion matrix saved:"
    )

    print(
        output_file
    )

    print(
        csv_file
    )


# =========================================================
# Print evaluation summary
# =========================================================

def print_summary(
    accuracy,
    precision_macro,
    recall_macro,
    f1_macro,
    report
):
    """
    Print human-readable evaluation results.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HY MOBILENETV2 TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAccuracy:       "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision:"
        f" {precision_macro:.4f}"
    )

    print(
        f"Macro Recall:   "
        f"{recall_macro:.4f}"
    )

    print(
        f"Macro F1-score: "
        f"{f1_macro:.4f}"
    )

    print(
        "\nPer-class results"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print(
        "-" * 70
    )

    for class_name in CLASS_NAMES:

        values = report[
            class_name
        ]

        print(
            f"{class_name:<15}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1-score']:>12.4f}"
            f"{int(values['support']):>12}"
        )

    print(
        "=" * 70
    )


# =========================================================
# Main evaluation
# =========================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HY MOBILENETV2 EVALUATION"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Validate saved model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"HY model not found:\n"
            f"{MODEL_PATH}"
        )


    # -----------------------------------------------------
    # Load test split
    # -----------------------------------------------------

    records = load_test_records()

    (
        image_paths,
        labels
    ) = get_test_data(
        records
    )


    print(
        f"\nTest images: "
        f"{len(image_paths):,}"
    )


    if len(image_paths) != len(
        records
    ):

        raise RuntimeError(
            "Some test images are missing. "
            "Evaluation aborted."
        )


    # -----------------------------------------------------
    # Build dataset
    # -----------------------------------------------------

    test_dataset = build_test_dataset(
        image_paths,
        labels
    )


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print(
        f"\nLoading model:"
    )

    print(
        MODEL_PATH
    )


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    print(
        "\nRunning predictions..."
    )


    probabilities = model.predict(
        test_dataset,
        verbose=1
    )


    y_true = np.array(
        labels,
        dtype=np.int32
    )


    y_pred = np.argmax(
        probabilities,
        axis=1
    )


    # =====================================================
    # Overall metrics
    # =====================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # =====================================================
    # Classification report
    # =====================================================

    report = save_classification_report(
        y_true,
        y_pred
    )


    # =====================================================
    # Save outputs
    # =====================================================

    save_overall_metrics(
        accuracy,
        precision_macro,
        recall_macro,
        f1_macro,
        precision_weighted,
        recall_weighted,
        f1_weighted
    )


    save_predictions(
        image_paths,
        y_true,
        y_pred,
        probabilities
    )


    save_confusion_matrix(
        y_true,
        y_pred
    )


    # =====================================================
    # Terminal summary
    # =====================================================

    print_summary(
        accuracy,
        precision_macro,
        recall_macro,
        f1_macro,
        report
    )


    print(
        f"\nAll evaluation results saved to:"
    )

    print(
        RESULTS_FOLDER
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()