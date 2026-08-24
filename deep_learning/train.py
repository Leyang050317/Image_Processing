from pathlib import Path
import csv
import tensorflow as tf


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEEP_LEARNING_FOLDER = Path(__file__).resolve().parent

MANIFEST_FILE = DEEP_LEARNING_FOLDER / "split_manifest.csv"

MODELS_FOLDER = DEEP_LEARNING_FOLDER / "models"
RESULTS_FOLDER = DEEP_LEARNING_FOLDER / "results"

MODELS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Experiment configuration
# =========================================================

CLASS_NAMES = [
    "overripe",
    "ripe",
    "rotten",
    "unripe"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

RANDOM_SEED = 42


# =========================================================
# Training configuration
# =========================================================

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20

INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.30

FINE_TUNE_LAST_N_LAYERS = 30


# =========================================================
# Experiment selection
#
# Change only these two values for each experiment.
# =========================================================

EXPERIMENT_NAME = "baseline"

SOURCE_FOLDER = (
    PROJECT_ROOT
    / "dataset"
)


# Example baseline:
#
# EXPERIMENT_NAME = "baseline"
# SOURCE_FOLDER = PROJECT_ROOT / "dataset"
#
# Later:
#
# EXPERIMENT_NAME = "jw"
# SOURCE_FOLDER = PROJECT_ROOT / "jinwen" / "output"


# =========================================================
# Reproducibility
# =========================================================

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)


# =========================================================
# Load split manifest
# =========================================================

def load_manifest():
    """
    Load the fixed Train / Validation / Test split.
    """

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Split manifest not found:\n"
            f"{MANIFEST_FILE}"
        )

    records = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

    return records


# =========================================================
# Build file lists
# =========================================================

def get_split_data(
    records,
    split_name
):
    """
    Return image paths and integer labels
    for one split.
    """

    image_paths = []
    labels = []

    missing_files = []

    for record in records:

        if record["split"] != split_name:
            continue

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

        class_name = record["class"]

        image_paths.append(
            str(image_path)
        )

        labels.append(
            CLASS_TO_INDEX[
                class_name
            ]
        )


    if missing_files:

        print(
            f"\n[WARNING] "
            f"{len(missing_files)} missing "
            f"{split_name} image(s)."
        )

        print(
            "First missing files:"
        )

        for missing in missing_files[:10]:

            print(
                f"  {missing}"
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
    Load image and prepare it for MobileNetV2.

    Note:
    MobileNetV2 preprocess_input() is NOT applied here.
    It is included inside the model itself.
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
# Build tf.data dataset
# =========================================================

def build_dataset(
    image_paths,
    labels,
    training=False
):
    """
    Build TensorFlow dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )


    if training:

        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )


    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    dataset = dataset.batch(
        BATCH_SIZE
    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


# =========================================================
# Build MobileNetV2
# =========================================================

def build_model():
    """
    MobileNetV2 transfer-learning architecture.

    Input
        ↓
    MobileNetV2 preprocess_input
        ↓
    MobileNetV2 ImageNet backbone
        ↓
    Global Average Pooling
        ↓
    Dropout
        ↓
    Dense 4-class Softmax
    """

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="input_image"
    )


    # -----------------------------------------------------
    # MobileNetV2-specific normalization
    #
    # [0, 255] -> approximately [-1, 1]
    # -----------------------------------------------------

    x = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            inputs
        )
    )


    # -----------------------------------------------------
    # ImageNet pretrained backbone
    # -----------------------------------------------------

    base_model = (
        tf.keras.applications.MobileNetV2(
            input_shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            ),
            include_top=False,
            weights="imagenet"
        )
    )


    # Stage 1:
    # freeze MobileNetV2
    base_model.trainable = False


    x = base_model(
        x,
        training=False
    )


    # -----------------------------------------------------
    # Classification head
    # -----------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)


    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout"
    )(x)


    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="ripeness_classifier"
    )(x)


    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="banana_ripeness_mobilenetv2"
    )


    return (
        model,
        base_model
    )


# =========================================================
# Compile Stage 1
# =========================================================

def compile_stage_one(
    model
):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=
            INITIAL_LEARNING_RATE
        ),

        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),

        metrics=[
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )


# =========================================================
# Fine-tuning configuration
# =========================================================

def enable_fine_tuning(
    model,
    base_model
):
    """
    Unfreeze only the final portion of MobileNetV2.
    """

    base_model.trainable = True


    freeze_until = (
        len(base_model.layers)
        - FINE_TUNE_LAST_N_LAYERS
    )


    for layer in base_model.layers[
        :freeze_until
    ]:

        layer.trainable = False


    for layer in base_model.layers[
        freeze_until:
    ]:

        # Keep BatchNormalization frozen
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True


    # IMPORTANT:
    # Recompile after changing trainable layers.

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=
            FINE_TUNE_LEARNING_RATE
        ),

        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),

        metrics=[
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )


# =========================================================
# Training callbacks
# =========================================================

def create_callbacks(
    stage_name
):

    checkpoint_path = (
        MODELS_FOLDER
        / f"{EXPERIMENT_NAME}_{stage_name}_best.keras"
    )


    callbacks = [

        # -------------------------------------------------
        # Save best validation model
        # -------------------------------------------------

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                checkpoint_path
            ),

            monitor="val_accuracy",

            save_best_only=True,

            mode="max",

            verbose=1
        ),


        # -------------------------------------------------
        # Stop if validation stops improving
        # -------------------------------------------------

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1
        ),


        # -------------------------------------------------
        # Reduce learning rate when improvement stalls
        # -------------------------------------------------

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.2,

            patience=3,

            min_lr=1e-7,

            verbose=1
        )
    ]


    return callbacks


# =========================================================
# Save training history
# =========================================================

def save_history(
    history,
    stage_name
):
    """
    Save training history to CSV.
    """

    history_path = (
        RESULTS_FOLDER
        / f"{EXPERIMENT_NAME}_{stage_name}_history.csv"
    )


    metric_names = list(
        history.history.keys()
    )


    epoch_count = len(
        history.history[
            metric_names[0]
        ]
    )


    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "epoch",
                *metric_names
            ]
        )


        for epoch_index in range(
            epoch_count
        ):

            writer.writerow(

                [
                    epoch_index + 1,

                    *[
                        history.history[
                            metric
                        ][epoch_index]

                        for metric
                        in metric_names
                    ]
                ]
            )


    print(
        f"\nTraining history saved:"
    )

    print(
        history_path
    )


# =========================================================
# Dataset information
# =========================================================

def print_dataset_information(
    train_paths,
    val_paths,
    test_paths
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Source folder:"
        f"\n{SOURCE_FOLDER}"
    )

    print(
        "\nDataset"
    )

    print(
        "-" * 70
    )

    print(
        f"Training images: "
        f"{len(train_paths):,}"
    )

    print(
        f"Validation images: "
        f"{len(val_paths):,}"
    )

    print(
        f"Test images: "
        f"{len(test_paths):,}"
    )

    total_available = (
            len(train_paths)
            + len(val_paths)
            + len(test_paths)
    )

    print(
        f"Total available: {total_available:,}"
    )

    print(
        "=" * 70
    )


# =========================================================
# Main training
# =========================================================

def main():

    # -----------------------------------------------------
    # Load fixed manifest
    # -----------------------------------------------------

    records = load_manifest()


    # -----------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------

    (
        train_paths,
        train_labels
    ) = get_split_data(
        records,
        "train"
    )


    (
        val_paths,
        val_labels
    ) = get_split_data(
        records,
        "validation"
    )


    (
        test_paths,
        test_labels
    ) = get_split_data(
        records,
        "test"
    )


    print_dataset_information(
        train_paths,
        val_paths,
        test_paths
    )


    # -----------------------------------------------------
    # Important safety check
    # -----------------------------------------------------

    expected_total = len(
        records
    )

    available_total = (
        len(train_paths)
        + len(val_paths)
        + len(test_paths)
    )


    if available_total != expected_total:

        raise RuntimeError(
            "\nTraining aborted.\n"
            f"Manifest expects "
            f"{expected_total:,} images, "
            f"but experiment source contains "
            f"{available_total:,}.\n"
            "Check preprocessing output before training."
        )


    # -----------------------------------------------------
    # Build datasets
    # -----------------------------------------------------

    train_dataset = build_dataset(
        train_paths,
        train_labels,
        training=True
    )


    validation_dataset = build_dataset(
        val_paths,
        val_labels,
        training=False
    )


    test_dataset = build_dataset(
        test_paths,
        test_labels,
        training=False
    )


    # -----------------------------------------------------
    # Build MobileNetV2
    # -----------------------------------------------------

    model, base_model = build_model()


    print(
        "\nMobileNetV2 created."
    )

    print(
        f"Total parameters: "
        f"{model.count_params():,}"
    )


    # =====================================================
    # STAGE 1
    # Feature extraction
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 1 — FEATURE EXTRACTION"
    )

    print(
        "=" * 70
    )

    print(
        "MobileNetV2 backbone: FROZEN"
    )

    print(
        "Classification head: TRAINABLE"
    )


    compile_stage_one(
        model
    )


    stage_one_history = model.fit(

        train_dataset,

        validation_data=
        validation_dataset,

        epochs=
        INITIAL_EPOCHS,

        callbacks=
        create_callbacks(
            "feature_extraction"
        )
    )


    save_history(
        stage_one_history,
        "feature_extraction"
    )


    # =====================================================
    # STAGE 2
    # Fine-tuning
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 2 — FINE-TUNING"
    )

    print(
        "=" * 70
    )

    print(
        f"Unfreezing final "
        f"{FINE_TUNE_LAST_N_LAYERS} "
        f"MobileNetV2 layers."
    )


    enable_fine_tuning(
        model,
        base_model
    )


    stage_two_history = model.fit(

        train_dataset,

        validation_data=
        validation_dataset,

        epochs=
        FINE_TUNE_EPOCHS,

        callbacks=
        create_callbacks(
            "fine_tuning"
        )
    )


    save_history(
        stage_two_history,
        "fine_tuning"
    )


    # =====================================================
    # Save final model
    # =====================================================

    final_model_path = (
        MODELS_FOLDER
        / f"{EXPERIMENT_NAME}_final.keras"
    )


    model.save(
        final_model_path
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFinal model saved:"
    )

    print(
        final_model_path
    )


    # -----------------------------------------------------
    # Simple preliminary test loss/accuracy
    #
    # Full evaluation will be evaluate.py
    # -----------------------------------------------------

    print(
        "\nRunning preliminary test evaluation..."
    )


    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )


    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()