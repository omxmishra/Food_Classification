import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAYERS,
    FINE_TUNE_LR,
    IMAGE_SIZE,
    INITIAL_LR,
    MODELS_DIR,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def build_image_dataframe(data_dir=DATA_DIR):
    class_names = sorted([folder.name for folder in data_dir.iterdir() if folder.is_dir()])

    image_paths = []
    for class_name in class_names:
        class_folder = data_dir / class_name
        for image in class_folder.glob("*"):
            image_paths.append({"filepath": str(image), "label": class_name})

    df = pd.DataFrame(image_paths)
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    df["label_idx"] = df["label"].map(label_to_index)

    return df, class_names


def stratified_split(df, test_size=0.30, val_ratio=0.50, seed=SEED):
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label_idx"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, stratify=temp_df["label_idx"], random_state=seed
    )
    return train_df, val_df, test_df


def load_image(path, label, image_size=IMAGE_SIZE):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    return image, label


def build_dataset(dataframe, batch_size=BATCH_SIZE, shuffle=False, image_size=IMAGE_SIZE, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe["filepath"].values, dataframe["label_idx"].values)
    )
    ds = ds.map(lambda p, l: load_image(p, l, image_size), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2000, seed=seed)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_augmentation_layer():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.10),
            layers.RandomTranslation(0.10, 0.10),
        ],
        name="data_augmentation",
    )


def apply_augmentation(dataset, augmentation_layer):
    dataset = dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.prefetch(AUTOTUNE)


def build_mobilenet_model(num_classes, image_size=IMAGE_SIZE):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model, base_model


def build_callbacks(checkpoint_path, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]


def unfreeze_for_fine_tuning(model, base_model, num_layers=FINE_TUNE_LAYERS, learning_rate=FINE_TUNE_LR):
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def evaluate_model(model, test_ds, class_names, test_df):
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_ds, verbose=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "Accuracy": float(test_accuracy),
        "Top5_Accuracy": float(test_top5_accuracy),
        "Precision_Weighted": float(precision),
        "Recall_Weighted": float(recall),
        "F1_Weighted": float(f1),
    }

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().iloc[:-3]

    cm = confusion_matrix(y_true, y_pred)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    confused_pairs = []
    for i in range(cm_no_diag.shape[0]):
        for j in range(cm_no_diag.shape[1]):
            if cm_no_diag[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm_no_diag[i, j]))

    confused_df = pd.DataFrame(
        confused_pairs, columns=["True_Label", "Predicted_Label", "Count"]
    ).sort_values("Count", ascending=False).head(10)

    test_reset = test_df.reset_index(drop=True)
    mask = y_true != y_pred
    misclassified_df = test_reset[mask].copy()
    misclassified_df["True_Label"] = misclassified_df["label"]
    misclassified_df["Predicted_Label"] = [class_names[p] for p in y_pred[mask]]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "report_df": report_df,
        "confusion_matrix": cm,
        "confused_df": confused_df,
        "misclassified_df": misclassified_df.reset_index(drop=True),
    }


def run_training():
    set_seed(SEED)
    enable_gpu_memory_growth()

    df, class_names = build_image_dataframe(DATA_DIR)
    train_df, val_df, test_df = stratified_split(df)

    train_ds = build_dataset(train_df, shuffle=True)
    val_ds = build_dataset(val_df)
    test_ds = build_dataset(test_df)

    augmentation = build_augmentation_layer()
    train_ds = apply_augmentation(train_ds, augmentation)

    model, base_model = build_mobilenet_model(len(class_names))

    callbacks = build_callbacks(f"{MODELS_DIR}/mobilenet_best.keras")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    model = unfreeze_for_fine_tuning(model, base_model)
    fine_tune_callbacks = build_callbacks(f"{MODELS_DIR}/mobilenet_finetuned.keras")
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=fine_tune_callbacks)

    results = evaluate_model(model, test_ds, class_names, test_df)

    model.save(f"{MODELS_DIR}/food101_mobilenetv2_finetuned.keras")
    with open(f"{MODELS_DIR}/class_names.json", "w") as f:
        json.dump(class_names, f)
    pd.Series(results["metrics"]).to_csv(f"{MODELS_DIR}/food101_test_metrics.csv")

    return model, base_model, class_names, test_df, results


if __name__ == "__main__":
    run_training()