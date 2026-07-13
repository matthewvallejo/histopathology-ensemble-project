"""
Ensemble Classifier GUI for Histopathology

A comprehensive graphical interface for training ensemble models and evaluating
individual models with full configuration options.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import io
import os
import json

import tensorflow as tf
import keras
from keras import layers
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import albumentations as A
from datetime import datetime


# Expanded model configurations (7 models)
SUPPORTED_MODELS = {
    'ResNet50V2': {
        'class': keras.applications.ResNet50V2,
        'preprocess': keras.applications.resnet_v2.preprocess_input,
        'input_size': (384, 384),
    },
    'ResNet101V2': {
        'class': keras.applications.ResNet101V2,
        'preprocess': keras.applications.resnet_v2.preprocess_input,
        'input_size': (384, 384),
    },
    'EfficientNetV2S': {
        'class': keras.applications.EfficientNetV2S,
        'preprocess': keras.applications.efficientnet_v2.preprocess_input,
        'input_size': (384, 384),
    },
    'EfficientNetV2M': {
        'class': keras.applications.EfficientNetV2M,
        'preprocess': keras.applications.efficientnet_v2.preprocess_input,
        'input_size': (384, 384),
    },
    'DenseNet201': {
        'class': keras.applications.DenseNet201,
        'preprocess': keras.applications.densenet.preprocess_input,
        'input_size': (384, 384),
    },
    'InceptionResNetV2': {
        'class': keras.applications.InceptionResNetV2,
        'preprocess': keras.applications.inception_resnet_v2.preprocess_input,
        'input_size': (299, 299),
    },
    'Xception': {
        'class': keras.applications.Xception,
        'preprocess': keras.applications.xception.preprocess_input,
        'input_size': (299, 299),
    },
}


def set_global_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=4)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = y_true_one_hot * tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    return loss_fn


class OutputRedirector:
    """Redirects stdout to a tkinter text widget."""

    _ANSI_ESCAPE = __import__('re').compile(r'\x1b\[[0-9;]*[mGKA-Z]')

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO()

    def write(self, text):
        text = self._ANSI_ESCAPE.sub('', text)
        if not text:
            return
        self.text_widget.configure(state='normal')
        # Handle \r: overwrite the current (last) line rather than appending
        if '\r' in text:
            for chunk in text.split('\r'):
                if not chunk:
                    continue
                # Delete the current last line then insert the replacement
                self.text_widget.delete("end-1l linestart", "end-1c")
                self.text_widget.insert(tk.END, chunk)
        else:
            self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        self.text_widget.update_idletasks()


class EnsembleTrainerGUI:
    """
    Trains multiple models with different seeds for ensemble prediction.
    Modified to accept GUI parameters.
    """

    def __init__(self, data_dir, model_name='ResNet50V2', batch_size=16,
                 output_dir='./ensemble_models'):
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.class_names = ["Normal", "Benign", "InSitu", "Invasive"]
        self.num_classes = len(self.class_names)

        self.model_config = SUPPORTED_MODELS[model_name]
        self.img_size = self.model_config['input_size']

        os.makedirs(output_dir, exist_ok=True)

        self.augment_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=6, p=1),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=0.5, p=1),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ])

        self.images = None
        self.labels = None

    def load_data(self, apply_stain_normalization=True):
        """Load and preprocess images."""
        images = []
        labels = []

        print(f"\nLoading data from {self.data_dir}...")

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found!")
                continue

            count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        img_array = np.array(img)

                        if apply_stain_normalization:
                            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                        img = Image.fromarray(img_array)
                        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img, dtype=np.float32)

                        images.append(img_array)
                        labels.append(class_idx)
                        count += 1

                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

            print(f"  {class_name}: {count} images")

        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels)

        print(f"\nTotal: {len(self.images)} images loaded")
        return self.images, self.labels

    def build_model(self, dropout_rate=0.4):
        """Build transfer learning model."""
        base_model = self.model_config['class'](
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(*self.img_size, 3))
        x = self.model_config['preprocess'](inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate * 0.7)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model, base_model

    def compute_class_weights(self, y, mode='balanced'):
        """Compute class weights."""
        if mode == 'none':
            return None
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        if mode == 'balanced':
            class_weights = class_weights / np.min(class_weights)
        elif mode == 'sqrt_balanced':
            class_weights = np.sqrt(class_weights)
            class_weights = class_weights / np.min(class_weights)
        return dict(zip(unique_classes, class_weights))

    def create_tf_dataset(self, X, y, training=False, seed=42):
        """Create TensorFlow dataset with augmentation for training."""

        def augment_fn(image, label):
            def apply_augmentation(img):
                img = img.numpy().astype(np.uint8)
                augmented = self.augment_pipeline(image=img)
                return augmented['image'].astype(np.float32)

            image = tf.py_function(apply_augmentation, [image], tf.float32)
            image.set_shape([*self.img_size, 3])
            return image, label

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            dataset = dataset.shuffle(len(X), seed=seed)
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train_single_model(self, seed, X_train, y_train, X_val, y_val,
                           epochs_frozen=50, epochs_finetune=10,
                           layers_to_unfreeze=30, loss_type='focal',
                           class_weight_mode='balanced'):
        """Train a single model with given seed."""
        print(f"\n{'='*60}")
        print(f"Training model with seed {seed}")
        print(f"{'='*60}")

        set_global_seed(seed)

        model, base_model = self.build_model()

        train_dataset = self.create_tf_dataset(X_train, y_train, training=True, seed=seed)
        val_dataset = self.create_tf_dataset(X_val, y_val, training=False)

        class_weights = self.compute_class_weights(y_train, mode=class_weight_mode)

        if loss_type == 'focal':
            loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss_fn = 'sparse_categorical_crossentropy'

        # Stage 1: Frozen backbone
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss_fn,
            metrics=['accuracy']
        )

        callbacks_stage1 = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5,
                patience=5, min_lr=1e-6, verbose=1
            )
        ]

        print("\nStage 1: Training with frozen backbone...")
        model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=epochs_frozen, callbacks=callbacks_stage1,
            class_weight=class_weights, verbose=1
        )

        # Stage 2: Fine-tune with specified layers unfrozen
        if epochs_finetune > 0 and layers_to_unfreeze > 0:
            base_model.trainable = True
            num_layers = len(base_model.layers)

            # Unfreeze the last N layers
            freeze_until = max(0, num_layers - layers_to_unfreeze)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False

            trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
            print(f"\nUnfreezing {trainable_count} layers (last {layers_to_unfreeze} requested)")

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss=loss_fn,
                metrics=['accuracy']
            )

            callbacks_stage2 = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=20,
                    restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy', factor=0.5,
                    patience=7, min_lr=1e-7, verbose=1
                )
            ]

            print("\nStage 2: Fine-tuning...")
            model.fit(
                train_dataset, validation_data=val_dataset,
                epochs=epochs_finetune, callbacks=callbacks_stage2,
                class_weight=class_weights, verbose=1
            )

        return model

    def train_ensemble(self, seeds, test_size=0.15, val_size=0.15,
                       epochs_frozen=50, epochs_finetune=10,
                       layers_to_unfreeze=30, loss_type='focal',
                       class_weight_mode='balanced', apply_stain_normalization=True):
        """Train multiple models with different seeds."""
        print("\n" + "=" * 70)
        print("ENSEMBLE TRAINING")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Seeds: {seeds}")
        print(f"Stain Normalization: {apply_stain_normalization}")
        print(f"Layers to Unfreeze: {layers_to_unfreeze}")
        print(f"Epochs (frozen): {epochs_frozen}")
        print(f"Epochs (fine-tune): {epochs_finetune}")
        print(f"Loss Type: {loss_type}")
        print(f"Class Weight Mode: {class_weight_mode}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

        # Load data
        self.load_data(apply_stain_normalization=apply_stain_normalization)

        # Create consistent data splits using first seed
        primary_seed = seeds[0]
        set_global_seed(primary_seed)

        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels,
            test_size=test_size, random_state=primary_seed, stratify=self.labels
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted, random_state=primary_seed, stratify=y_temp
        )

        print(f"\nData splits (consistent across all models):")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        self.X_test = X_test
        self.y_test = y_test

        # Train each model
        model_paths = []
        for seed in seeds:
            model = self.train_single_model(
                seed=seed,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                epochs_frozen=epochs_frozen,
                epochs_finetune=epochs_finetune,
                layers_to_unfreeze=layers_to_unfreeze,
                loss_type=loss_type,
                class_weight_mode=class_weight_mode
            )

            model_path = os.path.join(self.output_dir, f'{self.model_name}_seed{seed}.keras')
            model.save(model_path)
            model_paths.append(model_path)
            print(f"Saved: {model_path}")

            keras.backend.clear_session()

        # Save ensemble metadata
        metadata = {
            'model_name': self.model_name,
            'seeds': seeds,
            # Store filenames only so the metadata stays portable across machines
            'model_paths': [os.path.basename(p) for p in model_paths],
            'img_size': self.img_size,
            'class_names': self.class_names,
            'stain_normalization': apply_stain_normalization,
            'layers_to_unfreeze': layers_to_unfreeze,
            'epochs_frozen': epochs_frozen,
            'epochs_finetune': epochs_finetune,
            'loss_type': loss_type,
            'class_weight_mode': class_weight_mode,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.output_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nEnsemble metadata saved: {metadata_path}")
        return model_paths

    def evaluate_ensemble(self, model_paths):
        """Evaluate ensemble on test set."""
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION")
        print("=" * 70)

        all_probs = []

        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
            model = keras.models.load_model(model_path, compile=False)

            dataset = tf.data.Dataset.from_tensor_slices(self.X_test)
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            probs = model.predict(dataset, verbose=0)
            all_probs.append(probs)

            keras.backend.clear_session()

        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        print("\n" + "-" * 50)
        print("ENSEMBLE RESULTS (averaged predictions)")
        print("-" * 50)
        print(classification_report(
            self.y_test, ensemble_preds,
            target_names=self.class_names, digits=4
        ))

        cm = confusion_matrix(self.y_test, ensemble_preds)
        print("Confusion Matrix:")
        print(cm)

        print("\nPer-class performance:")
        for cls_idx, cls_name in enumerate(self.class_names):
            mask = self.y_test == cls_idx
            correct = np.sum(ensemble_preds[mask] == cls_idx)
            total = np.sum(mask)
            if total > 0:
                print(f"  {cls_name}: {correct}/{total} ({correct/total*100:.1f}%)")

        print("\n" + "-" * 50)
        print("COMPARISON: Single Model vs Ensemble")
        print("-" * 50)

        single_correct = np.sum(np.argmax(all_probs[0], axis=1) == self.y_test)
        ensemble_correct = np.sum(ensemble_preds == self.y_test)

        print(f"Single model accuracy:   {single_correct}/{len(self.y_test)} ({single_correct/len(self.y_test)*100:.1f}%)")
        print(f"Ensemble accuracy:       {ensemble_correct}/{len(self.y_test)} ({ensemble_correct/len(self.y_test)*100:.1f}%)")
        print(f"Improvement:             {ensemble_correct - single_correct} samples")

        return {
            'ensemble_predictions': ensemble_preds,
            'ensemble_probabilities': ensemble_probs,
            'individual_probabilities': all_probs,
            'confusion_matrix': cm
        }


class IndividualModelEvaluatorGUI:
    """Evaluator for individual models from an ensemble."""

    def __init__(self, ensemble_dir, data_dir):
        self.ensemble_dir = ensemble_dir
        self.data_dir = data_dir
        self.class_names = ["Normal", "Benign", "InSitu", "Invasive"]
        self.num_classes = len(self.class_names)

        metadata_path = os.path.join(ensemble_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.model_name = self.metadata['model_name']
        self.seeds = self.metadata['seeds']
        # Resolve model paths relative to ensemble_dir in case the directory was moved
        self.model_paths = [
            os.path.join(ensemble_dir, os.path.basename(p))
            for p in self.metadata['model_paths']
        ]
        self.img_size = tuple(self.metadata['img_size'])

        # Case-insensitive lookup for model config
        model_key = None
        for key in SUPPORTED_MODELS:
            if key.lower() == self.model_name.lower():
                model_key = key
                break
        if model_key is None:
            raise ValueError(f"Unknown model '{self.model_name}'. "
                             f"Supported: {list(SUPPORTED_MODELS.keys())}")
        self.model_config = SUPPORTED_MODELS[model_key]

        self.images = None
        self.labels = None
        self.X_test = None
        self.y_test = None
        self.results = {}

    def load_data(self, apply_stain_normalization=True):
        """Load and preprocess images."""
        images = []
        labels = []

        print(f"\nLoading data from {self.data_dir}...")

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found!")
                continue

            count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        img_array = np.array(img)

                        if apply_stain_normalization:
                            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                        img = Image.fromarray(img_array)
                        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img, dtype=np.float32)

                        images.append(img_array)
                        labels.append(class_idx)
                        count += 1

                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

            print(f"  {class_name}: {count} images")

        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels)

        print(f"\nTotal: {len(self.images)} images loaded")

    def create_test_split(self, test_size=0.15):
        """Create test split using primary seed."""
        primary_seed = self.seeds[0]
        _, X_test, _, y_test = train_test_split(
            self.images, self.labels,
            test_size=test_size, random_state=primary_seed, stratify=self.labels
        )
        self.X_test = X_test
        self.y_test = y_test
        print(f"\nTest set: {len(X_test)} images")

    def evaluate_all_models(self, batch_size=16):
        """Evaluate all individual models."""
        print("\n" + "=" * 70)
        print("INDIVIDUAL MODEL EVALUATION")
        print("=" * 70)

        for seed, model_path in zip(self.seeds, self.model_paths):
            print(f"\nEvaluating model with seed {seed}...")
            model = keras.models.load_model(model_path, compile=False)

            dataset = tf.data.Dataset.from_tensor_slices(self.X_test)
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            probs = model.predict(dataset, verbose=0)
            preds = np.argmax(probs, axis=1)

            accuracy = np.mean(preds == self.y_test)
            cm = confusion_matrix(self.y_test, preds)

            self.results[seed] = {
                'predictions': preds,
                'probabilities': probs,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }

            print(f"  Accuracy: {accuracy*100:.2f}%")

            keras.backend.clear_session()

    def print_comparison(self):
        """Print comparison of all models."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(f"\n{'Seed':<10} {'Accuracy':<12} {'Per-Class Accuracy'}")
        print("-" * 70)

        best_seed = None
        best_accuracy = 0

        for seed in self.seeds:
            result = self.results[seed]
            accuracy = result['accuracy']

            per_class = []
            for cls_idx in range(self.num_classes):
                mask = self.y_test == cls_idx
                if np.sum(mask) > 0:
                    cls_acc = np.mean(result['predictions'][mask] == cls_idx)
                    per_class.append(f"{self.class_names[cls_idx][:3]}:{cls_acc*100:.0f}%")

            per_class_str = " | ".join(per_class)
            print(f"{seed:<10} {accuracy*100:.2f}%      {per_class_str}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_seed = seed

        print("-" * 70)
        print(f"Best model: Seed {best_seed} with {best_accuracy*100:.2f}% accuracy")

        return best_seed

    def print_detailed_report(self, seed):
        """Print detailed report for a specific model."""
        if seed not in self.results:
            print(f"No results for seed {seed}")
            return

        result = self.results[seed]

        print(f"\n" + "=" * 70)
        print(f"DETAILED REPORT - SEED {seed}")
        print("=" * 70)

        print("\nClassification Report:")
        print(classification_report(
            self.y_test, result['predictions'],
            target_names=self.class_names, digits=4
        ))

        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])


class EnsembleGUI:
    """Main GUI application for ensemble training and evaluation."""

    def __init__(self, root):
        self.root = root
        self.root.title("Histopathology Ensemble Classifier")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)

        # Variables
        self.data_dir = tk.StringVar(value="./data/")
        self.output_dir = tk.StringVar(value="./ensemble_models")
        self.model_name = tk.StringVar(value="ResNet50V2")
        self.seeds_str = tk.StringVar(value="42, 123, 456")
        self.stain_normalization = tk.BooleanVar(value=True)
        self.layers_to_unfreeze = tk.IntVar(value=30)
        self.epochs_frozen = tk.IntVar(value=50)
        self.epochs_finetune = tk.IntVar(value=10)
        self.loss_type = tk.StringVar(value="focal")
        self.class_weight_mode = tk.StringVar(value="balanced")

        # Evaluation variables
        self.eval_ensemble_dir = tk.StringVar(value="./ensemble_models")
        self.eval_data_dir = tk.StringVar(value="./data/")

        # Prediction variables
        self.pred_model_dir = tk.StringVar(value="./ensemble_models")
        self.pred_use_ensemble = tk.BooleanVar(value=True)
        self.pred_selected_model = tk.StringVar(value="")
        self.pred_use_tta = tk.BooleanVar(value=False)
        self.pred_stain_norm = tk.BooleanVar(value=True)
        self.pred_image_files = []
        self.pred_models = []
        self.pred_model_names = []  # List of model names for dropdown
        self.pred_metadata = None
        self.parent_dir_keras_files = {}  # Maps filename to full path for parent directory .keras files

        self.is_running = False
        self.trainer = None

        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Training Tab
        train_frame = ttk.Frame(notebook, padding="10")
        notebook.add(train_frame, text="  Train Ensemble  ")

        # Evaluation Tab
        eval_frame = ttk.Frame(notebook, padding="10")
        notebook.add(eval_frame, text="  Evaluate Models  ")

        # Prediction Tab
        predict_frame = ttk.Frame(notebook, padding="10")
        notebook.add(predict_frame, text="  Predict  ")

        self._create_training_tab(train_frame)
        self._create_evaluation_tab(eval_frame)
        self._create_prediction_tab(predict_frame)

    def _create_training_tab(self, parent):
        """Create the training configuration tab."""
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(10, weight=1)

        row = 0

        # Title
        title_label = ttk.Label(parent, text="Ensemble Training Configuration",
                                font=('Helvetica', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Data Directory
        ttk.Label(parent, text="Data Directory:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.data_dir, width=60).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=self._browse_data_dir).grid(
            row=row, column=2, padx=5, pady=5)
        row += 1

        # Output Directory
        ttk.Label(parent, text="Output Directory:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.output_dir, width=60).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=self._browse_output_dir).grid(
            row=row, column=2, padx=5, pady=5)
        row += 1

        # Model Configuration Frame
        config_frame = ttk.LabelFrame(parent, text="Model Configuration", padding="10")
        config_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        row += 1

        # Model Selection
        ttk.Label(config_frame, text="Model:").grid(row=0, column=0, sticky="w", pady=5)
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_name,
                                   values=list(SUPPORTED_MODELS.keys()), state='readonly', width=20)
        model_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Loss Type
        ttk.Label(config_frame, text="Loss Type:").grid(row=0, column=2, sticky="w", padx=(20, 0), pady=5)
        loss_combo = ttk.Combobox(config_frame, textvariable=self.loss_type,
                                  values=['focal', 'categorical_crossentropy'], state='readonly', width=20)
        loss_combo.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # Seeds
        ttk.Label(config_frame, text="Seeds (comma-separated):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(config_frame, textvariable=self.seeds_str, width=30).grid(
            row=1, column=1, sticky="w", padx=5, pady=5)

        # Class Weight Mode
        ttk.Label(config_frame, text="Class Weight:").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=5)
        weight_combo = ttk.Combobox(config_frame, textvariable=self.class_weight_mode,
                                    values=['balanced', 'sqrt_balanced', 'none'], state='readonly', width=20)
        weight_combo.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Stain Normalization
        ttk.Checkbutton(config_frame, text="Apply Stain Normalization",
                        variable=self.stain_normalization).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=5)

        # Training Parameters Frame
        train_params_frame = ttk.LabelFrame(parent, text="Training Parameters", padding="10")
        train_params_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        train_params_frame.columnconfigure(1, weight=1)
        train_params_frame.columnconfigure(3, weight=1)
        row += 1

        # Layers to Unfreeze
        ttk.Label(train_params_frame, text="Layers to Unfreeze:").grid(row=0, column=0, sticky="w", pady=5)
        layers_spinbox = ttk.Spinbox(train_params_frame, from_=0, to=200,
                                     textvariable=self.layers_to_unfreeze, width=10)
        layers_spinbox.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Epochs Frozen
        ttk.Label(train_params_frame, text="Epochs (Frozen):").grid(row=0, column=2, sticky="w", padx=(20, 0), pady=5)
        epochs_frozen_spinbox = ttk.Spinbox(train_params_frame, from_=1, to=200,
                                            textvariable=self.epochs_frozen, width=10)
        epochs_frozen_spinbox.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # Epochs Fine-tune
        ttk.Label(train_params_frame, text="Epochs (Fine-tune):").grid(row=1, column=0, sticky="w", pady=5)
        epochs_finetune_spinbox = ttk.Spinbox(train_params_frame, from_=0, to=200,
                                              textvariable=self.epochs_finetune, width=10)
        epochs_finetune_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Info label
        info_text = "Tip: Set fine-tune epochs to 0 to skip fine-tuning. Layers to unfreeze " \
        "determines how many layers from the end of the backbone are trainable during fine-tuning."
        info_label = ttk.Label(train_params_frame, text=info_text, wraplength=600, foreground='gray')
        info_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))

        # Buttons Frame
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.train_button = ttk.Button(button_frame, text="Start Training",
                                       command=self._start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.clear_train_button = ttk.Button(button_frame, text="Clear Output",
                                             command=self._clear_train_output)
        self.clear_train_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.train_progress = ttk.Progressbar(parent, mode='indeterminate')
        self.train_progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        self.train_progress.grid_remove()
        row += 1

        # Output text area
        output_frame = ttk.LabelFrame(parent, text="Training Output", padding="5")
        output_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        row += 1

        self.train_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                           height=15, state='disabled',
                                                           font=('Consolas', 10))
        self.train_output_text.grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.train_status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.train_status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0))

    def _create_evaluation_tab(self, parent):
        """Create the evaluation tab."""
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(6, weight=1)

        row = 0

        # Title
        title_label = ttk.Label(parent, text="Individual Model Evaluation",
                                font=('Helvetica', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Ensemble Directory
        ttk.Label(parent, text="Ensemble Directory:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.eval_ensemble_dir, width=60).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=self._browse_eval_ensemble_dir).grid(
            row=row, column=2, padx=5, pady=5)
        row += 1

        # Data Directory
        ttk.Label(parent, text="Data Directory:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.eval_data_dir, width=60).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=self._browse_eval_data_dir).grid(
            row=row, column=2, padx=5, pady=5)
        row += 1

        # Options frame
        options_frame = ttk.LabelFrame(parent, text="Evaluation Options", padding="10")
        options_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        self.eval_detailed_seed = tk.StringVar(value="")
        ttk.Label(options_frame, text="Detailed Report for Seed:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(options_frame, textvariable=self.eval_detailed_seed, width=15).grid(
            row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(options_frame, text="(leave empty for none)").grid(row=0, column=2, sticky="w", pady=5)

        # Buttons Frame
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.eval_button = ttk.Button(button_frame, text="Run Evaluation",
                                      command=self._start_evaluation)
        self.eval_button.pack(side=tk.LEFT, padx=5)

        self.clear_eval_button = ttk.Button(button_frame, text="Clear Output",
                                            command=self._clear_eval_output)
        self.clear_eval_button.pack(side=tk.LEFT, padx=5)

        self.save_eval_button = ttk.Button(button_frame, text="Save Evaluation",
                                           command=self._save_evaluation, state='disabled')
        self.save_eval_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.eval_progress = ttk.Progressbar(parent, mode='indeterminate')
        self.eval_progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        self.eval_progress.grid_remove()
        row += 1

        # Output text area
        output_frame = ttk.LabelFrame(parent, text="Evaluation Output", padding="5")
        output_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        row += 1

        self.eval_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                          height=15, state='disabled',
                                                          font=('Consolas', 10))
        self.eval_output_text.grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.eval_status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.eval_status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0))

    def _create_prediction_tab(self, parent):
        """Create the prediction/inference tab."""
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(8, weight=1)

        row = 0

        # Title
        title_label = ttk.Label(parent, text="Image Classification",
                                font=('Helvetica', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Model Directory
        ttk.Label(parent, text="Model Directory:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.pred_model_dir, width=60).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=self._browse_pred_model_dir).grid(
            row=row, column=2, padx=5, pady=5)
        row += 1

        # Options Frame
        options_frame = ttk.LabelFrame(parent, text="Prediction Options", padding="10")
        options_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        options_frame.columnconfigure(1, weight=1)
        row += 1

        # Use Ensemble checkbox
        ttk.Checkbutton(options_frame, text="Use Ensemble (average predictions from all models)",
                        variable=self.pred_use_ensemble,
                        command=self._toggle_model_selection).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=5)

        # Single model selection (shown when Use Ensemble is unchecked)
        self.model_select_label = ttk.Label(options_frame, text="Select Model:")
        self.model_select_label.grid(row=1, column=0, sticky="w", pady=5, padx=(20, 0))
        self.model_select_combo = ttk.Combobox(options_frame, textvariable=self.pred_selected_model,
                                                state='readonly', width=40)
        self.model_select_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Populate dropdown with .keras files from model directory
        self._populate_model_dropdown_from_directory()

        # Initially hide since Use Ensemble is checked by default
        self.model_select_label.grid_remove()
        self.model_select_combo.grid_remove()

        # Use TTA checkbox
        ttk.Checkbutton(options_frame, text="Use Test-Time Augmentation (slower but more robust)",
                        variable=self.pred_use_tta).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=5)

        # Stain normalization checkbox
        ttk.Checkbutton(options_frame, text="Apply Stain Normalization",
                        variable=self.pred_stain_norm).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=5)

        # Image Selection Frame
        image_frame = ttk.LabelFrame(parent, text="Image Selection", padding="10")
        image_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        image_frame.columnconfigure(0, weight=1)
        row += 1

        # Selected images listbox with scrollbar
        list_frame = ttk.Frame(image_frame)
        list_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        list_frame.columnconfigure(0, weight=1)

        self.pred_images_listbox = tk.Listbox(list_frame, height=5, selectmode=tk.EXTENDED)
        self.pred_images_listbox.grid(row=0, column=0, sticky="ew")

        listbox_scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                          command=self.pred_images_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, sticky="ns")
        self.pred_images_listbox.configure(yscrollcommand=listbox_scrollbar.set)

        # Image buttons
        img_button_frame = ttk.Frame(image_frame)
        img_button_frame.grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Button(img_button_frame, text="Add Image(s)...",
                   command=self._add_pred_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_button_frame, text="Add Folder...",
                   command=self._add_pred_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_button_frame, text="Remove Selected",
                   command=self._remove_pred_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(img_button_frame, text="Clear All",
                   command=self._clear_pred_images).pack(side=tk.LEFT, padx=5)

        self.pred_count_label = ttk.Label(image_frame, text="0 images selected")
        self.pred_count_label.grid(row=2, column=0, sticky="w", pady=5)

        # Action Buttons
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.pred_load_button = ttk.Button(action_frame, text="Load Model(s)",
                                           command=self._load_pred_models)
        self.pred_load_button.pack(side=tk.LEFT, padx=5)

        self.pred_run_button = ttk.Button(action_frame, text="Run Prediction",
                                          command=self._run_prediction, state='disabled')
        self.pred_run_button.pack(side=tk.LEFT, padx=5)

        self.pred_export_button = ttk.Button(action_frame, text="Export Results",
                                             command=self._export_pred_results, state='disabled')
        self.pred_export_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(action_frame, text="Clear Results",
                   command=self._clear_pred_results).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.pred_progress = ttk.Progressbar(parent, mode='indeterminate')
        self.pred_progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        self.pred_progress.grid_remove()
        row += 1

        # Results Frame - using a canvas for scrollable results
        results_outer_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="5")
        results_outer_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=5)
        results_outer_frame.columnconfigure(0, weight=1)
        results_outer_frame.rowconfigure(0, weight=1)
        row += 1

        # Create canvas with scrollbar for results
        self.results_canvas = tk.Canvas(results_outer_frame, highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(results_outer_frame, orient="vertical",
                                          command=self.results_canvas.yview)
        self.results_scrollable_frame = ttk.Frame(self.results_canvas)

        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )

        self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)

        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        results_scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind mouse wheel scrolling
        self.results_canvas.bind_all("<MouseWheel>",
            lambda e: self.results_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Status bar
        self.pred_status_var = tk.StringVar(value="Ready - Load model(s) to begin")
        status_bar = ttk.Label(parent, textvariable=self.pred_status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        # Store results for export
        self.prediction_results = []

    # Browse functions
    def _browse_data_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.data_dir.get() if os.path.isdir(self.data_dir.get()) else ".",
            title="Select Data Directory"
        )
        if directory:
            self.data_dir.set(directory)

    def _browse_output_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.output_dir.get() if os.path.isdir(self.output_dir.get()) else ".",
            title="Select Output Directory"
        )
        if directory:
            self.output_dir.set(directory)

    def _browse_eval_ensemble_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.eval_ensemble_dir.get() if os.path.isdir(self.eval_ensemble_dir.get()) else ".",
            title="Select Ensemble Models Directory"
        )
        if directory:
            self.eval_ensemble_dir.set(directory)

    def _browse_eval_data_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.eval_data_dir.get() if os.path.isdir(self.eval_data_dir.get()) else ".",
            title="Select Data Directory"
        )
        if directory:
            self.eval_data_dir.set(directory)

    def _browse_pred_model_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.pred_model_dir.get() if os.path.isdir(self.pred_model_dir.get()) else ".",
            title="Select Model Directory"
        )
        if directory:
            self.pred_model_dir.set(directory)
            # Reset loaded models when directory changes
            self.pred_models = []
            self.pred_model_names = []
            self.pred_metadata = None
            # Populate dropdown with .keras files from the selected directory
            self._populate_model_dropdown_from_directory(directory)
            self.pred_run_button.configure(state='disabled')
            self.pred_status_var.set("Model directory changed - reload models")

    def _toggle_model_selection(self):
        """Show/hide model selection dropdown based on Use Ensemble checkbox."""
        if self.pred_use_ensemble.get():
            # Hide model selection when using ensemble
            self.model_select_label.grid_remove()
            self.model_select_combo.grid_remove()
        else:
            # Show model selection when not using ensemble
            self.model_select_label.grid()
            self.model_select_combo.grid()

    def _add_pred_images(self):
        """Add individual image files for prediction."""
        filetypes = [
            ("Image files", "*.tiff *.tif *.jpg *.jpeg *.png *.bmp"),
            ("TIFF files", "*.tiff *.tif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Image(s)",
            filetypes=filetypes
        )
        if files:
            for f in files:
                if f not in self.pred_image_files:
                    self.pred_image_files.append(f)
                    self.pred_images_listbox.insert(tk.END, os.path.basename(f))
            self._update_pred_count()

    def _add_pred_folder(self):
        """Add all images from a folder."""
        directory = filedialog.askdirectory(title="Select Folder with Images")
        if directory:
            extensions = ('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp')
            count = 0
            for filename in os.listdir(directory):
                if filename.lower().endswith(extensions):
                    filepath = os.path.join(directory, filename)
                    if filepath not in self.pred_image_files:
                        self.pred_image_files.append(filepath)
                        self.pred_images_listbox.insert(tk.END, filename)
                        count += 1
            self._update_pred_count()
            if count > 0:
                messagebox.showinfo("Images Added", f"Added {count} images from folder.")

    def _remove_pred_images(self):
        """Remove selected images from the list."""
        selected = list(self.pred_images_listbox.curselection())
        selected.reverse()  # Remove from end to avoid index shifting
        for idx in selected:
            self.pred_images_listbox.delete(idx)
            del self.pred_image_files[idx]
        self._update_pred_count()

    def _clear_pred_images(self):
        """Clear all images from the list."""
        self.pred_images_listbox.delete(0, tk.END)
        self.pred_image_files = []
        self._update_pred_count()

    def _update_pred_count(self):
        """Update the image count label."""
        count = len(self.pred_image_files)
        self.pred_count_label.configure(text=f"{count} image(s) selected")

    def _load_pred_models(self):
        """Load models for prediction."""
        if self.is_running:
            return

        model_dir = self.pred_model_dir.get()
        if not model_dir or not os.path.isdir(model_dir):
            messagebox.showerror("Error", "Please select a valid model directory.")
            return

        # Check for .keras files in the directory
        keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if not keras_files:
            messagebox.showerror("Error",
                f"No .keras model files found in:\n{model_dir}\n\n"
                "Please select a directory containing trained models.")
            return

        self.is_running = True
        self.pred_load_button.configure(state='disabled')
        self.pred_progress.grid()
        self.pred_progress.start(10)
        self.pred_status_var.set("Loading models...")

        thread = threading.Thread(target=self._load_models_worker, daemon=True)
        thread.start()

    def _load_models_worker(self):
        """Worker function to load models."""
        try:
            model_dir = self.pred_model_dir.get()
            metadata_path = os.path.join(model_dir, 'ensemble_metadata.json')

            # Try to load metadata if it exists
            self.pred_metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.pred_metadata = json.load(f)

            # Clear any existing models
            self.pred_models = []
            self.pred_model_names = []
            keras.backend.clear_session()

            # Determine which model files to load
            if self.pred_use_ensemble.get():
                # Load all .keras files for ensemble
                model_files = []
                for filename in os.listdir(model_dir):
                    if filename.endswith('.keras'):
                        model_files.append(os.path.join(model_dir, filename))
                model_files.sort()
            else:
                # Load only the selected model
                selected = self.pred_selected_model.get()
                if selected:
                    selected_path = os.path.join(model_dir, selected + '.keras')
                    if os.path.exists(selected_path):
                        model_files = [selected_path]
                    else:
                        raise ValueError(f"Selected model file not found: {selected_path}")
                else:
                    raise ValueError("No model selected. Please select a model from the dropdown.")

            if not model_files:
                raise ValueError(f"No .keras model files found in {model_dir}")

            # Load each model
            for i, path in enumerate(model_files):
                self.root.after(0, lambda idx=i, total=len(model_files): self.pred_status_var.set(
                    f"Loading model {idx+1}/{total}..."))
                model = keras.models.load_model(path, compile=False)
                self.pred_models.append(model)

                # Use filename as the display name
                filename = os.path.basename(path)
                # Remove .keras extension for cleaner display
                display_name = filename.replace('.keras', '')
                self.pred_model_names.append(display_name)

            # Create default metadata if none exists (for prediction to work)
            if self.pred_metadata is None:
                # Try to infer image size from first model's input shape
                input_shape = self.pred_models[0].input_shape
                if input_shape and len(input_shape) >= 3:
                    img_size = (input_shape[1], input_shape[2])
                else:
                    img_size = (384, 384)  # Default fallback

                self.pred_metadata = {
                    'img_size': img_size,
                    'class_names': ["Normal", "Benign", "InSitu", "Invasive"],
                    'stain_normalization': True
                }

            # Update model selection dropdown
            self.root.after(0, lambda: self._update_model_dropdown())

            # Update stain normalization checkbox based on training config
            stain_norm = self.pred_metadata.get('stain_normalization', True)
            self.root.after(0, lambda: self.pred_stain_norm.set(stain_norm))

            self.root.after(0, lambda: self.pred_status_var.set(
                f"Loaded {len(self.pred_models)} model(s) - Ready for prediction"))
            self.root.after(0, lambda: self.pred_run_button.configure(state='normal'))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load models:\n{e}"))
            self.root.after(0, lambda: self.pred_status_var.set("Failed to load models"))

        finally:
            self.root.after(0, self._load_models_complete)

    def _populate_model_dropdown_from_directory(self, directory=None):
        """Populate the model selection dropdown with .keras files from the specified directory."""
        if directory is None:
            directory = self.pred_model_dir.get()

        display_names = []
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.keras'):
                    display_names.append(filename.replace('.keras', ''))

        display_names.sort()
        # Store full paths for later use
        self.parent_dir_keras_files = {
            name: os.path.join(directory, name + '.keras') for name in display_names
        }
        self.model_select_combo['values'] = display_names
        if display_names:
            self.pred_selected_model.set(display_names[0])
        else:
            self.pred_selected_model.set("")

    def _update_model_dropdown(self):
        """Update the model selection dropdown with loaded models."""
        current_selection = self.pred_selected_model.get()
        self.model_select_combo['values'] = self.pred_model_names
        if current_selection in self.pred_model_names:
            self.pred_selected_model.set(current_selection)
        elif self.pred_model_names:
            self.pred_selected_model.set(self.pred_model_names[0])

    def _load_models_complete(self):
        """Called when model loading is complete."""
        self.is_running = False
        self.pred_load_button.configure(state='normal')
        self.pred_progress.stop()
        self.pred_progress.grid_remove()

    def _run_prediction(self):
        """Run prediction on selected images."""
        if self.is_running:
            return

        if not self.pred_models:
            messagebox.showerror("Error", "Please load model(s) first.")
            return

        if not self.pred_image_files:
            messagebox.showerror("Error", "Please add image(s) for prediction.")
            return

        # Capture UI state on the main thread before spawning worker
        self._worker_use_ensemble = self.pred_use_ensemble.get()
        self._worker_use_tta = self.pred_use_tta.get()
        self._worker_stain_norm = self.pred_stain_norm.get()
        self._worker_selected_model_idx = 0
        if not self._worker_use_ensemble:
            idx = self.model_select_combo.current()
            if 0 <= idx < len(self.pred_models):
                self._worker_selected_model_idx = idx

        self.is_running = True
        self.pred_run_button.configure(state='disabled')
        self.pred_progress.grid()
        self.pred_progress.start(10)
        self.pred_status_var.set("Running predictions...")

        thread = threading.Thread(target=self._prediction_worker, daemon=True)
        thread.start()

    def _prediction_worker(self):
        """Worker function for running predictions."""
        try:
            img_size = tuple(self.pred_metadata['img_size'])
            class_names = self.pred_metadata.get('class_names', ["Normal", "Benign", "InSitu", "Invasive"])
            use_ensemble = self._worker_use_ensemble
            use_tta = self._worker_use_tta
            apply_stain_norm = self._worker_stain_norm

            # Use model index captured on main thread
            selected_model_idx = self._worker_selected_model_idx

            self.prediction_results = []
            total_images = len(self.pred_image_files)

            for idx, img_path in enumerate(self.pred_image_files):
                self.root.after(0, lambda i=idx: self.pred_status_var.set(
                    f"Processing image {i+1}/{total_images}..."))

                # Load and preprocess image
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    img_array = np.array(img)

                    if apply_stain_norm:
                        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                    img = Image.fromarray(img_array)
                    img = img.resize(img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)
                    img_batch = np.expand_dims(img_array, axis=0)

                    # Get predictions
                    if use_ensemble:
                        all_probs = []
                        for model in self.pred_models:
                            if use_tta:
                                probs = self._predict_with_tta(model, img_batch)
                            else:
                                probs = model.predict(img_batch, verbose=0)
                            all_probs.append(probs)
                        avg_probs = np.mean(all_probs, axis=0)[0]
                    else:
                        # Use selected model
                        selected_model = self.pred_models[selected_model_idx]
                        if use_tta:
                            avg_probs = self._predict_with_tta(selected_model, img_batch)[0]
                        else:
                            avg_probs = selected_model.predict(img_batch, verbose=0)[0]

                    pred_class = np.argmax(avg_probs)
                    confidence = avg_probs[pred_class]

                    result = {
                        'filename': os.path.basename(img_path),
                        'filepath': img_path,
                        'predicted_class': class_names[pred_class],
                        'predicted_index': int(pred_class),
                        'confidence': float(confidence),
                        'all_probabilities': {class_names[i]: float(avg_probs[i])
                                              for i in range(len(class_names))}
                    }
                    self.prediction_results.append(result)

                except Exception as e:
                    result = {
                        'filename': os.path.basename(img_path),
                        'filepath': img_path,
                        'error': str(e)
                    }
                    self.prediction_results.append(result)

            # Display results
            self.root.after(0, self._display_prediction_results)
            self.root.after(0, lambda: self.pred_status_var.set(
                f"Completed - {len(self.prediction_results)} images processed"))
            self.root.after(0, lambda: self.pred_export_button.configure(state='normal'))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed:\n{e}"))
            self.root.after(0, lambda: self.pred_status_var.set("Prediction failed"))

        finally:
            self.root.after(0, self._prediction_complete)

    def _predict_with_tta(self, model, img_batch, n_augments=8):
        """Apply test-time augmentation."""
        all_probs = []

        # Original
        probs = model.predict(img_batch, verbose=0)
        all_probs.append(probs)

        # Augmented versions
        for _ in range(n_augments - 1):
            aug_batch = img_batch.copy()

            # Random flip and rotation
            if np.random.random() > 0.5:
                aug_batch = np.flip(aug_batch, axis=2)
            if np.random.random() > 0.5:
                aug_batch = np.flip(aug_batch, axis=1)
            k = np.random.randint(4)
            aug_batch = np.rot90(aug_batch, k=k, axes=(1, 2))

            probs = model.predict(aug_batch, verbose=0)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def _prediction_complete(self):
        """Called when prediction is complete."""
        self.is_running = False
        self.pred_run_button.configure(state='normal')
        self.pred_progress.stop()
        self.pred_progress.grid_remove()

    def _display_prediction_results(self):
        """Display prediction results with confidence bars."""
        # Clear previous results
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()

        class_names = self.pred_metadata.get('class_names', ["Normal", "Benign", "InSitu", "Invasive"])
        colors = {'Normal': '#4CAF50', 'Benign': '#2196F3', 'InSitu': '#FF9800', 'Invasive': '#F44336'}

        for idx, result in enumerate(self.prediction_results):
            # Create frame for each result
            result_frame = ttk.Frame(self.results_scrollable_frame, padding="10")
            result_frame.pack(fill='x', pady=5, padx=5)

            # Add separator except for first item
            if idx > 0:
                ttk.Separator(self.results_scrollable_frame, orient='horizontal').pack(fill='x', pady=5)

            if 'error' in result:
                # Error case
                ttk.Label(result_frame, text=f"File: {result['filename']}",
                          font=('Helvetica', 10, 'bold')).pack(anchor='w')
                ttk.Label(result_frame, text=f"Error: {result['error']}",
                          foreground='red').pack(anchor='w')
                continue

            # Filename
            ttk.Label(result_frame, text=f"File: {result['filename']}",
                      font=('Helvetica', 10, 'bold')).pack(anchor='w')

            # Predicted class with confidence
            pred_class = result['predicted_class']
            confidence = result['confidence']
            color = colors.get(pred_class, '#666666')

            pred_label = ttk.Label(result_frame,
                                   text=f"Prediction: {pred_class} ({confidence*100:.1f}%)",
                                   font=('Helvetica', 11, 'bold'))
            pred_label.pack(anchor='w', pady=(5, 10))

            # Confidence bars for all classes
            probs = result['all_probabilities']
            for class_name in class_names:
                prob = probs.get(class_name, 0)
                bar_color = colors.get(class_name, '#666666')

                class_frame = ttk.Frame(result_frame)
                class_frame.pack(fill='x', pady=2)

                # Class name label (fixed width)
                ttk.Label(class_frame, text=f"{class_name}:", width=10).pack(side='left')

                # Progress bar style confidence indicator
                bar_frame = ttk.Frame(class_frame)
                bar_frame.pack(side='left', fill='x', expand=True, padx=(5, 10))

                # Background bar
                bg_canvas = tk.Canvas(bar_frame, height=20, bg='#E0E0E0', highlightthickness=0)
                bg_canvas.pack(fill='x')

                # Update to get width after rendering
                self.root.update_idletasks()
                canvas_width = bg_canvas.winfo_width()
                if canvas_width > 1:
                    bar_width = int(canvas_width * prob)
                    bg_canvas.create_rectangle(0, 0, bar_width, 20, fill=bar_color, outline='')

                # Percentage label
                ttk.Label(class_frame, text=f"{prob*100:.1f}%", width=8).pack(side='right')

        # Update canvas scroll region
        self.results_scrollable_frame.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _export_pred_results(self):
        """Export prediction results to CSV."""
        if not self.prediction_results:
            messagebox.showwarning("Warning", "No results to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Results"
        )
        if filepath:
            try:
                import csv
                with open(filepath, 'w', newline='') as f:
                    class_names = self.pred_metadata.get('class_names',
                                                         ["Normal", "Benign", "InSitu", "Invasive"])
                    fieldnames = ['filename', 'filepath', 'predicted_class', 'confidence'] + \
                                 [f'prob_{c}' for c in class_names]

                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for result in self.prediction_results:
                        if 'error' in result:
                            row = {'filename': result['filename'], 'filepath': result['filepath'],
                                   'predicted_class': 'ERROR', 'confidence': 0}
                        else:
                            row = {
                                'filename': result['filename'],
                                'filepath': result['filepath'],
                                'predicted_class': result['predicted_class'],
                                'confidence': result['confidence']
                            }
                            for class_name in class_names:
                                row[f'prob_{class_name}'] = result['all_probabilities'].get(class_name, 0)
                        writer.writerow(row)

                messagebox.showinfo("Success", f"Results exported to:\n{filepath}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{e}")

    def _clear_pred_results(self):
        """Clear prediction results display."""
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        self.prediction_results = []
        self.pred_export_button.configure(state='disabled')
        self.pred_status_var.set("Results cleared")

    def _clear_train_output(self):
        self.train_output_text.configure(state='normal')
        self.train_output_text.delete(1.0, tk.END)
        self.train_output_text.configure(state='disabled')

    def _clear_eval_output(self):
        self.eval_output_text.configure(state='normal')
        self.eval_output_text.delete(1.0, tk.END)
        self.eval_output_text.configure(state='disabled')
        self.save_eval_button.configure(state='disabled')

    def _save_evaluation(self):
        """Save evaluation output to a text file."""
        content = self.eval_output_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No evaluation output to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Evaluation Report"
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Evaluation saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save evaluation:\n{e}")

    def _validate_training_inputs(self):
        """Validate training inputs."""
        data_dir = self.data_dir.get()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Error", "Please select a valid data directory.")
            return False

        class_names = ["Normal", "Benign", "InSitu", "Invasive"]
        found_class = False
        for class_name in class_names:
            if os.path.isdir(os.path.join(data_dir, class_name)):
                found_class = True
                break

        if not found_class:
            messagebox.showerror("Error",
                f"Data directory must contain at least one class folder:\n"
                f"{', '.join(class_names)}")
            return False

        # Validate seeds
        try:
            seeds_str = self.seeds_str.get()
            seeds = [int(s.strip()) for s in seeds_str.split(',')]
            if len(seeds) == 0:
                raise ValueError("No seeds provided")
        except ValueError:
            messagebox.showerror("Error", "Seeds must be comma-separated integers (e.g., 42, 123, 456)")
            return False

        return True

    def _validate_evaluation_inputs(self):
        """Validate evaluation inputs."""
        ensemble_dir = self.eval_ensemble_dir.get()
        data_dir = self.eval_data_dir.get()

        if not ensemble_dir or not os.path.isdir(ensemble_dir):
            messagebox.showerror("Error", "Please select a valid ensemble directory.")
            return False

        metadata_path = os.path.join(ensemble_dir, 'ensemble_metadata.json')
        if not os.path.exists(metadata_path):
            messagebox.showerror("Error",
                f"No ensemble_metadata.json found in:\n{ensemble_dir}")
            return False

        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Error", "Please select a valid data directory.")
            return False

        return True

    def _start_training(self):
        """Start ensemble training."""
        if self.is_running:
            return

        if not self._validate_training_inputs():
            return

        self.is_running = True
        self.train_button.configure(state='disabled')
        self.train_progress.grid()
        self.train_progress.start(10)
        self.train_status_var.set("Training in progress...")

        thread = threading.Thread(target=self._training_worker, daemon=True)
        thread.start()

    def _training_worker(self):
        """Worker function for training."""
        old_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.train_output_text)

        try:
            seeds = [int(s.strip()) for s in self.seeds_str.get().split(',')]

            trainer = EnsembleTrainerGUI(
                data_dir=self.data_dir.get(),
                model_name=self.model_name.get(),
                batch_size=16,
                output_dir=self.output_dir.get()
            )

            model_paths = trainer.train_ensemble(
                seeds=seeds,
                epochs_frozen=self.epochs_frozen.get(),
                epochs_finetune=self.epochs_finetune.get(),
                layers_to_unfreeze=self.layers_to_unfreeze.get(),
                loss_type=self.loss_type.get(),
                class_weight_mode=self.class_weight_mode.get(),
                apply_stain_normalization=self.stain_normalization.get()
            )

            # Evaluate ensemble
            trainer.evaluate_ensemble(model_paths)

            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Models saved to: {self.output_dir.get()}")

            self.root.after(0, lambda: self.train_status_var.set("Training complete!"))

        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.train_status_var.set(f"Error: {str(e)[:50]}..."))
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))

        finally:
            sys.stdout = old_stdout
            self.root.after(0, self._training_complete)

    def _training_complete(self):
        """Called when training is complete."""
        self.is_running = False
        self.train_button.configure(state='normal')
        self.train_progress.stop()
        self.train_progress.grid_remove()

    def _start_evaluation(self):
        """Start model evaluation."""
        if self.is_running:
            return

        if not self._validate_evaluation_inputs():
            return

        self.is_running = True
        self.eval_button.configure(state='disabled')
        self.eval_progress.grid()
        self.eval_progress.start(10)
        self.eval_status_var.set("Evaluation in progress...")

        thread = threading.Thread(target=self._evaluation_worker, daemon=True)
        thread.start()

    def _evaluation_worker(self):
        """Worker function for evaluation."""
        old_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.eval_output_text)

        try:
            evaluator = IndividualModelEvaluatorGUI(
                ensemble_dir=self.eval_ensemble_dir.get(),
                data_dir=self.eval_data_dir.get()
            )

            # Check if stain normalization was used during training
            stain_norm = evaluator.metadata.get('stain_normalization', True)
            evaluator.load_data(apply_stain_normalization=stain_norm)
            evaluator.create_test_split()
            evaluator.evaluate_all_models()
            best_model = evaluator.print_comparison()

            detailed_seed = self.eval_detailed_seed.get().strip()
            if detailed_seed:
                evaluator.print_detailed_report(int(detailed_seed))

            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE")
            print("=" * 70)

            self.root.after(0, lambda: self.eval_status_var.set("Evaluation complete!"))
            self.root.after(0, lambda: self.save_eval_button.configure(state='normal'))

        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.eval_status_var.set(f"Error: {str(e)[:50]}..."))
            self.root.after(0, lambda: messagebox.showerror("Evaluation Error", str(e)))

        finally:
            sys.stdout = old_stdout
            self.root.after(0, self._evaluation_complete)

    def _evaluation_complete(self):
        """Called when evaluation is complete."""
        self.is_running = False
        self.eval_button.configure(state='normal')
        self.eval_progress.stop()
        self.eval_progress.grid_remove()


def main():
    root = tk.Tk()

    # Apply a modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        for theme in ['clam', 'alt', 'default']:
            if theme in available_themes:
                style.theme_use(theme)
                break
    except Exception:
        pass

    app = EnsembleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()