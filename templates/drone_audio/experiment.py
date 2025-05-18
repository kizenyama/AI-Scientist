import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from tqdm import tqdm
import requests
import zipfile
import io
import shutil
import random
from typing import Dict, List, Tuple


class DroneAudioDataset:
    def __init__(self, dataset_path, frac_train=0.7, frac_val=0.15):
        self.dataset_path = dataset_path
        self.frac_train = frac_train
        self.frac_val = frac_val
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)
        
        # Download and prepare dataset
        self.download_dataset()
        
        # Find audio files for each class
        self.class_files = self.find_audio_files()
        
        # Report the number of files for each class
        for class_name, files in self.class_files.items():
            print(f"Found {len(files)} files for class {class_name}")
    
    def download_dataset(self):
        """Download Sara Al-Emadi's drone audio dataset from GitHub"""
        if not os.path.exists(os.path.join(self.dataset_path, "Binary_Drone_Audio")):
            print("\n--- DOWNLOADING DATASET ---")
            repo_url = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"
            
            try:
                print(f"Downloading from {repo_url}...")
                response = requests.get(repo_url)
                response.raise_for_status()
                
                # Extract the zip file
                print("Extracting dataset...")
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(os.path.dirname(self.dataset_path))
                
                # The extracted folder will be named DroneAudioDataset-master
                extracted_path = os.path.join(os.path.dirname(self.dataset_path), 'DroneAudioDataset-master')
                
                if os.path.exists(self.dataset_path) and not os.listdir(self.dataset_path):
                    os.rmdir(self.dataset_path)
                elif os.path.exists(self.dataset_path) and self.dataset_path != extracted_path:
                    shutil.rmtree(self.dataset_path)
                
                shutil.move(extracted_path, self.dataset_path)
                print(f"Dataset successfully downloaded and extracted to {self.dataset_path}")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Attempting to continue with existing data if available...")
        else:
            print(f"Dataset already exists at {self.dataset_path}")
    
    def find_audio_files(self):
        """Find 'yes_drone' and 'unknown' audio files for binary classification"""
        print("Looking for all drone and unknown audio files in the dataset...")
        
        # First try to find the binary folder specifically
        binary_dataset_path = os.path.join(self.dataset_path, 'Binary_Drone_Audio')
        if not os.path.exists(binary_dataset_path):
            print("Binary dataset folder not found. Searching in root folder...")
            binary_dataset_path = self.dataset_path
        
        print(f"Using binary dataset path: {binary_dataset_path}")
        
        yes_drone_files = []
        unknown_files = []
        
        # Walk through the dataset to find the files
        for root, _, files in os.walk(binary_dataset_path):
            is_drone_dir = any(drone_str in root.lower() for drone_str in ['yes_drone', 'drone', 'bebop', 'mambo'])
            is_unknown_dir = 'unknown' in root.lower()
            
            for file in files:
                if file.endswith('.wav'):
                    if is_drone_dir and not is_unknown_dir:
                        yes_drone_files.append(os.path.join(root, file))
                    elif is_unknown_dir:
                        unknown_files.append(os.path.join(root, file))
        
        print(f"Found {len(yes_drone_files)} yes_drone audio files")
        print(f"Found {len(unknown_files)} unknown audio files")
        
        if len(yes_drone_files) == 0 or len(unknown_files) == 0:
            raise FileNotFoundError("Missing audio files. Cannot proceed with experiment.")
        
        return {
            'yes_drone': yes_drone_files,
            'unknown': unknown_files
        }
    
    def extract_features(self, config):
        """Extract MFCC features from audio files"""
        print("\n--- EXTRACTING FEATURES ---")
        features_list = []
        labels = []
        class_names = list(self.class_files.keys())
        
        # Extract features from each file
        for class_idx, class_name in enumerate(class_names):
            print(f"Processing class: {class_name}")
            
            for audio_file in tqdm(self.class_files[class_name]):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=config['sample_rate'], 
                                          duration=config['duration'])
                    
                    # Pad or trim to fixed length
                    expected_length = int(config['sample_rate'] * config['duration'])
                    if len(audio) < expected_length:
                        audio = np.pad(audio, (0, expected_length - len(audio)))
                    audio = audio[:expected_length]
                    
                    # Extract MFCCs
                    mfccs = librosa.feature.mfcc(
                        y=audio,
                        sr=config['sample_rate'],
                        n_mfcc=config['n_mfcc'],
                        hop_length=config['hop_length']
                    )
                    
                    # Normalize
                    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
                    
                    features_list.append(mfccs)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        print(f"Feature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def prepare_dataset(self, config):
        """Prepare dataset for training"""
        # Extract features
        X, y = self.extract_features(config)
        
        # Add channel dimension for CNN
        X = X[..., np.newaxis]
        
        # Split data into train/val/test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - self.frac_train), random_state=42, stratify=y
        )
        
        # Calculate the fraction for validation from the remaining data
        temp_fraction = self.frac_val / (1 - self.frac_train)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - temp_fraction), random_state=42, stratify=y_temp
        )
        
        print(f"Dataset shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Validation class distribution: {np.bincount(y_val)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        # Calculate class weights for handling imbalance
        class_weight = None
        if len(np.unique(y_train)) > 1:
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weight = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
            print(f"Using class weights: {class_weight}")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "class_names": list(self.class_files.keys()),
            "class_weight": class_weight
        }


class ModelBuilder:
    """Class for building TinyML models for drone audio classification"""
    
    @staticmethod
    def build_cnn_model(input_shape: Tuple, num_classes: int) -> keras.Model:
        """
        Build a CNN model for drone audio classification
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=input_shape),
            
            # First Conv Block
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second Conv Block
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Global Pooling + Classification
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def build_lightweight_cnn_model(input_shape: Tuple, num_classes: int) -> keras.Model:
        """
        Build a lightweight CNN model for drone audio classification
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=input_shape),
            
            # Simplified Conv Block 1
            keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Simplified Conv Block 2
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Global Pooling + Classification
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def build_dscnn_model(input_shape: Tuple, num_classes: int) -> keras.Model:
        """
        Build a Depthwise Separable CNN model for drone audio classification
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=input_shape),
            
            # First Conv layer
            keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Depthwise Separable Conv Block
            keras.layers.DepthwiseConv2D(3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(16, 1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Global Pooling + Classification
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class TinyMLOptimizer:
    """Class for optimizing models for TinyML deployment"""
    
    @staticmethod
    def convert_to_tflite(model: keras.Model, out_path: str) -> Tuple[bytes, float]:
        """
        Convert Keras model to TFLite format
        
        Args:
            model: Keras model to convert
            out_path: Path to save TFLite model
            
        Returns:
            tflite_model: TFLite model
            model_size_kb: Size of TFLite model in KB
        """
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the model to disk
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size_kb = len(tflite_model) / 1024
        print(f"TFLite Model Size: {model_size_kb:.2f} KB")
        
        return tflite_model, model_size_kb
    
    @staticmethod
    def quantize_model(model: keras.Model, X_train: np.ndarray, out_path: str) -> Tuple[bytes, float]:
        """
        Quantize Keras model to reduce size
        
        Args:
            model: Keras model to quantize
            X_train: Training data for representative dataset
            out_path: Path to save quantized TFLite model
            
        Returns:
            quantized_tflite_model: Quantized TFLite model
            quantized_size_kb: Size of quantized TFLite model in KB
        """
        def representative_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1].astype(np.float32)]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        try:
            # Try full integer quantization first
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            quantized_tflite_model = converter.convert()
        except Exception as e:
            print(f"Full integer quantization failed: {e}")
            print("Falling back to float16 quantization")
            # Try float16 quantization as fallback
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            try:
                quantized_tflite_model = converter.convert()
            except Exception as e2:
                print(f"Float16 quantization also failed: {e2}")
                print("Using unquantized model")
                # Convert to TFLite without quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                quantized_tflite_model = converter.convert()
        
        # Save quantized model
        with open(out_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        # Calculate quantized model size
        quantized_size_kb = len(quantized_tflite_model) / 1024
        print(f"Quantized TFLite Model Size: {quantized_size_kb:.2f} KB")
        
        return quantized_tflite_model, quantized_size_kb


def build_model(model_type, input_shape, num_classes):
    """Build a CNN model based on the specified type"""
    if model_type == 'cnn':
        model = ModelBuilder.build_cnn_model(input_shape, num_classes)
    elif model_type == 'lightweight_cnn':
        model = ModelBuilder.build_lightweight_cnn_model(input_shape, num_classes)
    elif model_type == 'dscnn':
        model = ModelBuilder.build_dscnn_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def optimize_for_tinyml(model, X_train, out_dir, model_type):
    """Optimize model for TinyML deployment"""
    print("\n--- APPLYING TINYML OPTIMIZATIONS ---")
    tflite_model_path = os.path.join(out_dir, f"drone_detector_{model_type}.tflite")
    quantized_tflite_path = os.path.join(out_dir, f"drone_detector_{model_type}_quantized.tflite")
    
    # Convert to TFLite
    tflite_model, model_size_kb = TinyMLOptimizer.convert_to_tflite(model, tflite_model_path)
    
    # Quantize model
    _, quantized_size_kb = TinyMLOptimizer.quantize_model(model, X_train, quantized_tflite_path)
    
    # Calculate size reduction
    size_reduction_percent = (1 - quantized_size_kb/model_size_kb) * 100
    print(f"Size reduction: {size_reduction_percent:.2f}%")
    
    return {
        "model_size_kb": model_size_kb,
        "quantized_size_kb": quantized_size_kb,
        "size_reduction_percent": size_reduction_percent,
        "model_path": tflite_model_path,
        "quantized_model_path": quantized_tflite_path
    }


class Experiment:
    """Class for running TinyML drone audio classification experiments"""
    
    def __init__(self, out_dir: str, seed: int = 42):
        """
        Initialize experiment
        
        Args:
            out_dir: Output directory for experiment results
            seed: Random seed for reproducibility
        """
        self.out_dir = out_dir
        self.seed = seed
        
        # Create output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize results storage
        self.results = {}
    
    def run_single_experiment(self, model_type: str, dataset: DroneAudioDataset, config: Dict) -> Dict:
        """
        Run a single experiment
        
        Args:
            model_type: Type of model to build ('cnn', 'lightweight_cnn', or 'dscnn')
            dataset: Dataset to use
            config: Configuration dictionary
            
        Returns:
            Results dictionary
        """
        print(f"\n--- RUNNING EXPERIMENT: {model_type} ---")
        
        # Prepare dataset
        dataset_info = dataset.prepare_dataset(config)
        X_train = dataset_info["X_train"]
        X_val = dataset_info["X_val"]
        X_test = dataset_info["X_test"]
        y_train = dataset_info["y_train"]
        y_val = dataset_info["y_val"]
        y_test = dataset_info["y_test"]
        class_names = dataset_info["class_names"]
        class_weight = dataset_info["class_weight"]
        
        # Build model
        print("\n--- BUILDING MODEL ---")
        input_shape = X_train.shape[1:]
        num_classes = len(class_names)
        
        model = build_model(model_type, input_shape, num_classes)
        model.summary()
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-5
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.out_dir, f"logs_{model_type}"),
                histogram_freq=1
            )
        ]
        
        # Train model
        print("\n--- TRAINING MODEL ---")
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        print("\n--- EVALUATING MODEL ---")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Calculate confusion matrix and classification report
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        cr = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # TinyML optimization
        optimization_results = optimize_for_tinyml(model, X_train, self.out_dir, model_type)
        
        # Compile results
        results = {
            "model_type": model_type,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "training_time": training_time,
            "model_size_kb": optimization_results["model_size_kb"],
            "quantized_size_kb": optimization_results["quantized_size_kb"],
            "size_reduction_percent": optimization_results["size_reduction_percent"],
            "model_path": optimization_results["model_path"],
            "quantized_model_path": optimization_results["quantized_model_path"],
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "history": {k: [float(val) for val in v] for k, v in history.history.items()},
            "steps": list(range(1, len(history.history["loss"]) + 1))
        }
        
        # Save individual results
        with open(os.path.join(self.out_dir, f"results_{model_type}.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_experiments(self, dataset: DroneAudioDataset, config: Dict, model_types: List[str] = None) -> Dict:
        """
        Run experiments with different model types
        
        Args:
            dataset: Dataset to use
            config: Configuration dictionary
            model_types: List of model types to experiment with
            
        Returns:
            Results dictionary
        """
        if model_types is None:
            model_types = ['cnn', 'lightweight_cnn', 'dscnn']
        
        print("\n--- STARTING TINYML DRONE AUDIO CLASSIFICATION EXPERIMENTS ---")
        print("TensorFlow version:", tf.__version__)
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        
        # Configure experiment
        print("\n--- HYPOTHESIS ---")
        print("Hypothesis: Different CNN architectures optimized for TinyML will be compared")
        print("to find the best balance of accuracy vs. model size for binary drone detection.")
        
        # Run experiments for each model type
        all_results = {}
        for model_type in model_types:
            results = self.run_single_experiment(model_type, dataset, config)
            all_results[model_type] = results
        
        # Save combined results
        with open(os.path.join(self.out_dir, "all_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save as numpy file for compatibility with plotting script
        np.save(os.path.join(self.out_dir, "all_results.npy"), all_results)
        
        # Print summary of results
        print("\n--- EXPERIMENT SUMMARY ---")
        print("BINARY CLASSIFICATION RESULTS:")
        for model_type, results in all_results.items():
            print(f"Model: {model_type}")
            print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"  Model Size: {results['model_size_kb']:.2f} KB")
            print(f"  Quantized Size: {results['quantized_size_kb']:.2f} KB")
            print(f"  Size Reduction: {results['size_reduction_percent']:.2f}%")
        
        return all_results


def run_experiment(model_type, dataset, config, out_dir):
    """Run a single experiment with the specified model type"""
    print(f"\n--- RUNNING EXPERIMENT: {model_type} ---")
    
    # Prepare dataset
    dataset_info = dataset.prepare_dataset(config)
    X_train = dataset_info["X_train"]
    X_val = dataset_info["X_val"]
    X_test = dataset_info["X_test"]
    y_train = dataset_info["y_train"]
    y_val = dataset_info["y_val"]
    y_test = dataset_info["y_test"]
    class_names = dataset_info["class_names"]
    class_weight = dataset_info["class_weight"]
    
    # Build model
    print("\n--- BUILDING MODEL ---")
    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    model = build_model(model_type, input_shape, num_classes)
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]
    
    # Train model
    print("\n--- TRAINING MODEL ---")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    print("\n--- EVALUATING MODEL ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Calculate confusion matrix and classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    cr = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # TinyML optimization
    optimization_results = optimize_for_tinyml(model, X_train, out_dir, model_type)
    
    # Compile results
    results = {
        "model_type": model_type,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_time": training_time,
        "model_size_kb": optimization_results["model_size_kb"],
        "quantized_size_kb": optimization_results["quantized_size_kb"],
        "size_reduction_percent": optimization_results["size_reduction_percent"],
        "model_path": optimization_results["model_path"],
        "quantized_model_path": optimization_results["quantized_model_path"],
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "history": {k: [float(val) for val in v] for k, v in history.history.items()},
        "steps": list(range(1, len(history.history["loss"]) + 1))
    }
    
    # Save results
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"results_{model_type}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main(out_dir="run_0", dataset_path="/content/DroneAudioDataset", 
         model_types=None, seed=42):
    """
    Main function to run experiments, designed to work in both script and notebook environments
    
    Args:
        out_dir: Output directory for results
        dataset_path: Path to dataset
        model_types: List of model types to test
        seed: Random seed for reproducibility
    """
    # Handle default argument
    if model_types is None:
        model_types = ["cnn", "lightweight_cnn", "dscnn"]
    
    # If model_types is a string, convert to list
    if isinstance(model_types, str):
        model_types = [model_types]
    
    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Feature extraction and training configuration
    config = {
        'sample_rate': 16000,
        'duration': 1.0,
        'hop_length': 512,
        'n_mfcc': 13,
        'batch_size': 64,
        'epochs': 30
    }
    
    print("\n--- STARTING TINYML DRONE AUDIO CLASSIFICATION EXPERIMENTS ---")
    print("TensorFlow version:", tf.__version__)
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))
    
    print("\n--- HYPOTHESIS ---")
    print("Hypothesis: Different CNN architectures optimized for TinyML will be compared")
    print("to find the best balance of accuracy vs. model size for binary drone detection.")
    
    # Create dataset
    dataset = DroneAudioDataset(dataset_path)
    
    # Run experiments for each model type
    all_results = {}
    for model_type in model_types:
        results = run_experiment(model_type, dataset, config, out_dir)
        all_results[model_type] = results
    
    # Save combined results
    with open(os.path.join(out_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as numpy file for compatibility with plotting script
    np.save(os.path.join(out_dir, "all_results.npy"), all_results)
    
    # Print summary of results
    print("\n--- EXPERIMENT SUMMARY ---")
    print("BINARY CLASSIFICATION RESULTS:")
    for model_type, results in all_results.items():
        print(f"Model: {model_type}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Model Size: {results['model_size_kb']:.2f} KB")
        print(f"  Quantized Size: {results['quantized_size_kb']:.2f} KB")
        print(f"  Size Reduction: {results['size_reduction_percent']:.2f}%")
    # Convert to AI Scientist format
    ai_scientist_results = {}

    # Add metrics from each model type
    for model_type, results in all_results.items():
        ai_scientist_results[f"{model_type}_test_accuracy"] = {
            "means": [results["test_accuracy"]]
        }
        ai_scientist_results[f"{model_type}_model_size_kb"] = {
            "means": [results["model_size_kb"]]
        }
        ai_scientist_results[f"{model_type}_quantized_size_kb"] = {
            "means": [results["quantized_size_kb"]]
        }
        ai_scientist_results[f"{model_type}_size_reduction_percent"] = {
            "means": [results["size_reduction_percent"]]
        }

    # Save in the exact location and format AI Scientist expects
    with open(os.path.join(out_dir, "final_info.json"), 'w') as f:
        json.dump(ai_scientist_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    # When running as a script, handle command-line arguments
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run TinyML drone audio classification experiment")
        parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
        parser.add_argument("--dataset_path", type=str, default="/content/DroneAudioDataset", help="Path to dataset")
        parser.add_argument("--model_types", type=str, nargs='+', default=["cnn", "lightweight_cnn", "dscnn"], 
                           help="List of model types to experiment with")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        # For Colab compatibility, only use args we defined
        argv = []
        for arg in sys.argv[1:]:
            if any(arg.startswith(f"--{param}") for param in ["out_dir", "dataset_path", "model_types", "seed"]):
                argv.append(arg)
        
        args = parser.parse_args(argv)
        
        # Call main with parsed arguments
        main(
            out_dir=args.out_dir,
            dataset_path=args.dataset_path,
            model_types=args.model_types,
            seed=args.seed
        )

    except SystemExit:
        # If argument parsing fails, just use defaults
        print("Using default arguments due to parsing error")
        main()