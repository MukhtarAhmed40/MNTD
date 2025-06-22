import argparse
import yaml
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from models.mntd_model import MNTDModel
from models.awdv_optimizer import AWDVOptimizer
from data.sample_data_loader import load_dataset

def train_model(config: Dict[str, Any], optimize: bool = False):
    """Main training function for MNTD model"""
    # Load and preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(config['data_path'])
    
    # Initialize model
    mntd = MNTDModel(
        input_shape=tuple(config['input_shape']),
        num_classes=config['num_classes'],
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        bilstm_units=config['bilstm_units'],
        num_heads=config['num_heads'],
        head_size=config['head_size'],
        dense_units=config['dense_units'],
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg']
    )
    
    # Hyperparameter optimization
    if optimize:
        print("Running AWDV optimization...")
        optimizer = AWDVOptimizer(mntd)
        best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
        print(f"Best parameters: {best_params}")
        
        # Update model with optimized parameters
        mntd = MNTDModel(**best_params)
    
    # Compile model
    mntd.compile(learning_rate=config['learning_rate'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=config['reduce_lr_patience']),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"results/saved_models/mntd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5",
            save_best_only=True,
            monitor='val_loss')
    ]
    
    # Train model
    print("Starting training...")
    history = mntd.model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = mntd.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Train MNTD model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--optimize', action='store_true', help='Run AWDV optimization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory if not exists
    Path("results/saved_models").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # Run training
    history = train_model(config, optimize=args.optimize)
    
    # Save training history plot
    plot_history(history, config)

def plot_history(history, config):
    """Plot and save training history"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"results/figures/training_history_{timestamp}.png")
    plt.close()

if __name__ == "__main__":
    main()
