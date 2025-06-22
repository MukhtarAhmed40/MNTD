import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

class MNTD:
    def __init__(self, input_shape=(30, 64), num_classes=2):
        """
        Initialize the MNTD model with default parameters from the paper
        Args:
            input_shape: Shape of input data (number of packets, bytes per packet)
            num_classes: Number of output classes (binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
        # Hyperparameters from paper
        self.learning_rate = 0.002
        self.dropout_rate = 0.05
        self.l2_reg = 0.1
        self.batch_size = 128
        self.num_epochs = 50
        self.num_filters = 32
        self.kernel_size = 4
        self.pool_size = 5
        self.bilstm_units = 128
        self.num_heads = 8
        self.head_size = 64
        self.dense_units = 64
        
    def build_model(self):
        """Build the complete MNTD model architecture"""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # 1D CNN for spatial feature extraction
        x = layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, 
                         activation='elu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=self.pool_size)(x)
        
        # BiLSTM for temporal dependencies
        x = layers.Bidirectional(
            layers.LSTM(self.bilstm_units, return_sequences=True, 
                       kernel_regularizer=regularizers.l2(self.l2_reg))
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Multi-Head Attention
        x = self.multi_head_attention(x)
        
        # Global Average Pooling and Dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units, activation='elu', 
                        kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Custom loss combining cross-entropy and contrastive loss
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.adaptive_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def multi_head_attention(self, x):
        """Implement multi-head attention mechanism"""
        # Create multiple attention heads
        attention_outputs = []
        for _ in range(self.num_heads):
            # Query, Key, Value projections
            query = layers.Dense(self.head_size)(x)
            key = layers.Dense(self.head_size)(x)
            value = layers.Dense(self.head_size)(x)
            
            # Scaled dot-product attention
            attention_scores = tf.matmul(query, key, transpose_b=True)
            attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.head_size, tf.float32))
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attention_output = tf.matmul(attention_weights, value)
            
            attention_outputs.append(attention_output)
        
        # Concatenate all attention heads
        concatenated_attention = tf.concat(attention_outputs, axis=-1)
        
        # Final linear transformation
        output = layers.Dense(self.bilstm_units * 2)(concatenated_attention)  # *2 for BiLSTM
        return output
    
    def adaptive_loss(self, y_true, y_pred):
        """Combined loss function with cross-entropy and contrastive loss"""
        # Cross-entropy loss
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Contrastive loss (simplified implementation)
        batch_size = tf.shape(y_pred)[0]
        y_true_labels = tf.argmax(y_true, axis=1)
        contrastive_loss = self.compute_contrastive_loss(y_pred, y_true_labels, batch_size)
        
        # L2 regularization is already handled in the layers
        total_loss = ce_loss + 0.1 * contrastive_loss  # Weighting factor from paper
        
        return total_loss
    
    def compute_contrastive_loss(self, embeddings, labels, batch_size, temperature=0.1):
        """Compute contrastive loss for feature discrimination"""
        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
        
        # Create mask for positive pairs (same class)
        mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
        mask = tf.cast(mask, tf.float32)
        
        # Subtract max for numerical stability
        sim_max = tf.reduce_max(similarity_matrix, axis=1, keepdims=True)
        sim_exp = tf.exp(similarity_matrix - sim_max)
        
        # Sum of exponentials for positive and negative pairs
        sum_exp = tf.reduce_sum(sim_exp, axis=1, keepdims=True)
        pos_sim = tf.reduce_sum(sim_exp * mask, axis=1, keepdims=True)
        
        # Contrastive loss
        loss = -tf.math.log(pos_sim / sum_exp)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional validation data"""
        # Convert labels to one-hot encoding if needed
        if len(y_train.shape) == 1:
            y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
            if y_val is not None:
                y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set"""
        if len(y_test.shape) == 1:
            y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        loss, accuracy = results[0], results[1]
        
        # Get predictions for additional metrics
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        # Confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_classes, y_pred_classes))
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def optimize_with_awdv(self, X_train, y_train, X_val, y_val, num_particles=10, max_iter=20):
        """Optimize hyperparameters using Adaptive Weighted Delay Velocity (AWDV)"""
        # Initialize particles with random hyperparameters
        particles = []
        for _ in range(num_particles):
            particle = {
                'learning_rate': 10**np.random.uniform(-4, -2),
                'dropout_rate': np.random.uniform(0.01, 0.2),
                'batch_size': int(2**np.random.randint(5, 9)),
                'num_filters': int(2**np.random.randint(3, 6)),
                'kernel_size': int(2**np.random.randint(2, 5))
            }
            particles.append(particle)
        
        # Initialize personal and global best
        personal_best = [None] * num_particles
        global_best = None
        global_best_score = -np.inf
        
        # PSO-AWDV parameters from paper
        alpha = 0.9  # Influence of personal best
        beta = 0.5   # Influence of global best
        gamma = 0.3  # Adaptive delay component
        
        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}")
            
            for i, particle in enumerate(particles):
                # Update model with current particle's hyperparameters
                self.learning_rate = particle['learning_rate']
                self.dropout_rate = particle['dropout_rate']
                self.batch_size = particle['batch_size']
                self.num_filters = particle['num_filters']
                self.kernel_size = particle['kernel_size']
                
                # Rebuild and train model
                self.model = self.build_model()
                history = self.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_results = self.evaluate(X_val, y_val)
                current_score = val_results['accuracy']
                
                # Update personal best
                if personal_best[i] is None or current_score > personal_best[i]['score']:
                    personal_best[i] = {
                        'params': particle.copy(),
                        'score': current_score
                    }
                
                # Update global best
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best = particle.copy()
            
            # Update particle velocities and positions with AWDV
            for i, particle in enumerate(particles):
                # Calculate adaptive delay component
                delay = np.random.rand() * (global_best_score - personal_best[i]['score'])
                
                # Update velocity (simplified)
                for key in particle.keys():
                    # Calculate velocity components
                    cognitive = alpha * (personal_best[i]['params'][key] - particle[key])
                    social = beta * (global_best[key] - particle[key])
                    
                    # Update velocity with adaptive delay
                    velocity = cognitive + social + gamma * delay
                    
                    # Update position
                    if isinstance(particle[key], int):
                        particle[key] = max(1, int(particle[key] + velocity))
                    else:
                        particle[key] += velocity
                
                # Clamp values to valid ranges
                particle['learning_rate'] = np.clip(particle['learning_rate'], 1e-4, 1e-2)
                particle['dropout_rate'] = np.clip(particle['dropout_rate'], 0.01, 0.2)
                particle['batch_size'] = max(32, min(512, particle['batch_size']))
                particle['num_filters'] = max(8, min(64, particle['num_filters']))
                particle['kernel_size'] = max(2, min(8, particle['kernel_size']))
        
        # Set model to best found parameters
        print(f"Best validation accuracy: {global_best_score:.4f}")
        print("Best parameters:", global_best)
        
        self.learning_rate = global_best['learning_rate']
        self.dropout_rate = global_best['dropout_rate']
        self.batch_size = global_best['batch_size']
        self.num_filters = global_best['num_filters']
        self.kernel_size = global_best['kernel_size']
        
        self.model = self.build_model()
        return global_best

# Example usage
if __name__ == "__main__":
    # Load and preprocess your dataset (replace with actual data loading)
    # This is just a placeholder for demonstration
    print("Loading and preprocessing data...")
    num_samples = 10000
    num_packets = 30
    bytes_per_packet = 64
    
    # Generate synthetic data similar to CICIDS2017 format
    X = np.random.randint(0, 256, size=(num_samples, num_packets, bytes_per_packet)).astype('float32')
    X = X / 255.0  # Normalize byte values to [0,1]
    
    # Generate binary labels (0=benign, 1=malicious)
    y = np.random.randint(0, 2, size=(num_samples,))
    
    # Split data into train, validation, test sets (80:10:10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and train MNTD model
    print("Initializing MNTD model...")
    mntd = MNTD(input_shape=(num_packets, bytes_per_packet))
    
    # Option 1: Train with default hyperparameters
    print("Training model with default hyperparameters...")
    start_time = time.time()
    history = mntd.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_results = mntd.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Option 2: Optimize hyperparameters with AWDV first
    # print("Optimizing hyperparameters with AWDV...")
    # best_params = mntd.optimize_with_awdv(X_train, y_train, X_val, y_val)
    # print("Training with optimized hyperparameters...")
    # history = mntd.train(X_train, y_train, X_val, y_val)
    # test_results = mntd.evaluate(X_test, y_test)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
