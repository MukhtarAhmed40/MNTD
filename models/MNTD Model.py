
### 2. models/mntd_model.py

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple

class MultiHeadAttention(layers.Layer):
    """Implementation of multi-head attention mechanism"""
    def __init__(self, num_heads: int, head_size: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        
    def build(self, input_shape):
        self.query_dense = layers.Dense(self.num_heads * self.head_size)
        self.key_dense = layers.Dense(self.num_heads * self.head_size)
        self.value_dense = layers.Dense(self.num_heads * self.head_size)
        self.combine_heads = layers.Dense(input_shape[-1])
        
    def call(self, inputs):
        # Split inputs into queries, keys and values
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)
        
        # Split into multiple heads
        queries = tf.reshape(queries, (-1, tf.shape(inputs)[1], self.num_heads, self.head_size))
        keys = tf.reshape(keys, (-1, tf.shape(inputs)[1], self.num_heads, self.head_size))
        values = tf.reshape(values, (-1, tf.shape(inputs)[1], self.num_heads, self.head_size))
        
        # Scaled dot-product attention
        scores = tf.einsum('bqhd,bkhd->bhqk', queries, keys) / tf.sqrt(tf.cast(self.head_size, tf.float32))
        attention = tf.nn.softmax(scores, axis=-1)
        output = tf.einsum('bhqk,bkhd->bqhd', attention, values)
        
        # Concatenate heads and apply final linear layer
        output = tf.reshape(output, (-1, tf.shape(inputs)[1], self.num_heads * self.head_size))
        return self.combine_heads(output)

class MNTDModel:
    """Main MNTD model implementation"""
    def __init__(self, 
                 input_shape: Tuple[int, int] = (30, 64),
                 num_classes: int = 2,
                 num_filters: int = 32,
                 kernel_size: int = 3,
                 bilstm_units: int = 128,
                 num_heads: int = 4,
                 head_size: int = 64,
                 dense_units: int = 64,
                 dropout_rate: float = 0.05,
                 l2_reg: float = 0.1):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bilstm_units = bilstm_units
        self.num_heads = num_heads
        self.head_size = head_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the complete MNTD model architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN for spatial features
        x = layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(inputs)
        x = layers.MaxPooling1D(pool_size=5)(x)
        
        # BiLSTM for temporal patterns
        x = layers.Bidirectional(
            layers.LSTM(
                self.bilstm_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(self.l2_reg))
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Multi-Head Attention
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_size=self.head_size
        )(x)
        
        # Classification head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax'
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def compile(self, learning_rate: float = 0.002):
        """Compile the model with adaptive loss"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._adaptive_loss,
            metrics=['accuracy']
        )
    
    def _adaptive_loss(self, y_true, y_pred):
        """Combined loss function with cross-entropy and contrastive components"""
        # Cross-entropy loss
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Contrastive loss
        contrastive_loss = self._contrastive_loss(y_pred, tf.argmax(y_true, axis=1))
        
        # Total loss (weighting factors from paper)
        return ce_loss + 0.1 * contrastive_loss
    
    def _contrastive_loss(self, embeddings, labels, temperature=0.1):
        """Compute contrastive loss for feature discrimination"""
        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        
        # Compute similarity matrix
        sim_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
        
        # Create mask for positive pairs
        mask = tf.cast(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)), tf.float32)
        
        # Subtract max for numerical stability
        sim_max = tf.reduce_max(sim_matrix, axis=1, keepdims=True)
        sim_exp = tf.exp(sim_matrix - sim_max)
        
        # Compute loss
        pos_sim = tf.reduce_sum(sim_exp * mask, axis=1, keepdims=True)
        sum_exp = tf.reduce_sum(sim_exp, axis=1, keepdims=True)
        loss = -tf.math.log(pos_sim / sum_exp)
        
        return tf.reduce_mean(loss)
