# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_mntd(input_dim,
               conv_filters=64,
               conv_kernel=3,
               pool_size=2,
               bilstm_units=128,
               mha_heads=4,
               dense_units=64,
               dropout=0.05,
               l2_reg=1e-4):
    """
    Build the MNTD model per the paper:
    Conv1D -> MaxPool -> BiLSTM (ELU) x2 -> MultiHeadAttention -> Dense -> Softmax
    """
    inp = layers.Input(shape=(input_dim,), name='flow_input')  # flattened / preprocessed per-channel representation
    # For simplicity we treat input as 1D vector -> expand dims so Conv1D works
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel, activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    # flatten pool's time dimension and treat as sequence length for BiLSTM
    # reshape to (num_flows, features) form expected by BiLSTM
    # here we use the pooled time dimension as the 'time' axis
    x = layers.Permute((2,1))(x)  # swap dims -> (batch, channels, timesteps)
    # BiLSTM layers (ELU activation inside LSTM as in paper)
    x = layers.Bidirectional(layers.LSTM(bilstm_units//2, activation='elu', return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(bilstm_units//2, activation='elu', return_sequences=True))(x)
    # collapse sequence -> keep sequence as is for attention; apply MultiHeadAttention
    # Use Keras MultiHeadAttention which expects query/key/value shapes.
    # We'll use a global self-attention pooling after MHA
    mha = layers.MultiHeadAttention(num_heads=mha_heads, key_dim=bilstm_units//mha_heads)
    attn_out = mha(x, x)
    # Pool across time axis
    pooled = layers.GlobalAveragePooling1D()(attn_out)
    dense = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(pooled)
    drop = layers.Dropout(dropout)(dense)
    out = layers.Dense(2, activation='softmax')(drop)
    model = models.Model(inputs=inp, outputs=out, name='MNTD')
    return model
