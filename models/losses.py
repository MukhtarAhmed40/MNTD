# src/losses.py
import tensorflow as tf
import tensorflow.keras.backend as K

def cross_entropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def contrastive_loss_from_embeddings(anchor_emb, positive_embs, negative_embs, temperature=0.1):
    """
    Implements the contrastive loss variant from Eq.19.
    anchor_emb: (B, d)
    positive_emb: (B, d)
    negative_embs: (B, Nneg, d) or (B, d) for single neg per anchor
    """
    # normalize
    anchor = tf.math.l2_normalize(anchor_emb, axis=1)
    pos = tf.math.l2_normalize(positive_embs, axis=1)
    # compute numerator
    sim_pos = tf.reduce_sum(anchor * pos, axis=1) / temperature
    exp_pos = tf.exp(sim_pos)
    # negatives
    # if negatives shaped (B, d)
    if len(negative_embs.shape) == 2:
        neg = tf.math.l2_normalize(negative_embs, axis=1)
        sim_neg = tf.exp(tf.reduce_sum(anchor * neg, axis=1) / temperature)
        denom = exp_pos + sim_neg
    else:
        neg = tf.math.l2_normalize(negative_embs, axis=2)
        sim_neg = tf.reduce_sum(anchor[:, None, :] * neg, axis=2)  # (B, Nneg)
        exp_neg = tf.reduce_sum(tf.exp(sim_neg / temperature), axis=1)
        denom = exp_pos + exp_neg
    loss = -tf.reduce_mean(tf.math.log(exp_pos / denom + 1e-12))
    return loss

class AdaptiveTotalLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_l2=1e-4, lambda_contrastive=0.2):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_contrastive = lambda_contrastive

    def call(self, y_true, y_pred, model=None, contrastive_component=0.0):
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights]) * self.lambda_l2 if model is not None else 0.0
        return tf.reduce_mean(ce) + l2 + self.lambda_contrastive * contrastive_component
