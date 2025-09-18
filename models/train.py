# src/train.py
import os
import numpy as np
import tensorflow as tf
import yaml
from .preprocess import load_csvs, basic_preprocess
from .model import build_mntd
from .losses import AdaptiveTotalLoss, contrastive_loss_from_embeddings
from .sampling import triplet_sampling
from .awdv import awdv_optimize

def default_eval_fn(position, train_data, val_data, config):
    """
    position: hyperparams vector -> [lr, dropout, batch_size]
    Quick evaluate: train for a few epochs and return validation loss.
    This function should be replaced with a faster proxy for real AWDV runs.
    """
    lr = float(position[0])
    dropout = float(position[1])
    batch_size = int(position[2])
    # Build model with these hyperparams
    input_dim = config['model']['input_dim']
    model = build_mntd(input_dim,
                      conv_filters=config['model']['conv_filters'],
                      conv_kernel=config['model']['conv_kernel'],
                      pool_size=config['model']['pool_size'],
                      bilstm_units=config['model']['bilstm_units'],
                      mha_heads=config['model']['mha_heads'],
                      dense_units=config['model']['dense_units'],
                      dropout=dropout,
                      l2_reg=config['training']['lambda_l2'])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # quick one-epoch train on small subset for speed
    X_train, y_train = train_data
    X_val, y_val = val_data
    # sample small subset
    sidx = np.random.choice(len(X_train), min(1024, len(X_train)), replace=False)
    vidx = np.random.choice(len(X_val), min(512, len(X_val)), replace=False)
    model.fit(X_train[sidx], y_train[sidx], batch_size=batch_size, epochs=1, verbose=0)
    loss, acc = model.evaluate(X_val[vidx], y_val[vidx], verbose=0)
    return loss

def train_pipeline(config_path='config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Load CSVs
    csvs = cfg['dataset']['csv_paths']
    df = load_csvs(csvs)
    # Determine numeric and categorical columns automatically (simple heuristic)
    label_col = 'label'
    meta_cols = ['src_ip','dst_ip','dst_port','timestamp']  # used in channel construction if available
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in meta_cols + [label_col]]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in meta_cols + [label_col]]
    X, y, scaler, ohe = basic_preprocess(df, numeric_cols, cat_cols)
    # train/val/test split stratified
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=cfg['seed'])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=cfg['seed'])
    # AWDV hyperparam bounds: [lr, dropout, batch_size]
    bounds = [(1e-4, 1e-2), (0.01, 0.2), (32, 256)]
    def eval_fn(pos): return default_eval_fn(pos, (X_train, y_train), (X_val, y_val), cfg)
    best_pos, best_score = awdv_optimize(eval_fn, bounds,
                                         particles=cfg['awdv']['particles'],
                                         iterations=cfg['awdv']['iterations'],
                                         alpha=cfg['awdv']['alpha'],
                                         beta=cfg['awdv']['beta'],
                                         gamma=cfg['awdv']['gamma'])
    lr, dropout, batch_size = best_pos[0], best_pos[1], int(best_pos[2])
    print("AWDV selected lr,dropout,batch_size:", lr, dropout, batch_size)
    # Build final model
    model = build_mntd(cfg['model']['input_dim'],
                       conv_filters=cfg['model']['conv_filters'],
                       conv_kernel=cfg['model']['conv_kernel'],
                       pool_size=cfg['model']['pool_size'],
                       bilstm_units=cfg['model']['bilstm_units'],
                       mha_heads=cfg['model']['mha_heads'],
                       dense_units=cfg['model']['dense_units'],
                       dropout=dropout,
                       l2_reg=cfg['training']['lambda_l2'])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train with contrastive loss mixed in each batch
    epochs = cfg['training']['epochs']
    lambda_con = cfg['training']['lambda_contrastive']
    temp = cfg['contrastive']['temperature']
    for epoch in range(epochs):
        # simple epoch training by batches (no dataset API for clarity)
        idxs = np.arange(len(X_train))
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i+batch_size]
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            # sample triplets within full train set for contrastive component
            a,p,n = triplet_sampling(Xb, yb, pos_thresh=cfg['contrastive']['pos_cos_threshold'], neg_thresh=cfg['contrastive']['neg_cos_threshold'])
            # get embeddings from model up to the penultimate dense layer (create embedder)
            embedder = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)  # Dense before dropout
            anchor_emb = embedder.predict(Xb[a], verbose=0)
            pos_emb = embedder.predict(Xb[p], verbose=0)
            neg_emb = embedder.predict(Xb[n], verbose=0)
            contrastive_comp = contrastive_loss_from_embeddings(anchor_emb, pos_emb, neg_emb, temperature=temp)
            # standard train step
            with tf.GradientTape() as tape:
                preds = model(Xb, training=True)
                ce = tf.keras.losses.sparse_categorical_crossentropy(yb, preds)
                l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights]) * cfg['training']['lambda_l2']
                loss = tf.reduce_mean(ce) + l2 + lambda_con * contrastive_comp
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        # validation at epoch end
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")
    # final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss/acc:", test_loss, test_acc)
    # save model and preprocessing objects
    model.save('mntd_model.h5')
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    if ohe is not None:
        joblib.dump(ohe, 'ohe.pkl')

if __name__ == '__main__':
    train_pipeline('config.yaml')
