import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Global L2 regularizer
REG = regularizers.l2(1e-4)

def feed_forward_module(x, dim, ff_dim, dropout=0.25, name="ffn"):
    res = x
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Dense(ff_dim, activation="swish", kernel_regularizer=REG, name=f"{name}_dense1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = layers.Dense(dim, kernel_regularizer=REG, name=f"{name}_dense2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop2")(x)
    
    scaled_x = layers.Lambda(lambda t: t * 0.5, name=f"{name}_scale")(x)
    return layers.Add(name=f"{name}_add")([res, scaled_x])


def conformer_conv_module(x, dim, kernel_size=9, dropout=0.25, name="conv"):
    res = x
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Conv1D(filters=dim * 2, kernel_size=1, kernel_regularizer=REG, name=f"{name}_pw1")(x)
    x = layers.Activation("swish", name=f"{name}_glu")(x)
    x = layers.Conv1D(
        filters=dim, 
        kernel_size=kernel_size, 
        padding="same", 
        groups=dim, 
        kernel_regularizer=REG,
        name=f"{name}_dw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Conv1D(filters=dim, kernel_size=1, kernel_regularizer=REG, name=f"{name}_pw2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return layers.Add(name=f"{name}_add")([res, x])


def conformer_block(x, dim=32, num_heads=2, ff_dim=64, kernel_size=9, dropout=0.25, prefix="conf"):
    # 1. First FFN (Macaron half-step)
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, name=f"{prefix}_ffn1")
    
    # 2. Multi-Head Self-Attention with attention dropout
    res = x
    attn_norm = layers.LayerNormalization(name=f"{prefix}_attn_ln")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=dim // num_heads, 
        dropout=dropout,
        name=f"{prefix}_mha"
    )(attn_norm, attn_norm)
    attn_out = layers.Dropout(dropout, name=f"{prefix}_attn_drop")(attn_out)
    x = layers.Add(name=f"{prefix}_attn_add")([res, attn_out])
    
    # 3. Convolution Module
    x = conformer_conv_module(x, dim, kernel_size=kernel_size, dropout=dropout, name=f"{prefix}_conv")
    
    # 4. Second FFN (Macaron half-step)
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, name=f"{prefix}_ffn2")
    
    # 5. Final Normalization
    return layers.LayerNormalization(name=f"{prefix}_final_ln")(x)


# --- Architecture ---
inputs = keras.Input(shape=(3960,))
x = layers.Reshape((99, 40, 1))(inputs)

# Front-end Augmentations (Active only during training)
x = layers.GaussianNoise(0.05)(x)
x = layers.SpatialDropout2D(0.1)(x)  # Drop entire frequency/time bands

# 1. Light 2D Conv Subsampling
x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_regularizer=REG)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("swish")(x)

# Shape becomes (50, 20, 16) -> (50, 320)
x = layers.Reshape((50, 320))(x)

# 2. Linear projection with dropout
x = layers.Dense(32, kernel_regularizer=REG)(x)
x = layers.Dropout(0.2)(x)

# 3. Regularized Conformer Block (dropout increased to 0.25)
x = conformer_block(x, dim=32, num_heads=2, ff_dim=64, kernel_size=9, dropout=0.25)

# 4. Head with classification dropout
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(classes, activation="softmax", kernel_regularizer=REG)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# --- Training & Callbacks ---
EPOCHS = args.epochs or 30  # Allow more epochs since learning rate will step down
LEARNING_RATE = args.learning_rate or 0.003  # Lower initial LR for transformer stability
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE * 4)

prefetch_policy = 1 if ENSURE_DETERMINISM else tf.data.AUTOTUNE
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)

# Add anti-overfitting callbacks
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))
callbacks.append(EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5))

model.compile(
    optimizer=Nadam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_dataset, 
    validation_data=validation_dataset, 
    epochs=EPOCHS, 
    verbose=2, 
    callbacks=callbacks, 
    class_weight=ei_tensorflow.training.get_class_weights(Y_train)
)

