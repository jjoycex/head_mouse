import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import categorical_crossentropy

def feed_forward_module(x, dim, ff_dim, dropout=0.1, scale=0.5, name="ffn"):
    res = x
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Dense(ff_dim, activation="swish", name=f"{name}_dense1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = layers.Dense(dim, name=f"{name}_dense2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop2")(x)
    
    if scale != 1.0:
        x = layers.Lambda(lambda t: t * scale, name=f"{name}_scale")(x)
        
    return layers.Add(name=f"{name}_add")([res, x])

def attention_module(x, dim, num_heads=2, dropout=0.1, name="attn"):
    res = x
    attn_norm = layers.LayerNormalization(name=f"{name}_ln")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=dim // num_heads,
        dropout=dropout,
        name=f"{name}_mha"
    )(attn_norm, attn_norm)
    attn_out = layers.Dropout(dropout, name=f"{name}_drop")(attn_out)
    return layers.Add(name=f"{name}_add")([res, attn_out])

def dual_attention_block(x, dim=32, num_heads=2, ff_dim=64, dropout=0.1, prefix="block"):
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, scale=0.5, name=f"{prefix}_ffn1")
    x = attention_module(x, dim, num_heads=num_heads, dropout=dropout, name=f"{prefix}_attn1")
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, scale=1.0, name=f"{prefix}_ffn_mid")
    x = attention_module(x, dim, num_heads=num_heads, dropout=dropout, name=f"{prefix}_attn2")
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, scale=0.5, name=f"{prefix}_ffn2")
    return layers.LayerNormalization(name=f"{prefix}_final_ln")(x)

# ---------------------------------------------------------------
# Model Architecture Construction
# ---------------------------------------------------------------

# Accept flat feature vector from Edge Impulse
inputs = keras.Input(shape=(3960,))

# Reshape directly into 2D time-frequency sequence (99 time steps, 40 features)
x = layers.Reshape((99, 40))(inputs)

# 1. Linear Projection to hidden dimension (32)
x = layers.Dense(32, name="linear_projection")(x)

# 2. Transformer Block
x = dual_attention_block(x, dim=32, num_heads=2, ff_dim=64, dropout=0.1, prefix="dual_attn")

# 3. Output Classifier Head
x = layers.GlobalAveragePooling1D(name="gap1d")(x)
outputs = layers.Dense(classes, activation="softmax", name="outputs")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ---------------------------------------------------------------
# Training Setup
# ---------------------------------------------------------------

EPOCHS = args.epochs or 30
LEARNING_RATE = args.learning_rate or 0.001

ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE * 4)

prefetch_policy = 1 if ENSURE_DETERMINISM else tf.data.AUTOTUNE
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)

callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

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



