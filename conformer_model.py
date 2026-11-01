import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import categorical_crossentropy

def feed_forward_module(x, dim, ff_dim, dropout=0.1, name="ffn"):
    res = x
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Dense(ff_dim, activation="swish", name=f"{name}_dense1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = layers.Dense(dim, name=f"{name}_dense2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop2")(x)
    
    # Scale by 0.5 using a Lambda layer or Rescaling to maintain valid 3D tensor shape
    scaled_x = layers.Lambda(lambda t: t * 0.5, name=f"{name}_scale")(x)
    return layers.Add(name=f"{name}_add")([res, scaled_x])

def conformer_conv_module(x, dim, kernel_size=9, dropout=0.1, name="conv"):
    res = x
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Conv1D(filters=dim * 2, kernel_size=1, name=f"{name}_pw1")(x)
    x = layers.Activation("swish", name=f"{name}_glu")(x)
    x = layers.Conv1D(
        filters=dim, 
        kernel_size=kernel_size, 
        padding="same", 
        groups=dim, 
        name=f"{name}_dw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Conv1D(filters=dim, kernel_size=1, name=f"{name}_pw2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return layers.Add(name=f"{name}_add")([res, x])


def conformer_block(x, dim=32, num_heads=2, ff_dim=64, kernel_size=9, dropout=0.1, prefix="conf"):
    # 1. First FFN (Macaron half-step)
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, name=f"{prefix}_ffn1")
    
    # 2. Multi-Head Self-Attention
    res = x
    attn_norm = layers.LayerNormalization(name=f"{prefix}_attn_ln")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=dim // num_heads, 
        name=f"{prefix}_mha"
    )(attn_norm, attn_norm)
    x = layers.Add(name=f"{prefix}_attn_add")([res, attn_out])
    
    # 3. Convolution Module
    x = conformer_conv_module(x, dim, kernel_size=kernel_size, dropout=dropout, name=f"{prefix}_conv")
    
    # 4. Second FFN (Macaron half-step)
    x = feed_forward_module(x, dim, ff_dim, dropout=dropout, name=f"{prefix}_ffn2")
    
    # 5. Final Normalization
    return layers.LayerNormalization(name=f"{prefix}_final_ln")(x)
    
# Accept the flat feature vector from Edge Impulse
inputs = keras.Input(shape=(3960,))
# Reshape the flat vector back into the 2D time-frequency grid + channel format
x = layers.Reshape((99, 40, 1))(inputs)
# 1. Light 2D Conv Subsampling to shrink time and features (Down to ~1.2K params)
x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("swish")(x)
# Shape becomes (50, 20, 16) -> Collapse spatial axes: 20 * 16 = 320 features
x = layers.Reshape((50, 320))(x)
# 2. Linear projection to tiny model dimension (dim=32) (~10.3K params)
x = layers.Dense(32)(x)
# 3. Tiny Conformer Block with dim=32, ff_dim=64 (~26.5K params)
x = conformer_block(x, dim=32, num_heads=2, ff_dim=64, kernel_size=9)
# 4. Output Classifier Head (~330 params for 10 classes)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
EPOCHS = args.epochs or 30
LEARNING_RATE = args.learning_rate or 0.01
# If True, non-deterministic functions (e.g. shuffling batches) are not used.
# This is False by default.
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32
if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*4)
prefetch_policy = 1 if ENSURE_DETERMINISM else tf.data.AUTOTUNE
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))
model.compile(optimizer=Nadam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks, class_weight=ei_tensorflow.training.get_class_weights(Y_train))



