import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import categorical_crossentropy

INPUT_SHAPE = (99, 40, 1)

model = Sequential()
model.add(layers.InputLayer(
    input_shape=X_train[0].shape, name='x_input'))
model.add(layers.Reshape(INPUT_SHAPE))

# Layer 1: First Convolutional Block
model.add(layers.Conv2D(
        filters=8, 
        kernel_size=(3, 3), 
        padding='same', 
        name="conv2d_1"
    ))
model.add(layers.BatchNormalization(name="bn_1"))
model.add(layers.ReLU(name="relu_1"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))

# Layer 2: Second Convolutional Block
model.add(layers.Conv2D(
     filters=16, 
     kernel_size=(3, 3), 
     padding='same',
     name="conv2d_2"
))
model.add(layers.BatchNormalization(name="bn_2"))
model.add(layers.ReLU(name="relu_2"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"))

model.add(layers.Dropout(0.5))
# Global Pooling to reduce parameters for Microcontrollers
model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))

model.add(layers.Dense(classes, activation='softmax'))



EPOCHS = args.epochs or 100
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
