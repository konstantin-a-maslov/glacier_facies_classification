import tensorflow as tf


def build_cnn_model(
    input_features=6, output_shape=8, 
    activation=tf.keras.layers.ELU(),
    depth1=3, filters1=64,
    pool_size=(3, 3),
    depth2=3, filters2=64,
    depth3=2, units3=32,
    dropout_rate=0.5, return_latent=False,
    name="cnn_model",
    **kwargs
):
    input_shape = (None, None, input_features)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # scale 1
    x = inputs
    for _ in range(depth1):
        x = tf.keras.layers.Conv2D(
            filters=filters1,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)
    ccell1 = tf.keras.layers.Lambda(extract_center_cell)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
        
    # scale 2    
    for _ in range(depth2):
        x = tf.keras.layers.Conv2D(
            filters=filters2,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)
    ccell2 = tf.keras.layers.Lambda(extract_center_cell)(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = tf.keras.layers.Concatenate(axis=-1)([ccell1, ccell2, x])
    latent = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    # classification head
    x = latent
    for _ in range(depth3):
        x = tf.keras.layers.Dense(units=units3)(x)
        x = activation(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(
        units=output_shape,
        activation=tf.keras.activations.softmax,
    )(x)
    if return_latent:
        outputs = [outputs, latent]
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
    

def extract_center_cell(x):
    shape = tf.shape(x)
    ch = shape[1] // 2
    cw = shape[2] // 2
    ccell = x[:, ch, cw, :]
    return ccell
