import cv2
import numpy as np
import tensorflow as tf


def get_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(model, base_model, image, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(base_model)

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(base_model.input, last_conv_layer.output)

    head_layers = model.layers[model.layers.index(base_model) + 1:]

    image_batch = tf.expand_dims(image, axis=0)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(preprocessed)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in head_layers:
            x = layer(x)

        top_class = tf.argmax(x[0])
        loss = x[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(image, heatmap, image_size, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, image_size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)