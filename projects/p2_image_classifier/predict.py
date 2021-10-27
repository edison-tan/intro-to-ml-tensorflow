import argparse
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import json
from PIL import Image
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


parser = argparse.ArgumentParser(description ='load model for testing')
parser.add_argument('dir', type=str, default = './test_images/hard-leaved_pocket_orchid.jpg', help = 'Test image')
parser.add_argument('--top_k', dest = 'top_k', type=int, default = '5', help = 'Top K probabilites and classes')
parser.add_argument('--category_names', dest = 'class_names', help = 'Classes from data set')
parser.add_argument('model', type=str, help='nn model for predicting')


args = parser.parse_args()

model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer} )

model.summary()

print(args.dir)
print(args.class_names)
print(args.top_k)

#label mapping
with open(args.class_names, 'r') as f:
    class_names = json.load(f)

# TODO: Create the process_image function
def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (224, 224))/255.0
    return image
    
# TODO: Create the predict function
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    top_classes = [class_names[str(value+1)] for value in top_indices.numpy()[0]]
    return top_values.numpy()[0], top_classes
    
im = Image.open(args.dir)
test_image = np.asarray(im)
processed_test_image = process_image(test_image)
probs, classes = predict(args.dir, model, args.top_k)
print('Probabilities:',probs)
print('Classes:', classes)