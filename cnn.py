# ==============================================================================
#
#  FINGERPRINT VERIFICATION USING A SIAMESE NEURAL NETWORK
#  Complete, end-to-end script for training and evaluation.
#
# ==============================================================================

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# --- Configuration Section ---
# You can change these parameters

# Directory Paths
# Assumes the script is in /home/prayansh-chhablani/ and the data is in archive/NISTDB4_RAW
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'archive/NISTDB4_RAW')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_set')
VAL_DIR = os.path.join(DATA_DIR, 'val_set')
TEST_DIR = os.path.join(DATA_DIR, 'test_set')
SAVED_MODEL_PATH = os.path.join(BASE_DIR, 'fingerprint_siamese_model.h5')

# Model & Image Parameters
IMG_SHAPE = (105, 105) # Target image size
BATCH_SIZE = 32
EPOCHS = 15

# ==============================================================================
# STEP 1: DATA PREPARATION
# ==============================================================================

def generate_pairs(directory):
    """
    Generates genuine and impostor image pairs from a given directory.
    Assumes NISTDB4 file naming convention: 'f_XXXX_YY.png' and 's_XXXX_YY.png'
    where XXXX is the unique finger ID.
    """
    print(f"Generating pairs from directory: {directory}")
    pairs = []
    labels = []
    
    # 1. Group images by finger ID
    images_by_finger = {}
    for filename in os.listdir(directory):
        if not (filename.endswith('.png') or filename.endswith('.wsq')):
            continue
        
        # Extract finger ID from filenames like 'f_0001_01.png'
        try:
            finger_id = filename.split('_')[1]
        except IndexError:
            print(f"Warning: Could not parse finger ID from filename {filename}. Skipping.")
            continue
            
        if finger_id not in images_by_finger:
            images_by_finger[finger_id] = []
        images_by_finger[finger_id].append(os.path.join(directory, filename))

    finger_ids = list(images_by_finger.keys())
    
    # 2. Create genuine pairs (label = 1)
    for finger_id in finger_ids:
        impressions = images_by_finger[finger_id]
        if len(impressions) >= 2:
            # Take the first two impressions as a genuine pair
            pairs.append([impressions[0], impressions[1]])
            labels.append(1)

    # 3. Create impostor pairs (label = 0)
    num_genuine_pairs = len(pairs)
    for i in range(num_genuine_pairs):
        # Pick two different random fingers
        id1, id2 = random.sample(finger_ids, 2)
        
        # Pick one random impression from each finger
        img_path1 = random.choice(images_by_finger[id1])
        img_path2 = random.choice(images_by_finger[id2])
        
        pairs.append([img_path1, img_path2])
        labels.append(0)
        
    print(f"Generated {len(pairs)} total pairs ({num_genuine_pairs} genuine, {num_genuine_pairs} impostor).")
    return np.array(pairs), np.array(labels)

def load_and_preprocess_image(path):
    """Loads and preprocesses a single image file."""
    try:
        # Open image, convert to grayscale, and resize
        image = Image.open(path).convert('L').resize(IMG_SHAPE)
        image = np.array(image, dtype='float32')
        # Normalize pixel values to [0, 1]
        image /= 255.0
        # Add channel dimension for Keras
        image = np.expand_dims(image, axis=-1)
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_image_data_from_pairs(pairs_list):
    """Loads image data for left and right pairs into numpy arrays."""
    left_images, right_images = [], []
    for i, pair in enumerate(pairs_list):
        if i % 500 == 0 and i > 0:
            print(f"  ... loaded {i}/{len(pairs_list)} image pairs")
        left_images.append(load_and_preprocess_image(pair[0]))
        right_images.append(load_and_preprocess_image(pair[1]))
    return np.array(left_images), np.array(right_images)

# ==============================================================================
# STEP 2: MODEL ARCHITECTURE
# ==============================================================================

def create_base_network(input_shape):
    """
    Creates the base CNN model that transforms an image into a feature vector.
    This is the "head" of the siamese network that is shared between the two inputs.
    """
    input = Input(shape=input_shape)
    x = Conv2D(64, (10, 10), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4), activation='relu', kernel_initializer='he_uniform')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)  # The feature vector (embedding)
    
    return Model(input, x)

def euclidean_distance(vectors):
    """Calculates the euclidean distance between two output vectors."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# ==============================================================================
# STEP 3: LOSS FUNCTION
# ==============================================================================

def contrastive_loss(y_true, y_pred):
    """
    Custom contrastive loss function.
    It pushes genuine pairs (y_true=1) closer together (minimizing y_pred)
    and impostor pairs (y_true=0) farther apart.
    """
    y_true = tf.cast(y_true, tf.float32)
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# ==============================================================================
# STEP 4: TRAINING AND EVALUATION
# ==============================================================================

def main():
    """Main function to run the training pipeline."""
    
    # --- Load and Prepare Data ---
    train_pairs, train_labels = generate_pairs(TRAIN_DIR)
    val_pairs, val_labels = generate_pairs(VAL_DIR)
    
    print("\nLoading training images into memory...")
    X_train_left, X_train_right = load_image_data_from_pairs(train_pairs)
    print("Loading validation images into memory...")
    X_val_left, X_val_right = load_image_data_from_pairs(val_pairs)

    # --- Build the Model ---
    input_shape = (*IMG_SHAPE, 1) # e.g. (105, 105, 1)
    base_network = create_base_network(input_shape)
    
    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)
    
    vector_left = base_network(input_left)
    vector_right = base_network(input_right)
    
    distance = Lambda(euclidean_distance)([vector_left, vector_right])
    
    siamese_model = Model(inputs=[input_left, input_right], outputs=distance)
    
    print("\n--- Siamese Model Architecture ---")
    siamese_model.summary()
    
    # --- Compile and Train ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00006) # Use a low learning rate
    siamese_model.compile(loss=contrastive_loss, optimizer=optimizer)

    print("\n--- Starting Model Training ---")
    history = siamese_model.fit(
        [X_train_left, X_train_right], 
        train_labels,
        validation_data=([X_val_left, X_val_right], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # --- Save the Trained Model ---
    print(f"\nTraining complete. Saving model to {SAVED_MODEL_PATH}")
    siamese_model.save(SAVED_MODEL_PATH)
    print("Model saved successfully.")

# ==============================================================================
# STEP 5: INFERENCE (USING THE TRAINED MODEL)
# ==============================================================================
    
def verify_fingerprints(img_path1, img_path2, model, threshold=0.5):
    """
    Takes two image paths and a trained model to predict if they are a match.
    """
    # Load and preprocess the two images
    img1 = load_and_preprocess_image(img_path1)
    img2 = load_and_preprocess_image(img_path2)
    
    if img1 is None or img2 is None:
        print("Could not process one or both images.")
        return

    # The model expects a batch, so we add a batch dimension
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Make a prediction (the output is the distance)
    distance = model.predict([img1_batch, img2_batch])[0][0]
    
    print(f"\n--- Verification ---")
    print(f"Image 1: {os.path.basename(img_path1)}")
    print(f"Image 2: {os.path.basename(img_path2)}")
    print(f"Calculated Distance: {distance:.4f}")
    print(f"Threshold: {threshold}")
    
    if distance < threshold:
        print("Result: MATCH")
    else:
        print("Result: NO MATCH")
    return distance

def test_model_on_random_pair():
    """Loads the saved model and tests it on a random pair from the test set."""
    if not os.path.exists(SAVED_MODEL_PATH):
        print("Model file not found. Please train the model first.")
        return

    print("\n--- Loading saved model for a test verification ---")
    # When loading a model with custom objects, they must be specified.
    custom_objects = {'contrastive_loss': contrastive_loss}
    loaded_model = tf.keras.models.load_model(SAVED_MODEL_PATH, custom_objects=custom_objects)

    test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)]
    if len(test_files) < 2:
        print("Not enough files in the test set to perform verification.")
        return

    # Select two random files from the test set to compare
    img1_path, img2_path = random.sample(test_files, 2)
    
    verify_fingerprints(img1_path, img2_path, loaded_model, threshold=0.5)


if __name__ == '__main__':
    # Run the main training pipeline
    main()
    
    # After training, run a test verification on a random pair from the test set
    test_model_on_random_pair()