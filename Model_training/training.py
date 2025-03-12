import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 15

# Define threshold for underrepresented classes
UNDERREP_THRESHOLD = 1500

def create_model():
    """Create a model optimized for imbalanced data"""
    try:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:  # Fine-tune more layers
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            
            # Increase network capacity for better feature learning
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator to handle class imbalance"""
    def __init__(self, data_dir, batch_size, target_size, subset='training', validation_split=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.subset = subset
        self.validation_split = validation_split
        
        # Initialize data structures
        self.classes = sorted(os.listdir(data_dir))
        self.num_classes = len(self.classes)
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.prepare_data()
        
        # Create augmentation generators
        self.strong_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.normal_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
    def prepare_data(self):
        """Prepare data with balanced sampling"""
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            images = os.listdir(class_path)
            
            # Split indices for training/validation
            split_idx = int(len(images) * (1 - self.validation_split))
            if self.subset == 'training':
                images = images[:split_idx]
            else:
                images = images[split_idx:]
            
            # Add image paths and labels
            for img in images:
                self.image_paths.append(os.path.join(class_path, img))
                self.labels.append(self.class_indices[class_name])
        
        self.indices = np.arange(len(self.image_paths))
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_x = np.zeros((self.batch_size,) + self.target_size + (3,))
        batch_y = np.zeros((self.batch_size, self.num_classes))
        
        # Fill batch
        for i, idx in enumerate(batch_indices):
            # Load and preprocess image
            img_path = self.image_paths[idx]
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=self.target_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            
            # Apply appropriate augmentation based on class size
            class_name = os.path.basename(os.path.dirname(img_path))
            class_size = len(os.listdir(os.path.join(self.data_dir, class_name)))
            
            if class_size < UNDERREP_THRESHOLD:
                img = self.strong_aug.random_transform(img)
            else:
                img = self.normal_aug.random_transform(img)
            
            batch_x[i] = img / 255.0
            batch_y[i][self.labels[idx]] = 1
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def prepare_generators(data_dir):
    """Prepare balanced data generators"""
    try:
        # Create balanced generators
        train_gen = BalancedDataGenerator(
            data_dir=data_dir,
            batch_size=BATCH_SIZE,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            subset='training'
        )
        
        val_gen = BalancedDataGenerator(
            data_dir=data_dir,
            batch_size=BATCH_SIZE,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            subset='validation'
        )
        
        return train_gen, val_gen
        
    except Exception as e:
        print(f"Error preparing generators: {str(e)}")
        raise

def train_model(model, train_generator, validation_generator):
    """Train the model with custom learning rate and callbacks"""
    try:
        # Compile model with fixed learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        return history
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def plot_training_history(history):
    """Plot training history"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'])
        
        # Plot loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'])
        
        plt.show()
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

def evaluate_model(model, test_generator):
    """Evaluate the model and print classification report"""
    try:
        # Calculate the number of steps needed to cover all samples
        steps = len(test_generator)
        all_predictions = []
        all_true_labels = []
        
        # Collect predictions and true labels batch by batch
        for i in range(steps):
            x_batch, y_batch = test_generator[i]
            batch_predictions = model.predict(x_batch, verbose=0)
            all_predictions.extend(np.argmax(batch_predictions, axis=1))
            all_true_labels.extend(np.argmax(y_batch, axis=1))
        
        # Convert lists to numpy arrays
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        
        # Print classification report
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(
            all_true_labels, 
            all_predictions,
            target_names=test_generator.classes
        ))
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        raise
        


print("Starting the training process...")

# Set data directory
data_dir = '/kaggle/input/bangladeshi-crops-disease-dataset/CropDisease/Crop___DIsease'

# Prepare generators
train_generator, validation_generator = prepare_generators(data_dir)

# Create and train model
model = create_model()
print("\nModel created. Starting training...")

history = train_model(model, train_generator, validation_generator)

# Plot and evaluate
plot_training_history(history)
evaluate_model(model, validation_generator)



def save_model(model, save_dir='../model'):
    """Save the model and its weights"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_dir, 'crop_disease_model.keras')
        model.save(model_path)
        print(f"\nModel saved successfully at: {model_path}")
        
        # Save the model weights with correct extension
        weights_path = os.path.join(save_dir, 'model_weights.weights.h5')
        model.save_weights(weights_path)
        print(f"Model weights saved separately at: {weights_path}")
        return model_path, weights_path
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

# Function to load the saved model
def load_model(model_path):
    """Load a saved model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"\nModel loaded successfully from: {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Save the model after training
# model_path, weights_path = save_model(model)

model = load_model("../model/crop_disease_model.keras")

