import streamlit as st
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import plotly.express as px
import plotly.graph_objects as go

# Directories
train_dir = "/ibex/user/halimia/Forecasting-project/V3/Training/Deep_Learning/10_food_classes_all_data/train/"
test_dir = "/ibex/user/halimia/Forecasting-project/V3/Training/Deep_Learning/10_food_classes_all_data/test/"

# Setup class names
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

# Data Augmentation and Processing
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical')
test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                  target_size=(224, 224),
                                                                  batch_size=32,
                                                                  class_mode='categorical',
                                                                  shuffle=True)

# Streamlit Interface
st.title('Food Vision App')

# Load MobileNet as a base model
base_model = tf.keras.applications.mobilenet.MobileNet(include_top=False, 
                                                       weights='imagenet', 
                                                       input_shape=(224, 224, 3))

# Function to build the full model with a custom classifier
def build_full_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Modify the train_model function to train only the classifier
def train_model():
    # Build the full model
    model = build_full_model(base_model)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    
    # Training code with progress bar
    my_bar = st.progress(0)
    with st.spinner('Training in progress...'):
        for epoch in range(5):
            epoch_loss = 0
            epoch_acc = 0
            for step in range(len(train_data_augmented)):
                imgs, labels = next(train_data_augmented)
                loss, acc = model.train_on_batch(imgs, labels)
                epoch_loss += loss
                epoch_acc += acc

            epoch_loss /= len(train_data_augmented)
            epoch_acc /= len(train_data_augmented)

            test_loss, test_acc = model.evaluate(test_data, verbose=0)
            progress = (epoch + 1) / 5
            my_bar.progress(progress)
            st.write(f"Epoch {epoch+1}/{5} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    st.success('Training completed!')
    return model

# Prediction Function
def food_vision(model, image):
    if not isinstance(image, np.ndarray):
        image = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(image, size=(224, 224))
    img = img / 255.
    img = tf.expand_dims(img, axis=0)
    pred_probs = model.predict(img)[0]
    class_index = tf.argmax(pred_probs).numpy()
    class_probabilities = {class_name: float(prob) for class_name, prob in zip(class_names, pred_probs)}
    return class_names[class_index], class_probabilities

# Ask the user if they want to train a new model or use a pretrained one
choice = st.radio("Choose an option:", ["Train a new classifier", "Use the pretrained MobileNet"])

# Load the full model (MobileNet with a custom classifier on top) if the user chooses pretrained
if choice == "Use the pretrained MobileNet":
    if 'model_11' in st.session_state:
        model = tf.keras.models.load_model("saved_trained_model")
        st.session_state.model_11 = model
    else:
        model = tf.keras.models.load_model("saved_trained_model")
        st.session_state.model_11 = model

    # Debug: Print model state
    st.write(f"Model state: {model}")
    
    # Prediction section for the pretrained model
    uploaded_file = st.file_uploader("Choose an image for prediction...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("Predicting...")
        
        # Ensure model is not None before making predictions
        if model:
            label, probs = food_vision(model, uploaded_file)
            
            st.subheader(f"Prediction: {label} ({probs[label]:.2%})")
            
            sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:10])
            sorted_probs_percentage = {k: v * 100 for k, v in sorted_probs.items()}
            
            fig = px.bar(x=list(sorted_probs_percentage.values()), y=list(sorted_probs_percentage.keys()), orientation='h', labels={'x': 'Confidence (%)', 'y': 'Class'})
            fig.update_layout(
                autosize=False,
                width=800,
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
                font=dict(size=14),
                xaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=16)
                )
            )
            fig = go.Figure(fig)
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig)
        else:
            st.error("Model is not initialized. Please ensure the model is correctly loaded or trained.")

# Train a new classifier if that's the user's choice
elif choice == "Train a new classifier":
    if 'model_11' in st.session_state:
        if st.button("Re-train Classifier"):
            model = train_model()
            st.session_state.model_11 = model
        else:
            model = st.session_state.model_11
    else:
        if st.button("Train Classifier"):
            model = train_model()
            st.session_state.model_11 = model
        else:
            model = None

    # Prediction section
    if model:
        uploaded_file = st.file_uploader("Choose an image for prediction...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("Predicting...")
            label, probs = food_vision(model, uploaded_file)
            
            st.subheader(f"Prediction: {label} ({probs[label]:.2%})")
            
            sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:10])
            sorted_probs_percentage = {k: v * 100 for k, v in sorted_probs.items()}
            
            fig = px.bar(x=list(sorted_probs_percentage.values()), y=list(sorted_probs_percentage.keys()), orientation='h', labels={'x': 'Confidence (%)', 'y': 'Class'})
            fig.update_layout(
                autosize=False,
                width=800,
                height=500,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
                font=dict(size=14),
                xaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=16)
                )
            )
            fig = go.Figure(fig)
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig)
