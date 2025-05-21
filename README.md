data_preprocessing.py

import pandas as pd

def preprocess_data(df):
    """
    Performs basic data preprocessing steps: handling missing values and duplicates.
    Args:
        df (pd.DataFrame): Input DataFrame with 'text' and 'emotion' columns.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Handle missing values by dropping rows with any NaN
    df_cleaned = df.dropna()
    print(f"Number of rows after handling missing values: {len(df_cleaned)}")

    # Handle duplicates by dropping all duplicate rows
    df_no_duplicates = df_cleaned.drop_duplicates()
    print(f"Number of rows after handling duplicates: {len(df_no_duplicates)}")

    return df_no_duplicates

if __name__ == '__main__':
    # Example usage (assuming you have a CSV file named 'emotion_data.csv')
    try:
        df = pd.read_csv('emotion_data.csv')
        if 'text' not in df.columns or 'emotion' not in df.columns:
            print("Error: DataFrame must contain 'text' and 'emotion' columns.")
        else:
            df_processed = preprocess_data(df.copy())
            print("\nFirst 5 rows of the processed data:")
            print(df_processed.head())
            # You can save the processed data if needed
            # df_processed.to_csv('processed_emotion_data.csv', index=False)
    except FileNotFoundError:
        print("Error: emotion_data.csv not found. Please create this file with 'text' and 'emotion' columns for testing.")
# model_building.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import joblib

def train_logistic_regression(X_train, y_train, label_encoder):
    """
    Trains a Logistic Regression model.
    Args:
        X_train: Training features.
        y_train: Training labels.
        label_encoder: Fitted LabelEncoder.
    Returns:
        tuple: Trained Logistic Regression model and classification report.
    """
    logistic_model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42, max_iter=1000)
    logistic_model.fit(X_train, y_train)
    return logistic_model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates a trained model.
    Args:
        model: Trained classification model.
        X_test: Test features.
        y_test: Test labels.
        label_encoder: Fitted LabelEncoder.
    Returns:
        tuple: Accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    return accuracy, report

def train_distilbert(train_texts, train_labels, val_texts, val_labels, label_encoder, model_name='distilbert-base-uncased', max_length=128, batch_size=16, epochs=3):
    """
    Trains a DistilBERT model.
    Args:
        train_texts (list): List of training text.
        train_labels (list): List of training labels (encoded).
        val_texts (list): List of validation text.
        val_labels (list): List of validation labels (encoded).
        label_encoder: Fitted LabelEncoder.
        model_name (str): Name of the pre-trained DistilBERT model.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
    Returns:
        TFDistilBertForSequenceClassification: Trained DistilBERT model.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    num_classes = len(label_encoder.classes_)

    def tokenize_function(examples):
        return tokenizer(examples, truncation=True, padding='max_length', max_length=max_length)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(batch_size)

    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    return model, tokenizer

def evaluate_distilbert(model, tokenizer, test_texts, test_labels, label_encoder, max_length=128, batch_size=16):
    """
    Evaluates a trained DistilBERT model.
    Args:
        model (TFDistilBertForSequenceClassification): Trained DistilBERT model.
        tokenizer (DistilBertTokenizer): Tokenizer.
        test_texts (list): List of test text.
        test_labels (list): List of test labels (encoded).
        label_encoder: Fitted LabelEncoder.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for evaluation.
    Returns:
        tuple: Accuracy and classification report.
    """
    def tokenize_function(examples):
        return tokenizer(examples, truncation=True, padding='max_length', max_length=max_length)

    test_encodings = tokenize_function(test_texts)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(batch_size)

    evaluation = model.evaluate(test_dataset)
    print(f"DistilBERT Loss: {evaluation[0]:.4f}")
    print(f"DistilBERT Accuracy: {evaluation[1]:.4f}")

    predictions = model.predict(test_dataset)
    predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
    true_labels = tf.concat([labels for _, labels in test_dataset], axis=0).numpy()
    report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_, zero_division=0)
    return evaluation[1], report

if __name__ == '__main__':
    # Example usage (assuming you have processed and featurized data)
    try:
        df_processed = pd.read_csv('processed_emotion_data.csv')
        if 'text' not in df_processed.columns or 'emotion' not in df_processed.columns:
            print("Error: DataFrame must contain 'text' and 'emotion' columns.")
        else:
            from feature_engineering import vectorize_text, encode_labels
            X_tfidf, tfidf_vectorizer = vectorize_text(df_processed.copy())
            y_encoded, label_encoder = encode_labels(df_processed.copy())

            X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded = train_test_split(
                X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            print("\nTraining Logistic Regression...")
            logistic_model = train_logistic_regression(X_train_tfidf, y_train_encoded, label_encoder)
            lr_accuracy, lr_report = evaluate_model(logistic_model, X_test_tfidf, y_test_encoded, label_encoder)
            print("\nLogistic Regression Results:")
            print(f"Accuracy: {lr_accuracy:.4f}")
            print("Classification Report:\n", lr_report)

            # Prepare data for DistilBERT
            train_texts = df_processed['text'].tolist()
            labels = df_processed['emotion'].tolist()
            y_encoded_hf, label_encoder_hf = encode_labels(pd.DataFrame({'emotion': labels}))
            train_texts_hf, test_texts_hf, train_labels_hf, test_labels_hf = train_test_split(
                train_texts, y_encoded_hf, test_size=0.2, random_state=42, stratify=y_encoded_hf
            )
            train_texts_hf, val_texts_hf, train_labels_hf, val_labels_hf = train_test_split(
                train_texts_hf, train_labels_hf, test_size=0.1, random_state=42, stratify=train_labels_hf
            )

            print("\nTraining DistilBERT...")
            distilbert_model, distilbert_tokenizer = train_distilbert(
                train_texts_hf, train_labels_hf, val_texts_hf, val_labels_hf, label_encoder_hf
            )
            db_accuracy, db_report = evaluate_distilbert(
                distilbert_model, distilbert_tokenizer, test_texts_hf, test_labels_hf, label_encoder_hf
            )
            print("\nDistilBERT Results:")
            print(f"Accuracy: {db_accuracy:.4f}")
            print("Classification Report:\n", db_report)

            # Save the trained DistilBERT model and tokenizer
            distilbert_model.save_pretrained("emotion_model")
            distilbert_tokenizer.save_pretrained("emotion_tokenizer")
            joblib.dump(label_encoder_hf, 'distilbert_label_encoder.joblib')

    except FileNotFoundError:
        print("Error: processed_emotion_data.csv not found. Please run data_preprocessing.py first.")
    except ImportError as e:
        print(f"Error: Missing libraries. Please install them (e.g., pip install scikit-learn transformers tensorflow). Details: {e}") 
