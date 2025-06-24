import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import lightgbm as lgb
from movenet_detector import MoveNetDetector

def extract_keypoints_to_csv(dataset_path, output_csv_file):
    print(f"Starting keypoint extraction from: {dataset_path}")
    detector = MoveNetDetector()
    keypoints_data = []
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found classes: {classes}")
    
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    print(f"Class mapping: {class_to_idx}")
    
    for class_name in classes:
        print(f"Processing class: {class_name}")
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total_images = len(image_files)
        print(f"  Found {total_images} images in {class_name}")
        
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(class_path, image_file)
            try:
                keypoints_features = detector.extract_features(image_path)
                
                if keypoints_features is not None:
                    row_data = {
                        'filename': image_file,
                        'class': class_name,
                        'class_idx': class_to_idx[class_name]
                    }
                    
                    for i in range(len(keypoints_features)):
                        row_data[f'feature_{i}'] = float(keypoints_features[i])
                    
                    keypoints_data.append(row_data)
                else:
                    print(f"  Warning: Could not extract keypoints from {image_file}")
            except Exception as e:
                print(f"  Error processing {image_file}: {str(e)}")
            
            if (idx + 1) % 20 == 0:
                print(f"  Extracted {idx + 1}/{total_images} images")
                
    df = pd.DataFrame(keypoints_data)
    df.to_csv(output_csv_file, index=False)
    print(f"Saved {len(keypoints_data)} samples to {output_csv_file}")
    return df

def train_pose_classifier(csv_file, model_output_file="pose_classifier.pkl", scaler_output_file="pose_scaler.pkl"):
    print("Loading data...")
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_columns].values
    y = df['class_idx'].values
    class_names = df['class'].unique()
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    params = {
        'objective': 'multiclass',
        'num_class': len(class_names),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    print("Training LightGBM model...")
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    y_pred_proba = model.predict(X_test_scaled, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"LightGBM Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open(model_output_file, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_output_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    class_mapping = {
        'class_names': list(class_names),
        'class_to_idx': {name: idx for idx, name in enumerate(class_names)}
    }
    with open('class_mapping.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"Model saved to: {model_output_file}")
    print(f"Scaler saved to: {scaler_output_file}")
    print(f"Class mapping saved to: class_mapping.pkl")
    
    return model, scaler, class_mapping

def main():
    dataset_path = "DATASET/TRAIN"
    csv_file = "train_keypoints.csv"
    extract_keypoints_to_csv(dataset_path, csv_file)
    train_pose_classifier(csv_file)

if __name__ == "__main__":
    main()
