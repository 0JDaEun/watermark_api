import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 데이터 경로 설정
original_dir = 'data/input/class1'  # 클래스 하위 디렉토리로 변경
watermarked_dir = 'data/results/class1'  # 클래스 하위 디렉토리로 변경

# 이미지 크기 및 배치 크기 설정
img_height, img_width = 224, 224
batch_size = 32

# 데이터셋 생성 (image_dataset_from_directory 사용)
def create_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        validation_split=0.2,
        subset="both",
        seed=123
    )

# 원본 및 워터마크 데이터셋 로드
original_dataset = create_dataset(original_dir)
watermarked_dataset = create_dataset(watermarked_dir)

# 데이터셋 확인
print(f"Original training samples: {len(original_dataset.file_paths)}")
print(f"Watermarked training samples: {len(watermarked_dataset.file_paths)}")

# CNN 모델 정의
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 원본 데이터로 모델 학습 및 평가
original_model = create_model((img_height, img_width, 3), original_dataset.num_classes)
original_history = original_model.fit(
    original_dataset,
    epochs=10,
    validation_data=watermarked_dataset,
)

# 결과 시각화 및 출력
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(original_history)

print("Original Model - Final Accuracy:", original_history.history['accuracy'][-1])
print("Original Model - Final Validation Accuracy:", original_history.history['val_accuracy'][-1])
