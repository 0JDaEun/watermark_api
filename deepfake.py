import cv2
import numpy as np
import os

# Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 랜드마크 추출 함수 (Haar Cascade 사용)
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]  # 첫 번째 얼굴만 사용
    return (x, y, w, h)

# 얼굴 합성 함수
def face_swap(source_image, target_image):
    src_face = detect_face(source_image)
    tgt_face = detect_face(target_image)

    if src_face is None or tgt_face is None:
        print("얼굴을 감지할 수 없습니다.")
        return target_image
    
    sx, sy, sw, sh = src_face
    tx, ty, tw, th = tgt_face

    # 소스 얼굴 영역 추출
    src_face_roi = source_image[sy:sy+sh, sx:sx+sw]
    
    # 타겟 얼굴 영역 추출
    tgt_face_roi = target_image[ty:ty+th, tx:tx+tw]
    
    # 소스 얼굴 영역 크기를 타겟 얼굴 영역 크기로 조정
    src_face_resized = cv2.resize(src_face_roi, (tw, th))
    
    # 타겟 얼굴 영역에 소스 얼굴 합성
    target_image[ty:ty+th, tx:tx+tw] = src_face_resized
    
    return target_image

# 메인 함수
if __name__ == "__main__":
    # 이미지 경로 설정
    source_path = "data/santa.jpg"
    target_path = "data/input/sample1.jpg"

    # 이미지 로드
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    if source_image is None or target_image is None:
        print("이미지를 로드할 수 없습니다.")
        exit()

    # 얼굴 합성 수행
    output_image = face_swap(source_image, target_image)

    # 결과 저장 및 시각화
    output_path = "data/output.jpg"
    cv2.imwrite(output_path, output_image)
    
    print(f"합성된 이미지를 저장했습니다: {output_path}")

    # 결과 출력
    cv2.imshow("Result", output_image)
    cv2.waitKey(0)
