from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def isBlack(boxes, classes, image):
    results = []
    
    for i in range(len(classes)):
        x1, y1, x2, y2 = map(int, boxes[i]) 
        block = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

        # 검정 마스크
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        mask_black = cv2.inRange(block, lower_black, upper_black)

        # 남색 마스크
        lower_navy = np.array([100, 30, 20])
        upper_navy = np.array([130, 255, 110])
        mask_navy = cv2.inRange(block, lower_navy, upper_navy)

        # 두 마스크 결합
        mask_b = cv2.bitwise_or(mask_black, mask_navy)

        lower_white = np.array([0, 0, 210])     
        upper_white = np.array([180, 30, 255]) 
        mask_w = cv2.inRange(block, lower_white, upper_white)

        # inRange()의 결과로, 255값을 가지는 pixel의 비율 확인
        ratio_b = np.count_nonzero(mask_b) / mask_b.size
        ratio_w = np.count_nonzero(mask_w) / mask_w.size
        
        if ratio_b >= ratio_w + 0.01 or ratio_b + 0.02 >= ratio_w:
            print("ok", (ratio_b, ratio_w))
            results.append((x1, y1, x2, y2))
        else:
            print("not ok", (ratio_b, ratio_w))
    return results

def draw_formation(image, player):
    centers = []

    for x1, y1, x2, y2 in player:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))
        # 점 표시
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    return image

model = YOLO("yolov8s.pt")
results = model("test.png")
image = cv2.imread("test.png")

visualization = results[0].plot()

boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()

print("Boxes:", boxes)
print("Classes:", classes)

sharp_kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1,],   
                        [-1, -1, -1]])

sharpening_image = cv2.filter2D(image, -1, sharp_kernel)

blackTeamPlayer = isBlack(boxes, classes, sharpening_image)
print(blackTeamPlayer, len(blackTeamPlayer))
formation_image = draw_formation(image, blackTeamPlayer)

# 이미지를 저장
# Jupyter notebook 환경에서 matplot 사용하니 메모리 충돌 발생
cv2.imwrite("formation.jpg", formation_image)