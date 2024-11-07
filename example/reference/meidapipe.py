import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 读取输入图像
image_path = "example/reference/0001.png"
output_path = "example/reference/pose_estimation_result.png"  # 保存结果的路径
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Please check the file path.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行姿态估计
    results = pose.process(image_rgb)

    # 检查是否检测到姿态
    if results.pose_landmarks:
        # 在图像上绘制姿态
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

        # 输出关节点坐标
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            print(f"Landmark {idx}: ({cx}, {cy})")

        # 保存结果图像
        cv2.imwrite(output_path, image)
        print(f"Pose estimation result saved at {output_path}")
    else:
        print("No pose detected.")

# 释放资源
pose.close()
