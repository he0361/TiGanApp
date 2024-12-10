#此文件暂时不用，执行HttpRequest文件即可
import cv2
import numpy as np
import mediapipe as mp
import PoseTrackingModule as ptm

frame_count = 0  # 初始化帧计数器
output_frequency = 5  # 设置输出频率，输出每5帧的角度
controller = ptm.LobotServoController()  # 创建控制器对象
#卡尔曼滤波
# #卡尔曼滤波 滤过误差值
# class KalmanFilter:
#     def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
#         self.kalman = cv2.KalmanFilter(4, 2)
#         self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#         self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#         self.kalman.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
#         self.kalman.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
#
#     def correct(self, x, y):
#         measurement = np.array([[np.float32(x)], [np.float32(y)]])
#         return self.kalman.correct(measurement)
#
#     def predict(self):
#         return self.kalman.predict()
#
# kalman_filters = {i: KalmanFilter() for i in range(33)}
# def smooth_with_kalman(lmList, img_shape):
#     smoothed_keypoints = []
#     for i, lm in enumerate(lmList):
#         if lm.visibility > 0.5:
#             prediction = kalman_filters[i].predict()
#             corrected = kalman_filters[i].correct(lm.x * img_shape[1], lm.y * img_shape[0])
#             x, y = corrected[0], corrected[1]
#             smoothed_keypoints.append((x, y, lm.z))
#         else:
#             smoothed_keypoints.append((lm.x * img_shape[1], lm.y * img_shape[0], lm.z))
#     return smoothed_keypoints
#舵机位置初始化
servos = [ptm.LobotServo(6, 0), ptm.LobotServo(7, 500), ptm.LobotServo(8, 0), ptm.LobotServo(14, 1000),
          ptm.LobotServo(15, 0), ptm.LobotServo(16, 0)]  # 创建6个舵机对象
#获取视频流
cap = cv2.VideoCapture(0)

detector = ptm.PoseDetector()  # 创建姿势检测器对象
# 平滑参数设置
alpha = 0.5  # 平滑因子，值越接近1表示对当前值权重越高，越接近0表示对之前值权重越高

# 初始化平滑后的角度变量
smoothed_horizontal_angle_left = 0.0  # 初始化左侧水平角度
smoothed_vertical_angle_left = 0.0  # 初始化左侧垂直角度
smoothed_horizontal_angle_right = 0.0  # 初始化右侧水平角度
smoothed_vertical_angle_right = 0.0  # 初始化右侧垂直角度

# 平滑参数设置
alpha = 0.5  # 平滑因子，值越接近1表示对当前值权重越高，越接近0表示对之前值权重越高

# 初始化平滑后的角度变量
smoothed_horizontal_angle_left = 0.0
smoothed_vertical_angle_left = 0.0
smoothed_horizontal_angle_right = 0.0
smoothed_vertical_angle_right = 0.0

# 增益因子
gain_factor = 2.0

# 初始化伺服角度变量
left_servo_angle = 0.0
right_servo_angle = 0.0

# 简单的平滑函数（加权平均）
def smooth_angle(current_angle, previous_angle, alpha=0.85):
    return alpha * current_angle + (1 - alpha) * previous_angle

#获取视频流
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.find_pose(img)
    lmList = detector.results.pose_landmarks.landmark

    if lmList:
        #卡尔曼滤波
        # smoothed_keypoints = smooth_with_kalman(lmList, img.shape)
        #
        # # 左侧关键点坐标
        # left_shoulder = smoothed_keypoints[11]
        # left_elbow = smoothed_keypoints[13]
        #
        # # 右侧关键点坐标
        # right_shoulder = smoothed_keypoints[12]
        # right_elbow = smoothed_keypoints[14]
        #左右侧关键节点构成分解的空间角度
        left_shoulder = [lmList[11].x, lmList[11].y, lmList[11].z]
        right_shoulder = [lmList[12].x, lmList[12].y, lmList[12].z]
        left_elbow = [lmList[13].x, lmList[13].y, lmList[13].z]
        right_elbow = [lmList[14].x, lmList[14].y, lmList[14].z]
        #计算左侧水平和垂直角度
        horizontal_angle_left = detector.calculate_horizontal_angle(left_shoulder, right_shoulder)
        vertical_angle_left = detector.calculate_vertical_angle(left_shoulder, left_elbow)
        #计算右侧水平和垂直角度
        horizontal_angle_right = detector.calculate_horizontal_angle(right_shoulder, left_shoulder)
        vertical_angle_right = detector.calculate_vertical_angle(right_shoulder, right_elbow)

        # 对左侧角度进行平滑处理
        smoothed_left_horizontal_angle = smooth_angle(horizontal_angle_left, smoothed_horizontal_angle_left, alpha)
        smoothed_left_vertical_angle= smooth_angle(vertical_angle_left, smoothed_vertical_angle_left, alpha)

        # 对右侧角度进行平滑处理
        smoothed_right_horizontal_angle = smooth_angle(horizontal_angle_right, smoothed_horizontal_angle_right, alpha)
        smoothed_right_vertical_angle = smooth_angle(vertical_angle_right, smoothed_vertical_angle_right, alpha)

        #映射平滑后的角度到舵机角度上
        left1_servo_angle = detector.map_angle_to_servo(smoothed_left_vertical_angle)
        left2_servo_angle = detector.map_angle_to_servo(smoothed_left_horizontal_angle)
        right1_servo_angle = detector.map_angle_to_servo(smoothed_right_vertical_angle)
        right2_servo_angle = detector.map_angle_to_servo(smoothed_right_horizontal_angle)
        # 二维角度
        left3_servo_angle = detector.get_angle(img, 11, 13, 15)  # Shoulder, Elbow, Wrist
        right3_servo_angle = detector.get_angle(img, 12, 14, 16)
        frame_count += 1  # 增加帧计数器
#最上面两行的角度打印
        cv2.putText(img, str(left1_servo_angle), (40, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, str(left2_servo_angle), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # 每隔指定的帧数输出一次角度
        if frame_count % output_frequency == 0:
            # print(f'Left Arm Angle: {angle_left1}' f' {angle_left2}')
            # print(f'Right Arm Angle: {angle_right1}'f' {angle_right2}')
            # 基于角度调整舵机目标位置
            # 假设
            # 假设舵机的位置范围是0到180度
            servo1_position = int(right3_servo_angle)  # 6号舵机与右手平面角对应
            servo2_position = int(right2_servo_angle)  # 7号舵机与右肘部水平角对应
            servo3_position = int(right1_servo_angle)  # 8号舵机与右肩部竖直角对应
            servo4_position = int(left3_servo_angle)  # 14号舵机与左手平面角对应
            servo5_position = int(left2_servo_angle)  # 15号舵机与左肘部水平角对应
            servo6_position = int(left1_servo_angle)  # 16号舵机与左肩部竖直角对应

            # 限制舵机的位置范围   确保舵机的位置范围限制在合法的范围内 这部分保证舵机的位置不超过180度。如果 servoX_position 大于180，会返回180。
            servo1_position = max(0, min(180, servo1_position))
            servo2_position = max(0, min(180, servo2_position))
            servo3_position = max(0, min(180, servo3_position))
            servo4_position = max(0, min(180, servo4_position))
            servo5_position = max(0, min(180, servo5_position))
            servo6_position = max(0, min(180, servo6_position))

            # 更新舵机目标位置
            servos[0].Position = int((servo1_position / 180) * 1000)
            servos[1].Position = int((servo2_position / 180) * 1000)
            servos[2].Position = int((servo3_position / 180) * 1000)
            servos[3].Position = int((servo4_position / 180) * 1000)
            servos[4].Position = int((servo3_position / 180) * 1000)
            servos[5].Position = int((servo4_position / 180) * 1000)

            # 控制舵机移动
            move_time = 1000  # 设定舵机移动时间为1000ms
            controller.move_servos(servos, len(servos), move_time)  # 发送控制命令


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
