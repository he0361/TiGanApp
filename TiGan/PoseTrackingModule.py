import cv2
import numpy as np
import mediapipe as mp
import math


# 舵机控制指令和定义
FRAME_HEADER = 0x55  # 假设帧头为0x55
CMD_SERVO_MOVE = 0x03  # 假设舵机移动指令为0x03
color = (255, 0, 0)
scale = 5
#舵机指令
class LobotServo:
        def __init__(self, ID, Position):
            self.ID = ID
            self.Position = Position

def get_low_byte(value):
        return value & 0xFF

def get_high_byte(value):
        return (value >> 8) & 0xFF

class LobotServoController:
    def __init__(self):
            pass  # 可根据需要添加串行通信初始化

    def move_servos(self, servos, num, time):
            if num < 1 or num > 16 or time <= 0:
                return  # 舵机数必须在1到16之间，且时间大于零

            buf = bytearray()
            buf.append(FRAME_HEADER)
            buf.append(FRAME_HEADER)
            buf.append(num * 3 + 5)  # 数据长度 = 要控制舵机数*3+5
            buf.append(CMD_SERVO_MOVE)  # 0x03是移动指令
            buf.append(num)  # 控制舵机个数
            buf.append(get_low_byte(time))  # 取得时间的低八位
            buf.append(get_high_byte(time))  # 取得时间的高八位

            for servo in servos:
                buf.append(servo.ID)  # 填充舵机ID
                buf.append(get_low_byte(servo.Position))  # 填充目标位置低八位
                buf.append(get_high_byte(servo.Position))  # 填充目标位置高八位

            print("Data to send (hex):", ' '.join(f"{byte:02X}" for byte in buf))  # 打印要发送的数据
class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.8, trackCon=0.8):
        self.smooth = smooth
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_angle(self, img, point1, point2, point3, draw=True):
        if self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            h, w, _ = img.shape

            p1 = (int(landmarks[point1].x * w), int(landmarks[point1].y * h))
            p2 = (int(landmarks[point2].x * w), int(landmarks[point2].y * h))
            p3 = (int(landmarks[point3].x * w), int(landmarks[point3].y * h))

            angle = self.calculate_angle(p1, p2, p3)
            if draw:
                cv2.putText(img, str(int(angle)), p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.circle(img, p1, 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, p2, 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, p3, 5, (0, 255, 0), cv2.FILLED)
                cv2.line(img, p1, p2, (0, 255, 0), 2)
                cv2.line(img, p2, p3, (0, 255, 0), 2)
            return angle
        return None

    def calculate_angle(self, p1, p2, p3):
        vector_1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        vector_2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        length_1 = np.linalg.norm(vector_1)
        length_2 = np.linalg.norm(vector_2)

        dot_product = np.dot(vector_1, vector_2)
        cos_theta = dot_product / (length_1 * length_2)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
#水平角度
    def calculate_horizontal_angle(self, left_shoulder, right_shoulder):
        dx = right_shoulder[1] - left_shoulder[1]
        dy = right_shoulder[2] - left_shoulder[2]
        angle = math.atan2(dy, dx) * 180 / math.pi
        return angle
#垂直角度
    def calculate_vertical_angle(self, shoulder, elbow):
        dx = elbow[1] - shoulder[1]
        dy = elbow[2] - shoulder[2]
        angle = math.atan2(dy, dx) * 180 / math.pi
        return angle

    def map_angle_to_servo(self, angle, servo_min=0, servo_max=180):
        servo_angle = (angle + 180) / 360 * (servo_max - servo_min) + servo_min
        return servo_angle
