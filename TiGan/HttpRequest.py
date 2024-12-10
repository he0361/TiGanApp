import asyncio
import websockets
import numpy as np
import cv2
import mediapipe as mp
import PoseTrackingModule as ptm
import sys
frame_count = 0  # 初始帧计数器
output_frequency = 5  # 设置输出频率，输出每5帧的角度
controller = ptm.LobotServoController()  # 创建控制器对象
servos = [ptm.LobotServo(6, 0), ptm.LobotServo(7, 500), ptm.LobotServo(8, 0), ptm.LobotServo(14, 1000),
          ptm.LobotServo(15, 0), ptm.LobotServo(16, 0)]  # 创建6个舵机对象

detector = ptm.PoseDetector()  # 创建姿势检测器对象
alpha = 0.5  # 平滑因子
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
# 初始化平滑后的角度变量
smoothed_horizontal_angle_left = 0.0  # 初始化左侧水平角度
smoothed_vertical_angle_left = 0.0  # 初始化左侧垂直角度
smoothed_horizontal_angle_right = 0.0  # 初始化右侧水平角度
smoothed_vertical_angle_right = 0.0  # 初始化右侧垂直角度
# WebSocket 服务器处理函数

async  def handle(websocket):
    global frame_count
    print("Connected to client")
    try:
        while True:
            # 接收视频帧数据
            data = await websocket.recv()
            print(f"Received data: {len(data)} bytes")

            # 将二进制数据转换为 NumPy 数组
            np_arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 处理视频帧
                img = detector.find_pose(img)
                lmList = detector.results.pose_landmarks.landmark

                if lmList:
                    left_shoulder = [lmList[11].x, lmList[11].y, lmList[11].z]
                    right_shoulder = [lmList[12].x, lmList[12].y, lmList[12].z]
                    left_elbow = [lmList[13].x, lmList[13].y, lmList[13].z]
                    right_elbow = [lmList[14].x, lmList[14].y, lmList[14].z]

                    # 计算左右肩膀和肘部的角度
                    horizontal_angle_left = detector.calculate_horizontal_angle(left_shoulder, right_shoulder)
                    vertical_angle_left = detector.calculate_vertical_angle(left_shoulder, left_elbow)
                    horizontal_angle_right = detector.calculate_horizontal_angle(right_shoulder, left_shoulder)
                    vertical_angle_right = detector.calculate_vertical_angle(right_shoulder, right_elbow)

                    # 对左右角度进行平滑处理
                    smoothed_left_horizontal_angle = smooth_angle(horizontal_angle_left, smoothed_horizontal_angle_left,
                                                                  alpha)
                    smoothed_left_vertical_angle = smooth_angle(vertical_angle_left, smoothed_vertical_angle_left,
                                                                alpha)
                    smoothed_right_horizontal_angle = smooth_angle(horizontal_angle_right,
                                                                   smoothed_horizontal_angle_right, alpha)
                    smoothed_right_vertical_angle = smooth_angle(vertical_angle_right, smoothed_vertical_angle_right,
                                                                 alpha)

                    # 映射角度到舵机角度
                    left1_servo_angle = detector.map_angle_to_servo(smoothed_left_vertical_angle)
                    left2_servo_angle = detector.map_angle_to_servo(smoothed_left_horizontal_angle)
                    right1_servo_angle = detector.map_angle_to_servo(smoothed_right_vertical_angle)
                    right2_servo_angle = detector.map_angle_to_servo(smoothed_right_horizontal_angle)

                    left3_servo_angle = detector.get_angle(img, 11, 13, 15)  # Left shoulder, elbow, wrist
                    right3_servo_angle = detector.get_angle(img, 12, 14, 16)  # Right shoulder, elbow, wrist

                    frame_count += 1  # 增加帧计数器

                    # 每隔指定的帧数输出一次角度
                    if frame_count % output_frequency == 0:
                        # 控制舵机
                        servo1_position = int(right3_servo_angle)
                        servo2_position = int(right2_servo_angle)
                        servo3_position = int(right1_servo_angle)
                        servo4_position = int(left3_servo_angle)
                        servo5_position = int(left2_servo_angle)
                        servo6_position = int(left1_servo_angle)

                        # 确保舵机位置在合法范围内
                        servo1_position = max(0, min(180, servo1_position))
                        servo2_position = max(0, min(180, servo2_position))
                        servo3_position = max(0, min(180, servo3_position))
                        servo4_position = max(0, min(180, servo4_position))
                        servo5_position = max(0, min(180, servo5_position))
                        servo6_position = max(0, min(180, servo6_position))

                        # 更新舵机位置
                        servos[0].Position = int((servo1_position / 180) * 1000)
                        servos[1].Position = int((servo2_position / 180) * 1000)
                        servos[2].Position = int((servo3_position / 180) * 1000)
                        servos[3].Position = int((servo4_position / 180) * 1000)
                        servos[4].Position = int((servo5_position / 180) * 1000)
                        servos[5].Position = int((servo6_position / 180) * 1000)

                        # 控制舵机
                        move_time = 1000  # 设置舵机移动时间
                        controller.move_servos(servos, len(servos), move_time)  # 发送控制命令

                cv2.imshow("Processed Frame", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")


async def handle_video_stream(websocket,path):
    global frame_count
    print("Connected to client")
    try:
        while True:
            # 接收视频帧数据
            data = await websocket.recv()
            print(f"Received data: {len(data)} bytes")

            # 将二进制数据转换为 NumPy 数组
            np_arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 处理视频帧
                img = detector.find_pose(img)
                lmList = detector.results.pose_landmarks.landmark

                if lmList:
                    left_shoulder = [lmList[11].x, lmList[11].y, lmList[11].z]
                    right_shoulder = [lmList[12].x, lmList[12].y, lmList[12].z]
                    left_elbow = [lmList[13].x, lmList[13].y, lmList[13].z]
                    right_elbow = [lmList[14].x, lmList[14].y, lmList[14].z]

                    # 计算左右肩膀和肘部的角度
                    horizontal_angle_left = detector.calculate_horizontal_angle(left_shoulder, right_shoulder)
                    vertical_angle_left = detector.calculate_vertical_angle(left_shoulder, left_elbow)
                    horizontal_angle_right = detector.calculate_horizontal_angle(right_shoulder, left_shoulder)
                    vertical_angle_right = detector.calculate_vertical_angle(right_shoulder, right_elbow)

                    # 对左右角度进行平滑处理
                    smoothed_left_horizontal_angle = smooth_angle(horizontal_angle_left, smoothed_horizontal_angle_left, alpha)
                    smoothed_left_vertical_angle = smooth_angle(vertical_angle_left, smoothed_vertical_angle_left, alpha)
                    smoothed_right_horizontal_angle = smooth_angle(horizontal_angle_right, smoothed_horizontal_angle_right, alpha)
                    smoothed_right_vertical_angle = smooth_angle(vertical_angle_right, smoothed_vertical_angle_right, alpha)

                    # 映射角度到舵机角度
                    left1_servo_angle = detector.map_angle_to_servo(smoothed_left_vertical_angle)
                    left2_servo_angle = detector.map_angle_to_servo(smoothed_left_horizontal_angle)
                    right1_servo_angle = detector.map_angle_to_servo(smoothed_right_vertical_angle)
                    right2_servo_angle = detector.map_angle_to_servo(smoothed_right_horizontal_angle)

                    left3_servo_angle = detector.get_angle(img, 11, 13, 15)  # Left shoulder, elbow, wrist
                    right3_servo_angle = detector.get_angle(img, 12, 14, 16)  # Right shoulder, elbow, wrist

                    frame_count += 1  # 增加帧计数器

                    # 每隔指定的帧数输出一次角度
                    if frame_count % output_frequency == 0:
                        # 控制舵机
                        servo1_position = int(right3_servo_angle)
                        servo2_position = int(right2_servo_angle)
                        servo3_position = int(right1_servo_angle)
                        servo4_position = int(left3_servo_angle)
                        servo5_position = int(left2_servo_angle)
                        servo6_position = int(left1_servo_angle)

                        # 确保舵机位置在合法范围内
                        servo1_position = max(0, min(180, servo1_position))
                        servo2_position = max(0, min(180, servo2_position))
                        servo3_position = max(0, min(180, servo3_position))
                        servo4_position = max(0, min(180, servo4_position))
                        servo5_position = max(0, min(180, servo5_position))
                        servo6_position = max(0, min(180, servo6_position))

                        # 更新舵机位置
                        servos[0].Position = int((servo1_position / 180) * 1000)
                        servos[1].Position = int((servo2_position / 180) * 1000)
                        servos[2].Position = int((servo3_position / 180) * 1000)
                        servos[3].Position = int((servo4_position / 180) * 1000)
                        servos[4].Position = int((servo5_position / 180) * 1000)
                        servos[5].Position = int((servo6_position / 180) * 1000)

                        # 控制舵机
                        move_time = 1000  # 设置舵机移动时间
                        controller.move_servos(servos, len(servos), move_time)  # 发送控制命令

                cv2.imshow("Processed Frame", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

# 启动 WebSocket 服务器
async def main():
    try:
        print("Starting WebSocket server on ws://localhost:8765...")
        async with websockets.serve(handle, "localhost", 8765):
            print("Server started successfully.")
            await asyncio.Future()  # 运行服务器，直到被终止
    except Exception as e:
        print(f"Error starting WebSocket server: {e}")
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # 适配 Windows 环境
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped manually.")