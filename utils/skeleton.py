import os
import cv2
import mediapipe as mp
from collections import deque


# 在代码中，左右手的定义是基于手语者的视角，与观看视频的观众的左右相反
class VideoProcessor:
    def __init__(self, root_video_dir, output_base_dir):
        self.mp_holistic = mp.solutions.holistic   # 初始化 Mediapipe 模块
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,         # 选择静态识别，提高识别成功的概率
            model_complexity=2,             # 选择复杂度最高的模型
            min_detection_confidence=0.1,   # 模型置性度
            min_tracking_confidence=0.3     # 由于静态识别模式选择为True，该功能已经失效
        )
        self.all_frames = []                           # 存储所有帧数据
        self.valid_frames_indices = deque(maxlen=200)  # 存储有效帧索引

        self.root_video_dir = root_video_dir
        self.output_base_dir = output_base_dir

    def clear_frames(self):
        self.all_frames = []                           # 存储所有帧数据
        self.valid_frames_indices = deque(maxlen=200)  # 存储有效帧索引


class FrameData:
    def __init__(self, idx):
        self.idx = idx
        self.left_hand_3d = []     # 存储左手 3D 坐标 (x, y, z)
        self.right_hand_3d = []    # 存储右手 3D 坐标 (x, y, z)
        self.shoulders_3d = []     # 存储肩部和肘部的 3D 坐标
        self.pose_3d = []
        self.is_valid = False      # 标记是否正常识别到双手，用于辅助插值
        self.left_hand_valid = False
        self.right_hand_valid = False
        self.pose_valid = False


def get_max_consecutive_length(nums):
    """
    查找列表最大连续帧
    参数：
        nums：待检测列表
    返回：最大连续帧
    """
    if not nums:
        return 0

    nums = sorted(set(nums))  # 去重并排序
    max_len = 1
    current_len = 1

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1

    return max_len


def find_nearest_valid_frames(current_idx, all_frames, part_type='both'):
    """
    查找最近的前后有效帧
    参数：
        current_idx: 当前帧索引
        part_type: 'left'表示左手, 'right'表示右手, 'pose'表示pose, 'both'表示双手
    返回：(前有效帧, 后有效帧)
    """
    prev_valid = None
    next_valid = None
    min_prev_distance = float('inf')
    min_next_distance = float('inf')

    # 向前搜索（选择最近的前帧）
    for i in reversed(range(current_idx)):
        if i < 0:
            break
        frame = all_frames[i]
        # 根据part_type检查对应的部分是否有效
        if part_type == 'left' and frame.left_hand_3d:
            distance = current_idx - i
            if distance < min_prev_distance:
                min_prev_distance = distance
                prev_valid = frame
        elif part_type == 'right' and frame.right_hand_3d:
            distance = current_idx - i
            if distance < min_prev_distance:
                min_prev_distance = distance
                prev_valid = frame
        elif part_type == 'pose' and frame.shoulders_3d:
            distance = current_idx - i
            if distance < min_prev_distance:
                min_prev_distance = distance
                prev_valid = frame
        elif part_type == 'both' and (frame.left_hand_3d and frame.right_hand_3d):
            distance = current_idx - i
            if distance < min_prev_distance:
                min_prev_distance = distance
                prev_valid = frame

    # 向后搜索（选择最近的后帧）
    for i in range(current_idx + 1, len(all_frames)):
        if i >= len(all_frames):
            break
        frame = all_frames[i]
        # 根据part_type检查对应的部分是否有效
        if part_type == 'left' and frame.left_hand_3d:
            distance = i - current_idx
            if distance < min_next_distance:
                min_next_distance = distance
                next_valid = frame
        elif part_type == 'right' and frame.right_hand_3d:
            distance = i - current_idx
            if distance < min_next_distance:
                min_next_distance = distance
                next_valid = frame
        elif part_type == 'pose' and frame.shoulders_3d:
            distance = i - current_idx
            if distance < min_next_distance:
                min_next_distance = distance
                next_valid = frame
        elif part_type == 'both' and (frame.left_hand_3d and frame.right_hand_3d):
            distance = i - current_idx
            if distance < min_next_distance:
                min_next_distance = distance
                next_valid = frame

    return prev_valid, next_valid


def linear_weighted_interpolation(target_frame, all_frames, valid_frames_indices):
    """
    3D坐标(x, y, z)线性加权插值实现，左右手和pose分开插值
    参数：
        current_idx: 需要补全的目标帧数据 (FrameData对象)
        part_type: 'left'表示左手, 'right'表示右手, 'pose'表示pose, 'both'表示双手
    """

    def interpolate_part(part_type, all_frames):
        """线性插值指定部分 3D 坐标"""
        prev_valid, next_valid = find_nearest_valid_frames(
            target_frame.idx, all_frames, part_type)

        if not prev_valid and not next_valid:
            return None

        if prev_valid and next_valid:
            prev_distance = target_frame.idx - prev_valid.idx
            next_distance = next_valid.idx - target_frame.idx
            total_distance = prev_distance + next_distance

            prev_weight = next_distance / total_distance
            next_weight = prev_distance / total_distance

            # print(f"  {part_type} - prev_distance={prev_distance}, next_distance={next_distance}, prev_weight={prev_weight}, next_weight={next_weight}")

            interpolated = []
            # 关键点数量需要根据插值部分进行调整
            if part_type == 'pose':
                num_landmarks = 4
                prev_landmarks = prev_valid.shoulders_3d
                next_landmarks = next_valid.shoulders_3d
            else:
                num_landmarks = 21
                if part_type == 'left':
                    prev_landmarks = prev_valid.left_hand_3d
                    next_landmarks = next_valid.left_hand_3d
                else:
                    prev_landmarks = prev_valid.right_hand_3d
                    next_landmarks = next_valid.right_hand_3d

            if prev_landmarks and next_landmarks:
                for i in range(num_landmarks):
                    px, py, pz = prev_landmarks[i]
                    nx, ny, nz = next_landmarks[i]

                    # 对 x, y, z 坐标分别进行线性加权插值
                    interpolated_x = px * prev_weight + nx * next_weight
                    interpolated_y = py * prev_weight + ny * next_weight
                    interpolated_z = pz * prev_weight + nz * next_weight

                    interpolated.append(
                        (interpolated_x, interpolated_y, interpolated_z))
                return interpolated
            else:
                return None
        elif prev_valid:
            # print(f"  Only prev_valid for {part_type}")
            if part_type == 'pose':
                return prev_valid.shoulders_3d.copy()
            elif part_type == 'left':
                return prev_valid.left_hand_3d.copy()
            else:
                return prev_valid.right_hand_3d.copy()
        elif next_valid:
            # print(f"  Only next_valid for {part_type}")
            if part_type == 'pose':
                return next_valid.shoulders_3d.copy()
            elif part_type == 'left':
                return next_valid.left_hand_3d.copy()
            else:
                return next_valid.right_hand_3d.copy()
        else:
            return None

    # 分别处理左手、右手和pose的插值
    if not target_frame.left_hand_3d:
        # print("Interpolating left hand")
        target_frame.left_hand_3d = interpolate_part('left', all_frames)

    if not target_frame.right_hand_3d:
        # print("Interpolating right hand")
        target_frame.right_hand_3d = interpolate_part('right', all_frames)

    if not target_frame.shoulders_3d:
        # print("Interpolating pose")
        target_frame.shoulders_3d = interpolate_part('pose', all_frames)

    # 更新有效标记 - 只要有一只手或者pose有效就标记为有效
    if target_frame.left_hand_3d or target_frame.right_hand_3d or target_frame.shoulders_3d:
        target_frame.is_valid = True
        valid_frames_indices.append(target_frame.idx)


def process_video(video_processor: VideoProcessor, video_path, output_file, invalid_file, logger):
    """
    对指定路径下的视频进行处理,并在指定的文件中生成相应的.txt文件
    参数：
        video_path: 待处理的视频地址
        output_file: 保存的骨骼点txt信息的输出路径
        invalid_file: 不合格视频输出路径
    返回:
        missing_hands_count: 缺失手的总数
    """
    first_single_hand_frames = []             # 存储首次检测到只检测到一只手的帧索引
    video_processor.clear_frames()
    all_frames = video_processor.all_frames
    valid_frames_indices = video_processor.valid_frames_indices

    # 添加用于统计缺失帧的列表
    missing_left_hand_indices = []
    missing_right_hand_indices = []

    # 确保 output_file 目录存在
    output_dir_path = os.path.dirname(output_file)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    missing_hands_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_data = FrameData(frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=20)  # 提高对比度

        # Holistic检测
        results = video_processor.holistic.process(frame_rgb)

        hands_detected = 0
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        pose_landmarks = results.pose_landmarks

        if left_hand_landmarks:
            current_data.left_hand_3d = [(lm.x, lm.y, lm.z)
                                         for lm in left_hand_landmarks.landmark]
            hands_detected += 1
        else:
            # print(f"Frame {frame_idx}: Left hand not detected")
            missing_hands_count += 1
            missing_left_hand_indices.append(frame_idx)

        if right_hand_landmarks:
            current_data.right_hand_3d = [
                (lm.x, lm.y, lm.z) for lm in right_hand_landmarks.landmark]
            hands_detected += 1
        else:
            # print(f"Frame {frame_idx}: Right hand not detected")
            missing_hands_count += 1
            missing_right_hand_indices.append(frame_idx)

        if pose_landmarks:
            pose = pose_landmarks.landmark
            current_data.shoulders_3d = [
                (pose[11].x, pose[11].y, pose[11].z),  # 左肩
                (pose[12].x, pose[12].y, pose[12].z),  # 右肩
                (pose[13].x, pose[13].y, pose[13].z),  # 左肘
                (pose[14].x, pose[14].y, pose[14].z)   # 右肘
            ]
            current_data.pose_valid = True
        else:
            current_data.pose_valid = False
         #   print(f"Frame {frame_idx}: Pose not detected")

        if hands_detected != 2 and current_data.idx not in first_single_hand_frames:
            first_single_hand_frames.append(current_data.idx)

        if hands_detected == 2:
            current_data.is_valid = True
            valid_frames_indices.append(frame_idx)
        # else:
           # print(f"Frame {frame_idx}: Only {hands_detected} hands detected")

        all_frames.append(current_data)
        frame_idx += 1

    cap.release()

    # 对检测到一只手的数据进行线性插值补全
    for i in range(len(all_frames)):
        current_frame = all_frames[i]
        if not current_frame.is_valid:
            # print(f"not valid : {i}")
            linear_weighted_interpolation(
                current_frame, all_frames, valid_frames_indices)

    # 打印缺失帧的统计信息
    # print("\n缺失帧统计信息:")
    # print(f"左手缺失帧数: {len(missing_left_hand_indices)}")
    # print(f"左手缺失帧索引: {missing_left_hand_indices}")
    # print(f"右手缺失帧数: {len(missing_right_hand_indices)}")
    # print(f"右手缺失帧索引: {missing_right_hand_indices}")

    num_left = get_max_consecutive_length(missing_left_hand_indices)
    num_right = get_max_consecutive_length(missing_right_hand_indices)

    if num_left <= 10 and num_right <= 10 and missing_hands_count <= 50:
        # 保存骨骼数据到 .txt 文件，格式化输出
        with open(output_file, "w") as f:
            for frame in all_frames:
                # 只保存左手、右手、肩部坐标，不包含其他信息
                all_landmarks = frame.left_hand_3d + frame.right_hand_3d + frame.shoulders_3d
                if all_landmarks:
                    for lm in all_landmarks:
                        # 写入每个关键点的 x, y, z 坐标，保留10位小数，按照空格分隔
                        f.write(f"{lm[0]:.10f} {lm[1]:.10f} {lm[2]:.10f} ")

                    # 每一帧按换行符分割
                    f.write("\n")

        logger.info(
            f"{video_path} skeleton data has been saved to {output_file}")
        # print(list(valid_frames_indices))
    else:
        with open(output_file, "w") as f:
            pass  # 创建空文件
        with open(invalid_file, "a") as f:
            f.write(output_file + "\n")
        logger.warning(f"{video_path} is invalid!")

    return missing_hands_count, num_left, num_right


def process_one(video_processor: VideoProcessor, subdir, file_name, invalid_file, logger):
    video_path = os.path.join(
        video_processor.root_video_dir, subdir, file_name)
    relative_path = os.path.relpath(video_path, video_processor.root_video_dir)
    output_file = os.path.join(
        video_processor.output_base_dir, relative_path.replace(".mp4", "_skeleton.txt"))

    # 确保文件夹路径存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"Processing {video_path}...")
    process_video(video_processor, video_path,
                  output_file, invalid_file, logger)
