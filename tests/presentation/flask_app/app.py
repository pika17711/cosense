from flask import Flask, render_template, Response, jsonify
from utils.sharedInfo import SharedInfo
from appConfig import AppConfig
from detection.detectionRPCClient import DetectionRPCClient
import numpy as np
import cv2


def create_app(cfg: AppConfig, shared_info: SharedInfo):
    app = Flask(__name__)

    def gen_pcd_img():
        while True:
            pcd_img = shared_info.get_presentation_info_copy().get('pcd_img')       # pcd_img.shape = (1080, 1920, 3)
            if pcd_img is not None and pcd_img.size != 0:
                pcd_img *= 255
                img_cv = cv2.cvtColor(pcd_img, cv2.COLOR_RGB2BGR)

                ret, jpeg = cv2.imencode('.jpg', img_cv)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # 如果没有图像数据，发送一个空白图像或等待
                blank_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
                blank_image[:] = 255
                ret, jpeg = cv2.imencode('.jpg', blank_image)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    
    def gen_request_map():
        while True:
            request_map = 1 - shared_info.get_presentation_info_copy().get('ego_comm_mask').squeeze()   # request_map.shape = (48, 176)
            if request_map is not None and request_map.size != 0:

                request_map = np.flipud(request_map)        # 上下翻转

                request_map_img = np.zeros((*request_map.shape, 3), dtype=np.uint8)

                color_0 = [255, 0, 255]     # 紫色
                color_1 = [0, 255, 255]     # 黄色

                request_map_img[request_map == 0] = color_0
                request_map_img[request_map == 1] = color_1

                original_height, original_width = request_map_img.shape[:2]

                scale_factor = 5
                new_width = original_width * scale_factor
                new_height = original_height * scale_factor

                request_map_img = cv2.resize(request_map_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                ret, jpeg = cv2.imencode('.jpg', request_map_img)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # 如果没有图像数据，发送一个空白图像或等待
                blank_image = np.zeros((240, 880, 3), dtype=np.uint8)
                blank_image[:] = 255
                ret, jpeg = cv2.imencode('.jpg', blank_image)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def gen_others_comm_mask():
        while True:
            others_comm_mask = shared_info.get_presentation_info_copy().get('others_comm_mask').squeeze()   # others_comm_mask.shape = (1, 1, 48, 176)
            if others_comm_mask is not None and others_comm_mask.size != 0:

                others_comm_mask = np.flipud(others_comm_mask)  # 上下翻转

                others_comm_mask_img = np.zeros((*others_comm_mask.shape, 3), dtype=np.uint8)

                color_0 = [255, 0, 255]     # 紫色
                color_1 = [0, 255, 255]     # 黄色

                others_comm_mask_img[others_comm_mask == 0] = color_0
                others_comm_mask_img[others_comm_mask == 1] = color_1

                original_height, original_width = others_comm_mask_img.shape[:2]

                scale_factor = 5
                new_width = original_width * scale_factor
                new_height = original_height * scale_factor

                others_comm_mask_img = cv2.resize(others_comm_mask_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                ret, jpeg = cv2.imencode('.jpg', others_comm_mask_img)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # 如果没有图像数据，发送一个空白图像或等待
                blank_image = np.zeros((240, 880, 3), dtype=np.uint8)
                blank_image[:] = 255
                ret, jpeg = cv2.imencode('.jpg', blank_image)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def gen_ego_feature():
        while True:
            ego_feature = shared_info.get_presentation_info_copy().get('ego_feature').squeeze()   # ego_feature.shape = (1, 256, 48, 176)
            if ego_feature is not None and ego_feature.size != 0:

                ego_feature = ego_feature.sum(0)
                ego_feature = np.flipud(ego_feature)  # 上下翻转

                # 将feature的值缩放到0到1之间
                # 找到最小值和最大值
                min_val = np.min(ego_feature)
                max_val = np.max(ego_feature)

                # 避免除以零的情况，如果所有值都相同
                if max_val == min_val:
                    normalized_ego_feature = np.zeros_like(ego_feature, dtype=np.float32)
                else:
                    normalized_ego_feature = (ego_feature - min_val) / (max_val - min_val)

                # BGR格式
                green = np.array([0, 255, 0], dtype=np.float32)  # 绿
                blue = np.array([255, 0, 0], dtype=np.float32)  # 蓝

                # 将 normalized_feature 从 (H, W) 扩展到 (H, W, 1)，以便与 (3,) 形状的颜色向量进行广播
                alpha_channel = normalized_ego_feature[:, :, np.newaxis]

                # 直接进行颜色插值
                interpolated_colors = (1 - alpha_channel) * green + alpha_channel * blue    # 越大越蓝

                # 确保颜色值在0-255范围内，并转换为np.uint8
                ego_feature_img = np.clip(interpolated_colors, 0, 255).astype(np.uint8)

                original_height, original_width = ego_feature_img.shape[:2]

                scale_factor = 5
                new_width = original_width * scale_factor
                new_height = original_height * scale_factor

                ego_feature_img = cv2.resize(ego_feature_img, (new_width, new_height),
                                             interpolation=cv2.INTER_NEAREST)

                ret, jpeg = cv2.imencode('.jpg', ego_feature_img)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # 如果没有图像数据，发送一个空白图像或等待
                blank_image = np.zeros((240, 880, 3), dtype=np.uint8)
                blank_image[:] = 255
                ret, jpeg = cv2.imencode('.jpg', blank_image)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def gen_fused_feature():
        while True:
            fused_feature = shared_info.get_presentation_info_copy().get('fused_feature').squeeze()   # fused_feature.shape = (1, 256, 48, 176)
            if fused_feature is not None and fused_feature.size != 0:
                fused_feature = fused_feature.sum(0)
                fused_feature = np.flipud(fused_feature)  # 上下翻转

                # 将feature的值缩放到0到1之间
                # 找到最小值和最大值
                min_val = np.min(fused_feature)
                max_val = np.max(fused_feature)

                # 避免除以零的情况，如果所有值都相同
                if max_val == min_val:
                    normalized_fused_feature = np.zeros_like(fused_feature, dtype=np.float32)
                else:
                    normalized_fused_feature = (fused_feature - min_val) / (max_val - min_val)

                # BGR格式
                green = np.array([0, 255, 0], dtype=np.float32)  # 绿
                blue = np.array([255, 0, 0], dtype=np.float32)  # 蓝

                # 将 normalized_feature 从 (H, W) 扩展到 (H, W, 1)，以便与 (3,) 形状的颜色向量进行广播
                alpha_channel = normalized_fused_feature[:, :, np.newaxis]

                # 直接进行颜色插值
                interpolated_colors = (1 - alpha_channel) * green + alpha_channel * blue    # 越大越蓝

                # 确保颜色值在0-255范围内，并转换为np.uint8
                fused_feature_img = np.clip(interpolated_colors, 0, 255).astype(np.uint8)

                original_height, original_width = fused_feature_img.shape[:2]

                scale_factor = 5
                new_width = original_width * scale_factor
                new_height = original_height * scale_factor

                fused_feature_img = cv2.resize(fused_feature_img, (new_width, new_height),
                                               interpolation=cv2.INTER_NEAREST)

                ret, jpeg = cv2.imencode('.jpg', fused_feature_img)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # 如果没有图像数据，发送一个空白图像或等待
                blank_image = np.zeros((240, 880, 3), dtype=np.uint8)
                blank_image[:] = 255
                ret, jpeg = cv2.imencode('.jpg', blank_image)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    @app.route('/')
    def home():
        return render_template('home.html')
    
    @app.route('/get_pcd_img')
    def get_pcd_img():
        return Response(gen_pcd_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_request_map')
    def get_request_map():
        return Response(gen_request_map(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_others_comm_mask')
    def get_others_comm_mask():
        return Response(gen_others_comm_mask(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_ego_feature')
    def get_ego_feature():
        return Response(gen_ego_feature(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_fused_feature')
    def get_fused_feature():
        return Response(gen_fused_feature(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/get_car_state')
    def get_car_state():
        presentation_info = shared_info.get_presentation_info_copy()
        lidar_pose = presentation_info.get('lidar_pose')    # (x, y, z, roll, pitch, yaw)
        if lidar_pose is not None and lidar_pose.size == 6:
            longitude, latitude, _, _, _, heading = lidar_pose
            longitude = f'{abs(longitude):.2f}°' + ('E' if longitude > 0 else 'W')
            latitude = f'{abs(latitude):.2f}°' + ('N' if latitude > 0 else 'S')
            heading = f'{heading:.2f}°'
        else:
            longitude = 'N/A'
            latitude = 'N/A'
            heading = 'N/A'
        
        speed = presentation_info.get('speed')
        if speed is not None and speed.size == 1:
            speed = f"{speed[0]:.2f}km/s"
        else:
            speed = 'N/A'

        return jsonify(longitude=longitude, latitude=latitude, heading=heading, speed=speed)
    
    @app.route('/get_car_id')
    def get_car_id():
        return jsonify(car_id=cfg.id)

    return app
