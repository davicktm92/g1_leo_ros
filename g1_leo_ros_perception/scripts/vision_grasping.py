#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#
# Author: David Orbea <david.orbea@upm.es>
# Creation date: 2026-04-10
# Description: ROS2 node for detecting boxes and define grasping points for the phase of sliding box and grasping it.
#
# Copyright (c) 2026 David Orbea, Center for Automation and Robotics.
# Todos los derechos reservados.

#!/usr/bin/env python3
"""ROS 2 Humble: YOLOE + ORB + RealSense RGB-D para caja REPSOL.

Un solo archivo. Sin OCR, .geometry, scipy ni message_filters.
Publica puntos 3D en TF, NO poses de agarre completas ni órdenes al G1.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from ultralytics import YOLOE
import os
from ament_index_python.packages import get_package_share_directory
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor



class BoxYoloeNode(Node):
    def __init__(self):
        super().__init__('yoloe_logo_box_vision')

        # Coordenadas [xmin, ymin, xmax, ymax] sobre la FOTO DE REFERENCIA.
        defaults = {
            'reference_box_xyxy': [0.0, 32.0, 616.0, 470.0],
            'repsol_box_xyxy': [262.0, 203.0, 535.0, 247.0],
            'lubricants_box_xyxy': [264.0, 273.0, 551.0, 314.0],
            'model': 'yoloe-26m-seg.pt',
            'device': '0',  # '0' para GPU NVIDIA compatible
            'confidence': 0.30,
            'imgsz': 640,
            'max_hz': 15.0,
            'grasp_span_factor': 0.8,
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'sync_slop_s': 0.10,
            'depth_scale_m': 0.001,
        }
        self.p = {}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            self.p[name] = self.get_parameter(name).value
        if float(self.p['max_hz']) <= 0:
            raise ValueError('max_hz debe ser positivo')
        if float(self.p['grasp_span_factor']) <= 0:
            raise ValueError('grasp_span_factor debe ser positivo')

        #self.reference = cv2.imread(str(self.p['reference_image']))
        self.reference = cv2.imread(os.path.join(get_package_share_directory('g1_leo_ros_perception'), 'config', '1.jpg'))
        if self.reference is None:
            raise ValueError('Especifica reference_image con la ruta de tu fotografía.')
        rh, rw = self.reference.shape[:2]
        self.ref_box = self.valid_rect(self.p['reference_box_xyxy'], rw, rh)
        if self.ref_box is None:
            self.ref_box = np.array([0., 0., float(rw), float(rh)])

        self.word_boxes = {}
        for word, param in [('REPSOL', 'repsol_box_xyxy'),
                            ('LUBRICANTS', 'lubricants_box_xyxy')]:
            box = self.valid_rect(self.p[param], rw, rh)
            if box is None:
                raise ValueError(
                    f'Falta {param}: indica el recuadro de {word} en la imagen '
                    'de referencia. No se utiliza OCR.'
                )
            self.word_boxes[word] = box

        # OpenCV: características de la referencia precalculadas una sola vez.
        self.orb = cv2.ORB_create(nfeatures=1800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ref_mask = np.zeros((rh, rw), np.uint8)
        x0, y0, x1, y1 = self.ref_box.astype(int)
        ref_mask[y0:y1, x0:x1] = 255
        gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, ref_mask)
        if self.des_ref is None or len(self.kp_ref) < 10:
            raise ValueError('La referencia tiene pocos detalles para ORB.')

        self.model = YOLOE(str(self.p['model']))
        self.prompt_ready = False
        self.bridge = CvBridge()
        self.tf_pub = TransformBroadcaster(self)
        self.camera_info = None
        self.rgb = None
        self.depth = None
        self.last_stamp = None

        self.info_sub = self.create_subscription(
            CameraInfo, self.p['camera_info_topic'], self.info_callback,
            qos_profile_sensor_data)
        self.rgb_sub = self.create_subscription(
            Image, self.p['color_topic'], self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, self.p['depth_topic'], self.depth_callback, qos_profile_sensor_data)
        self.detected_pub = self.create_publisher(Bool, 'box_vision/detected', 10)
        self.mask_pub = self.create_publisher(Image, 'box_vision/mask', 2)
        self.debug_pub = self.create_publisher(Image, 'box_vision/debug_image', 2)
        self.timer = self.create_timer(1.0 / float(self.p['max_hz']), self.process)
        self.get_logger().info('YOLOE + ORB preparado; esperando RGB-D y CameraInfo.')

    @staticmethod
    def valid_rect(box, w, h):
        a = np.asarray(box, dtype=float)
        if a.size != 4 or not np.isfinite(a).all():
            return None
        return a if (0 <= a[0] < a[2] <= w and 0 <= a[1] < a[3] <= h) else None

    @staticmethod
    def corners(box):
        x0, y0, x1, y1 = box
        return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def info_callback(self, msg):
        if msg.k[0] > 0 and msg.k[4] > 0:
            self.camera_info = msg

    def rgb_callback(self, msg):
        self.rgb = msg

    def depth_callback(self, msg):
        self.depth = msg

    @staticmethod
    def timestamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def detect(self, image):

        arguments = dict(
            source=image,
            conf=float(self.p['confidence']),
            imgsz=int(self.p['imgsz']),
            device=str(self.p['device']),
            retina_masks=True,
            verbose=False
        )

        if not self.prompt_ready:

            prompt = {
                'bboxes': np.array([self.ref_box], dtype=np.float32),
                'cls': np.array([0], dtype=np.int64)
            }

            self.get_logger().info(
                f"RGB: {type(image).__name__}, {image.shape} | "
                f"Referencia: {type(self.reference).__name__}, "
                f"{self.reference.shape}"
            )

            results = self.model.predict(
                **arguments,
                refer_image=self.reference,
                visual_prompts=prompt,
                predictor=YOLOEVPSegPredictor
            )

            self.prompt_ready = True
            self.get_logger().info("Referencia YOLOE configurada correctamente.")

        else:

            results = self.model.predict(**arguments)

        return results[0]

    def project_words(self, image, mask):
        """Homografía: posiciones del texto en la referencia -> imagen actual."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Buscar puntos visuales únicamente dentro de la caja segmentada.
        kp, des = self.orb.detectAndCompute(gray, mask * 255)
        if des is None or len(kp) < 10:
            return None
        matches = self.matcher.knnMatch(self.des_ref, des, k=2)
        good = [a for pair in matches if len(pair) == 2
                for a, b in [pair] if a.distance < 0.75 * b.distance]
        if len(good) < 10:
            return None
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inliers is None or int(inliers.sum()) < 10:
            return None

        words = {}
        h, w = mask.shape
        for word, rect in self.word_boxes.items():
            quad = cv2.perspectiveTransform(self.corners(rect)[None], H)[0]
            if not np.isfinite(quad).all():
                return None
            u, v = np.rint(np.mean(quad, axis=0)).astype(int)
            if not (0 <= u < w and 0 <= v < h and mask[v, u]):
                return None
            words[word] = quad
        return words

    def grasp_pixels(self, words):
        """Dos referencias sobre la línea de texto, aunque las palabras estén apiladas."""
        q1, q2 = words['REPSOL'], words['LUBRICANTS']
        c1, c2 = q1.mean(axis=0), q2.mean(axis=0)
        axis1 = 0.5 * ((q1[1] - q1[0]) + (q1[2] - q1[3]))
        axis2 = 0.5 * ((q2[1] - q2[0]) + (q2[2] - q2[3]))
        n1, n2 = np.linalg.norm(axis1), np.linalg.norm(axis2)
        if min(n1, n2) < 3.0:
            return None
        axis1, axis2 = axis1 / n1, axis2 / n2
        if np.dot(axis1, axis2) < 0.5:
            return None
        direction = axis1 + axis2
        direction /= np.linalg.norm(direction)
        midpoint = 0.5 * (c1 + c2)
        half_span = 0.5 * float(self.p['grasp_span_factor']) * max(n1, n2)
        a, b = midpoint - half_span * direction, midpoint + half_span * direction
        if np.linalg.norm(a - b) < 12.0:
            return None
        left, right = sorted((a, b), key=lambda p: float(p[0]))
        return {'repsol': c1, 'lubricants': c2, 'left': left, 'right': right}

    def xyz_at(self, pixel, raw_depth, mask):
        """Mediana de profundidad válida de un parche interior a la máscara."""
        u, v = np.rint(pixel).astype(int)
        h, w = raw_depth.shape
        x0, x1 = max(0, u-6), min(w, u+7)
        y0, y1 = max(0, v-6), min(h, v+7)
        if x0 >= x1 or y0 >= y1:
            return None
        depth = raw_depth[y0:y1, x0:x1]
        if raw_depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * float(self.p['depth_scale_m'])
        valid = (np.isfinite(depth) & (depth > 0.2) & (depth < 2.0)
                 & (mask[y0:y1, x0:x1] > 0))
        if np.count_nonzero(valid) < 4:
            return None
        z = float(np.median(depth[valid]))
        k = self.camera_info.k
        return ((float(pixel[0]) - k[2]) * z / k[0],
                (float(pixel[1]) - k[5]) * z / k[4], z)

    @staticmethod
    def make_tf(header, child, xyz):
        msg = TransformStamped()
        msg.header = header
        msg.child_frame_id = child
        msg.transform.translation.x = float(xyz[0])
        msg.transform.translation.y = float(xyz[1])
        msg.transform.translation.z = float(xyz[2])
        msg.transform.rotation.w = 1.0   # Frame óptico, no orientación de pinza
        return msg

    def process(self):
        rgb_msg, depth_msg = self.rgb, self.depth
        if rgb_msg is None or depth_msg is None or self.camera_info is None:
            return
        stamp = (rgb_msg.header.stamp.sec, rgb_msg.header.stamp.nanosec)
        if stamp == self.last_stamp:
            return
        if abs(self.timestamp(rgb_msg) - self.timestamp(depth_msg)) > float(self.p['sync_slop_s']):
            return
        self.last_stamp = stamp
        if not rgb_msg.header.frame_id:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            if image.shape[:2] != depth.shape[:2]:
                raise ValueError('La profundidad debe estar ALINEADA con el RGB.')
            if (self.camera_info.height, self.camera_info.width) != image.shape[:2]:
                raise ValueError('CameraInfo no coincide con la resolución RGB.')
            if depth.dtype not in (np.uint16, np.float32, np.float64):
                raise ValueError(f'Formato depth no soportado: {depth.dtype}')

            result = self.detect(image)
            mask_selected = np.zeros(image.shape[:2], dtype=np.uint8)
            found = None
            if result.boxes is not None and result.masks is not None:
                confidences = result.boxes.conf.cpu().numpy()
                masks = result.masks.data.cpu().numpy()
                for i in np.argsort(-confidences):
                    if i >= len(masks):
                        continue
                    mask = masks[i]
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)
                    if int(mask.sum()) < 1200:
                        continue
                    words = self.project_words(image, mask)
                    if words is None:
                        continue
                    pixels = self.grasp_pixels(words)
                    if pixels is None:
                        continue
                    points = {}
                    for name, uv in pixels.items():
                        xyz = self.xyz_at(uv, depth, mask)
                        if xyz is None:
                            break
                        points[name] = xyz
                    if len(points) != 4:
                        continue
                    m = cv2.moments(mask)
                    if m['m00'] == 0:
                        continue
                    center = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])
                    points['center'] = self.xyz_at(center, depth, mask)
                    if points['center'] is None:
                        continue
                    found = (points, pixels)
                    mask_selected = mask
                    break

            self.detected_pub.publish(Bool(data=(found is not None)))
            debug = image.copy()
            if found is not None:
                points, pixels = found
                frames = {'center': 'box_center', 'repsol': 'box_repsol',
                          'lubricants': 'box_lubricants',
                          'left': 'box_grasp_left', 'right': 'box_grasp_right'}
                self.tf_pub.sendTransform([
                    self.make_tf(rgb_msg.header, child, points[key])
                    for key, child in frames.items()
                ])
                contours, _ = cv2.findContours(mask_selected, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
                for name, uv in pixels.items():
                    x, y = np.rint(uv).astype(int)
                    cv2.circle(debug, (x, y), 6, (0, 255, 255), -1)
                    cv2.putText(debug, name, (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(debug, 'BOX / WORDS NOT FOUND', (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = rgb_msg.header
            self.debug_pub.publish(debug_msg)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_selected * 255, encoding='mono8')
            mask_msg.header = rgb_msg.header
            self.mask_pub.publish(mask_msg)
        except Exception as exc:
            self.detected_pub.publish(Bool(data=False))
            self.get_logger().error(f'Vision: {exc}', throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BoxYoloeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
