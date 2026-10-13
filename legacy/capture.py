"""
Capture synchronisee LiDAR + camera RGB depuis un vehicule CARLA.
Sauvegarde des paires (image, point cloud) avec calibration associee.
"""
import carla
import numpy as np
import cv2
import json
import queue
import os
import random

# ---------- Config ----------
OUTPUT_DIR = "dataset"
N_FRAMES = 50          # nombre de frames a capturer pour ce premier test
IMG_WIDTH = 800
IMG_HEIGHT = 600
FOV = 90.0
FIXED_DELTA_SECONDS = 0.05  # 20 FPS simulation


def build_projection_matrix(width, height, fov):
    """Matrice intrinseque K a partir de la resolution et du FOV."""
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def sensor_callback(data, sensor_queue, sensor_name):
    sensor_queue.put((data.frame, sensor_name, data))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # --- Mode synchrone ---
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    actor_list = []
    sensor_queue = queue.Queue()

    try:
        # --- Spawn vehicule ---
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        print(f"Vehicule spawn : {vehicle.type_id} a {spawn_point.location}")

        # --- Camera RGB ---
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # --- LiDAR ---
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '200000')
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / FIXED_DELTA_SECONDS))
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)

        # --- Callbacks vers la queue ---
        camera.listen(lambda data: sensor_callback(data, sensor_queue, "camera"))
        lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))

        # --- Calibration (fixe, calculee une fois) ---
        K = build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)
        # Transform LiDAR -> camera (extrinseque relative, car tous deux attaches au vehicule)
        lidar_to_camera = np.array(
            camera_transform.get_inverse_matrix()
        ) @ np.array(lidar_transform.get_matrix())

        calib = {
            "K": K.tolist(),
            "lidar_to_camera": lidar_to_camera.tolist(),
            "img_width": IMG_WIDTH,
            "img_height": IMG_HEIGHT,
            "fov": FOV,
        }
        with open(os.path.join(OUTPUT_DIR, "calib.json"), "w") as f:
            json.dump(calib, f, indent=2)
        print("Calibration sauvegardee dans dataset/calib.json")

        # --- Boucle de capture ---
        frame_count = 0
        while frame_count < N_FRAMES:
            world.tick()

            try:
                # On attend les deux capteurs pour CE frame
                data_frame = {}
                for _ in range(2):
                    frame_id, sensor_name, data = sensor_queue.get(timeout=2.0)
                    data_frame[sensor_name] = data

                frame_dir = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}")
                os.makedirs(frame_dir, exist_ok=True)

                # Sauvegarde image
                img_data = data_frame["camera"]
                img_array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
                img_array = img_array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]  # BGRA -> BGR
                cv2.imwrite(os.path.join(frame_dir, "rgb.png"), img_array)

                # Sauvegarde point cloud
                lidar_data = data_frame["lidar"]
                points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
                points = points.reshape((-1, 4))  # x, y, z, intensity
                np.save(os.path.join(frame_dir, "lidar.npy"), points)

                print(f"Frame {frame_count:04d} capturee : "
                      f"{points.shape[0]} points LiDAR, image {img_array.shape}")
                frame_count += 1

            except queue.Empty:
                print("Timeout en attendant les capteurs, frame ignoree.")
                continue

        print(f"\nCapture terminee : {frame_count} frames dans {OUTPUT_DIR}/")

    finally:
        # --- Nettoyage ---
        print("Nettoyage des acteurs...")
        for actor in actor_list:
            actor.destroy()
        world.apply_settings(original_settings)
        print("Termine.")


if __name__ == '__main__':
    main()