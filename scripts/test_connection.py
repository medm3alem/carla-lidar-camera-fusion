"""
Test de connexion basique au serveur CARLA.
A lancer une fois CarlaUE4.exe demarre et la scene chargee.
"""
import carla
import sys

def main():
    print("Tentative de connexion au serveur CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        version = client.get_server_version()
        print(f"Connecte ! Version serveur CARLA : {version}")
    except RuntimeError as e:
        print(f"Echec de connexion : {e}")
        print("Verifie que CarlaUE4.exe est bien lance et que la scene est chargee.")
        sys.exit(1)

    world = client.get_world()
    print(f"Map actuelle : {world.get_map().name}")

    blueprint_library = world.get_blueprint_library()
    n_vehicles = len(blueprint_library.filter('vehicle.*'))
    n_sensors = len(blueprint_library.filter('sensor.*'))
    print(f"Blueprints disponibles : {n_vehicles} vehicules, {n_sensors} capteurs")

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    print(f"LiDAR blueprint trouve : {lidar_bp.id}")
    print(f"Camera blueprint trouvee : {camera_bp.id}")

    print("\nTout fonctionne. Pret pour la suite.")

if __name__ == '__main__':
    main()