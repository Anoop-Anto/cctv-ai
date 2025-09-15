import cv2
import numpy as np
import json
import os

class ZoneIntrusionDetector:
    def __init__(self, zone_config_path="zones/zone_config.json"):
        self.zones = self.load_zones(zone_config_path)

    def load_zones(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def point_in_polygon(self, point, polygon):
        # Ensure point is a tuple of two ints/floats
        if isinstance(point, (np.ndarray, list, tuple)) and len(point) == 2:
            pt = (float(point[0]), float(point[1]))
        else:
            return False
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), pt, False) >= 0

    def detect_intrusions(self, camera_name, tracked_objects):
        intrusions = []
        if camera_name not in self.zones:
            return intrusions

        for obj_id, obj in tracked_objects.items():
            centroid = obj.get("centroid", None)
            if centroid is None or (isinstance(centroid, (np.ndarray, list, tuple)) and len(centroid) == 0):
                continue

            for zone in self.zones[camera_name]:
                if zone["type"] == "restricted":
                    if self.point_in_polygon(tuple(centroid), zone["points"]):
                        intrusions.append({
                            "object_id": obj_id,
                            "zone_id": zone["id"],
                            "centroid": centroid
                        })

        return intrusions

    def draw_zones(self, frame, camera_name):
        if camera_name not in self.zones:
            return frame

        for zone in self.zones[camera_name]:
            color = (0, 0, 255) if zone["type"] == "restricted" else (255, 255, 0)
            cv2.polylines(frame, [np.array(zone["points"], dtype=np.int32)], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, zone["id"], tuple(zone["points"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame
