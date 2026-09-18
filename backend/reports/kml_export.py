"""
backend/reports/kml_export.py

Autonomous Dive Route & Tactical Geographic KML Export for SonarVision C2.
Generates standard OGC KML 2.2 XML with:
1. Target Contact Placemarks styled by C2 threat score (Critical/Elevated/Monitor).
2. Autonomous Underwater Vehicle (AUV) tactical inspection route computed via 
   a nearest-neighbor trajectory optimization algorithm.
"""

import math
from typing import List, Dict, Any, Optional
from xml.sax.saxutils import escape
from datetime import datetime


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two WGS84 points in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def sort_points_nearest_neighbor(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes an optimal tactical inspection trajectory through all target contacts
    using a greedy nearest-neighbor algorithm.
    Starts at the contact with the highest threat priority.
    """
    if len(points) <= 1:
        return list(points)

    unvisited = list(points)
    # Sort initially so the starting point is the highest threat contact
    unvisited.sort(key=lambda p: (p.get("threat_score", 0), p.get("confidence", 0)), reverse=True)

    ordered = [unvisited.pop(0)]

    while unvisited:
        curr = ordered[-1]
        c_lat, c_lon = curr["latitude"], curr["longitude"]

        # Find closest unvisited waypoint
        best_idx = 0
        best_dist = float("inf")
        for idx, candidate in enumerate(unvisited):
            dist = haversine_distance(c_lat, c_lon, candidate["latitude"], candidate["longitude"])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        ordered.append(unvisited.pop(best_idx))

    return ordered


def generate_kml_export(
    detections: List[Dict[str, Any]],
    source_filename: Optional[str] = None,
    analysis_id: Optional[str] = None,
    mission_title: Optional[str] = None
) -> str:
    """
    Generates a standard OGC KML 2.2 XML file for Google Earth, QGIS, and tactical ECDIS.
    Includes individual target placemarks and an optimized AUV autonomous dive inspection route.
    """
    title = mission_title or (f"SonarVision Dive Plan - {source_filename}" if source_filename else "SonarVision Tactical Dive Plan")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Filter georeferenced detections
    geo_detections = []
    for idx, det in enumerate(detections):
        geo = det.get("geolocation")
        if hasattr(geo, "latitude") and hasattr(geo, "longitude"):
            lat = getattr(geo, "latitude")
            lon = getattr(geo, "longitude")
        elif isinstance(geo, dict):
            lat = geo.get("latitude")
            lon = geo.get("longitude")
        else:
            lat = None
            lon = None

        if lat is not None and lon is not None:
            d_copy = dict(det)
            d_copy["latitude"] = float(lat)
            d_copy["longitude"] = float(lon)
            d_copy["id"] = det.get("id", idx + 1)
            geo_detections.append(d_copy)

    # Sort contacts using nearest-neighbor trajectory
    route_points = sort_points_nearest_neighbor(geo_detections)

    # Build KML XML document
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">',
        '  <Document>',
        f'    <name>{escape(title)}</name>',
        f'    <description><![CDATA[SonarVision Autonomous Edge Intelligence (SIH26057)<br/>Generated: {timestamp}<br/>Total Target Contacts: {len(geo_detections)}]]></description>',
        '',
        '    <!-- Style Definitions -->',
        '    <Style id="criticalThreat">',
        '      <IconStyle>',
        '        <color>ff0000ff</color> <!-- Red (aabbggrr) -->',
        '        <scale>1.3</scale>',
        '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href></Icon>',
        '      </IconStyle>',
        '      <LabelStyle><color>ff0000ff</color><scale>0.9</scale></LabelStyle>',
        '    </Style>',
        '',
        '    <Style id="elevatedThreat">',
        '      <IconStyle>',
        '        <color>ff00a5ff</color> <!-- Amber (aabbggrr) -->',
        '        <scale>1.1</scale>',
        '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '      </IconStyle>',
        '      <LabelStyle><color>ff00a5ff</color><scale>0.85</scale></LabelStyle>',
        '    </Style>',
        '',
        '    <Style id="monitorThreat">',
        '      <IconStyle>',
        '        <color>ff00ff00</color> <!-- Green (aabbggrr) -->',
        '        <scale>1.0</scale>',
        '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '      </IconStyle>',
        '      <LabelStyle><color>ff00ff00</color><scale>0.8</scale></LabelStyle>',
        '    </Style>',
        '',
        '    <Style id="auvTacticalRoute">',
        '      <LineStyle>',
        '        <color>ffffff00</color> <!-- Cyan/Yellow in aabbggrr -->',
        '        <width>4</width>',
        '      </LineStyle>',
        '      <PolyStyle><color>44ffff00</color></PolyStyle>',
        '    </Style>',
        '',
        '    <!-- Tactical Target Folders -->',
        '    <Folder>',
        '      <name>Classified Acoustic Targets</name>',
    ]

    for order_idx, pt in enumerate(route_points):
        threat = pt.get("threat_score", 0)
        if threat > 75:
            style_url = "#criticalThreat"
            threat_level = "CRITICAL"
            badge_color = "#dc2626"
        elif threat >= 40:
            style_url = "#elevatedThreat"
            threat_level = "ELEVATED"
            badge_color = "#d97706"
        else:
            style_url = "#monitorThreat"
            threat_level = "MONITOR"
            badge_color = "#059669"

        cname = pt.get("class_name", "target")
        obj_type = pt.get("object_type", cname.replace("_", " ").title())
        material = pt.get("material_density", "Unclassified")
        height = pt.get("estimated_height_meters")
        h_str = f"{height:.2f} m" if (height is not None and height > 0) else "Not Mensurated"
        conf = pt.get("confidence", 0.0)
        p95 = pt.get("peak_backscatter_p95")
        p95_str = f"{p95:.1f}/255" if p95 is not None else "N/A"

        placemark_name = f"#{pt['id']:02d} {obj_type} [Threat: {threat}]"

        desc_html = f"""<![CDATA[
        <div style="font-family: monospace, sans-serif; font-size: 12px; color: #0f172a; line-height: 1.5;">
          <h3 style="margin: 0 0 6px 0; color: #0284c7; border-bottom: 2px solid #0284c7; padding-bottom: 3px;">
            Target #{pt['id']:02d} — {obj_type}
          </h3>
          <p style="margin: 4px 0;">
            <b>Tactical Threat Score:</b> 
            <span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold;">
              {threat}/100 ({threat_level})
            </span>
          </p>
          <table style="width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 11px;">
            <tr><td style="padding: 2px 4px; font-weight: bold;">Route Waypoint:</td><td>Step {order_idx + 1} of {len(route_points)}</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Acoustic Material:</td><td>{material}</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Peak Backscatter P95:</td><td>{p95_str}</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Target Relief (Height):</td><td>{h_str}</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Detection Confidence:</td><td>{conf*100:.1f}%</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Latitude:</td><td>{pt['latitude']:.7f}°</td></tr>
            <tr><td style="padding: 2px 4px; font-weight: bold;">Longitude:</td><td>{pt['longitude']:.7f}°</td></tr>
          </table>
        </div>
        ]]>"""

        kml_lines.extend([
            '      <Placemark>',
            f'        <name>{escape(placemark_name)}</name>',
            f'        <styleUrl>{style_url}</styleUrl>',
            f'        <description>{desc_html}</description>',
            '        <Point>',
            f'          <coordinates>{pt["longitude"]:.7f},{pt["latitude"]:.7f},0</coordinates>',
            '        </Point>',
            '      </Placemark>',
        ])

    kml_lines.append('    </Folder>')

    # Autonomous Dive Route (LineString) connecting waypoints in nearest-neighbor sequence
    if len(route_points) >= 2:
        coord_strings = [f"{p['longitude']:.7f},{p['latitude']:.7f},0" for p in route_points]
        coords_joined = " ".join(coord_strings)

        # Calculate total mission path length
        total_dist_m = 0.0
        for i in range(len(route_points) - 1):
            total_dist_m += haversine_distance(
                route_points[i]["latitude"], route_points[i]["longitude"],
                route_points[i+1]["latitude"], route_points[i+1]["longitude"]
            )

        route_desc = f"""<![CDATA[
        <div style="font-family: monospace, sans-serif; font-size: 12px; line-height: 1.5;">
          <h3 style="color: #0284c7; margin: 0 0 6px 0;">Autonomous Underwater Vehicle (AUV) Dive Route</h3>
          <p>Tactical Nearest-Neighbor trajectory visiting {len(route_points)} target contacts.</p>
          <p><b>Total Route Length:</b> {total_dist_m:.1f} meters ({total_dist_m/1852.0:.2f} nautical miles)</p>
          <p><b>Recommended Velocity:</b> 2.5 - 3.5 knots for high-frequency synthetic aperture inspection.</p>
        </div>
        ]]>"""

        kml_lines.extend([
            '    <Placemark>',
            '      <name>AUV Tactical Dive Inspection Route</name>',
            f'      <description>{route_desc}</description>',
            '      <styleUrl>#auvTacticalRoute</styleUrl>',
            '      <LineString>',
            '        <extrude>1</extrude>',
            '        <tessellate>1</tessellate>',
            '        <altitudeMode>clampToGround</altitudeMode>',
            f'        <coordinates>{coords_joined}</coordinates>',
            '      </LineString>',
            '    </Placemark>',
        ])

    kml_lines.extend([
        '  </Document>',
        '</kml>',
    ])

    return "\n".join(kml_lines)
