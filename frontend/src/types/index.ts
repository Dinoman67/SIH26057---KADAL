export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface CenterPixel {
  x: number;
  y: number;
}

export interface Geolocation {
  latitude: number;
  longitude: number;
  crs: string;
  coordinate_source: string;
  utm_x?: number | null;
  utm_y?: number | null;
}

export interface DetectionRecord {
  id: number;
  class_id: number;
  class_name: string;
  object_type?: string | null;
  confidence: number;
  bbox: BoundingBox;
  center_pixel: CenterPixel;
  geolocation?: Geolocation | null;
  material_density?: string | null;
  estimated_height_meters?: number | null;
  target_length_meters?: number | null;
  target_width_meters?: number | null;
  peak_backscatter_p95?: number | null;
  shadow_length_meters?: number | null;
  threat_score?: number | null;
  uncertainty_meters?: number | null;
  uncertainty_method?: string | null;
  review_verdict?: 'confirmed' | 'rejected' | null;
}

export type ReviewVerdict = 'confirmed' | 'rejected';

export type VerdictMap = Record<number, ReviewVerdict>;

export interface FileMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
  file_size_bytes: number;
  file_size_human: string;
  channels: number;
}

export interface GeospatialMetadata {
  georeferenced: boolean;
  crs?: string | null;
  bounds?: {
    left?: number;
    bottom?: number;
    right?: number;
    top?: number;
    wgs84_min_lon?: number;
    wgs84_min_lat?: number;
    wgs84_max_lon?: number;
    wgs84_max_lat?: number;
  } | null;
  pixel_resolution?: [number, number] | null;
  lat_lon_available: boolean;
  coordinate_source?: string | null;
  status_message: string;
  camera_latitude?: number | null;
  camera_longitude?: number | null;
  camera_altitude?: number | null;
  capture_direction?: number | null;
  footprint_geojson?: any | null;
  towfish_altitude_m?: number | null;
  slant_range_corrected?: boolean | null;
}

export interface ModelMetadata {
  model_name: string;
  format: string;
  input_resolution: string;
  execution_provider: string;
  model_path: string;
  sha256_hash: string;
  classes: Record<number, string>;
  architecture: string;
  attention_mechanism: string;
}

export interface AnalysisSummary {
  debris_detected: boolean;
  total_detections: number;
  highest_confidence?: number | null;
  average_confidence?: number | null;
  class_counts: Record<string, number>;
  material_counts?: Record<string, number>;
  inference_time_ms: number;
  total_time_ms: number;
  status: string;
  message: string;
  noise_filtering_active?: boolean;
}

export interface AnalysisResponse {
  analysis_id: string;
  timestamp: string;
  summary: AnalysisSummary;
  detections: DetectionRecord[];
  file_metadata: FileMetadata;
  geospatial_metadata: GeospatialMetadata;
  model_metadata: ModelMetadata;
  original_image_url: string;
  annotated_image_url: string;
  detection_mask_url: string;
  colormap_image_url: string;
  evidence_image_url: string;
  csv_export_url: string;
  json_export_url: string;
  pdf_report_url: string;
  nmea_export_url?: string;
  kml_export_url?: string;
  bundle_export_url?: string;
}

export interface SampleItem {
  id: string;
  name: string;
  type: string;
  description: string;
  has_geolocation: boolean;
  filename: string;
}

// Waterfall simulator (simulated demo transect; does not affect real analysis)
export type WaterfallColormap = 'phosphor' | 'amber' | 'cyan' | 'grayscale';

export type WaterfallThreatLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface WaterfallTarget {
  id: number;
  class_name: string;
  confidence: number;
  y_trigger_px: number;
  bbox: [number, number, number, number];
  latitude: number;
  longitude: number;
  threat_level: WaterfallThreatLevel;
}

export interface WaterfallSurvey {
  survey_id: string;
  title: string;
  strip_image_url: string;
  total_length_meters: number;
  swath_width_meters: number;
  vessel_speed_knots: number;
  ping_rate_hz: number;
  targets: WaterfallTarget[];
}

export interface WaterfallStreamState {
  isPlaying: boolean;
  speedMultiplier: number;
  currentY: number;
  vesselSpeedKts: number;
  colormap: WaterfallColormap;
  activeTargets: WaterfallTarget[];
  lockedTargetId: number | null;
}

const WATERFALL_CLASS_IDS: Record<string, number> = {
  unknown_debris: 0,
  airplane: 1,
  mine: 2,
  wreck: 3,
};

const WATERFALL_THREAT_SCORES: Record<WaterfallThreatLevel, number> = {
  CRITICAL: 90,
  HIGH: 70,
  MEDIUM: 50,
  LOW: 25,
};

export function waterfallTargetToDetection(target: WaterfallTarget): DetectionRecord {
  const class_id = WATERFALL_CLASS_IDS[target.class_name] ?? 0;
  const threat_score = WATERFALL_THREAT_SCORES[target.threat_level] ?? 25;
  const isMetallic = target.class_name !== 'unknown_debris';
  // Relative bbox mapped onto a 512px reference frame for inventory display.
  const [rx, ry, rw, rh] = target.bbox;
  const x1 = Math.round(rx * 512);
  const y1 = Math.round(ry * 512);
  const x2 = Math.round((rx + rw) * 512);
  const y2 = Math.round((ry + rh) * 512);
  return {
    id: target.id,
    class_id,
    class_name: target.class_name,
    object_type: target.class_name,
    confidence: target.confidence,
    bbox: { x1, y1, x2, y2 },
    center_pixel: { x: Math.round((x1 + x2) / 2), y: Math.round((y1 + y2) / 2) },
    geolocation: {
      latitude: target.latitude,
      longitude: target.longitude,
      crs: 'WGS84',
      coordinate_source: 'waterfall-simulated',
    },
    material_density: isMetallic ? 'Hard (Metallic)' : 'Soft (Synthetic/Plastic)',
    estimated_height_meters: null,
    peak_backscatter_p95: null,
    shadow_length_meters: null,
    threat_score,
  };
}
