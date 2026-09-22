import axios from 'axios';
import type { AnalysisResponse, ModelMetadata, SampleItem, VerdictMap, WaterfallSurvey } from '../types';

const API_BASE = '/api';

export async function fetchModelInfo(): Promise<ModelMetadata> {
  const res = await axios.get(`${API_BASE}/model-info`);
  return res.data;
}

export async function fetchSamples(): Promise<SampleItem[]> {
  const res = await axios.get(`${API_BASE}/samples`);
  return res.data;
}

export async function analyzeImage(
  file: File,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  useTiling?: boolean
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('confidence_threshold', confThreshold.toString());
  formData.append('iou_threshold', iouThreshold.toString());
  if (useTiling !== undefined) {
    formData.append('use_tiling', useTiling.toString());
  }

  const res = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

export async function analyzeSample(
  sampleId: string,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  useTiling?: boolean
): Promise<AnalysisResponse> {
  const res = await axios.post(`${API_BASE}/analyze-sample`, {
    sample_id: sampleId,
    confidence_threshold: confThreshold,
    iou_threshold: iouThreshold,
    use_tiling: useTiling,
  });
  return res.data;
}

export async function fetchVerdicts(analysisId: string): Promise<VerdictMap> {
  const res = await axios.get(`/api/export/${analysisId}/verdicts`);
  const raw = (res.data?.verdicts ?? {}) as Record<string, string>;
  const out: VerdictMap = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === 'confirmed' || v === 'rejected') out[Number(k)] = v;
  }
  return out;
}

export async function saveVerdicts(analysisId: string, verdicts: VerdictMap): Promise<VerdictMap> {
  const res = await axios.patch(`/api/export/${analysisId}/verdicts`, { verdicts });
  const raw = (res.data?.verdicts ?? {}) as Record<string, string>;
  const out: VerdictMap = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === 'confirmed' || v === 'rejected') out[Number(k)] = v;
  }
  return out;
}

const FALLBACK_WATERFALL_SURVEYS: WaterfallSurvey[] = [
  {
    survey_id: 'noaa-mcm-salvage-transect',
    title: 'NOAA Coastal MCM & Salvage Transect (SIMULATED)',
    strip_image_url: '/static/samples/sample_noaa_debris.png',
    total_length_meters: 1200,
    swath_width_meters: 150,
    vessel_speed_knots: 4.0,
    ping_rate_hz: 15.0,
    targets: [
      {
        id: 1,
        class_name: 'mine',
        confidence: 0.884,
        y_trigger_px: 420,
        bbox: [0.42, 0.21, 0.16, 0.08],
        latitude: 29.7142,
        longitude: -85.1204,
        threat_level: 'CRITICAL',
      },
      {
        id: 2,
        class_name: 'wreck',
        confidence: 0.792,
        y_trigger_px: 1150,
        bbox: [0.3, 0.575, 0.4, 0.12],
        latitude: 29.7188,
        longitude: -85.115,
        threat_level: 'HIGH',
      },
      {
        id: 3,
        class_name: 'unknown_debris',
        confidence: 0.915,
        y_trigger_px: 1680,
        bbox: [0.55, 0.84, 0.2, 0.07],
        latitude: 29.721,
        longitude: -85.1112,
        threat_level: 'MEDIUM',
      },
    ],
  },
];

export async function fetchWaterfallSurveys(): Promise<WaterfallSurvey[]> {
  try {
    const res = await axios.get(`${API_BASE}/waterfall/surveys`);
    if (Array.isArray(res.data) && res.data.length > 0) return res.data as WaterfallSurvey[];
    return FALLBACK_WATERFALL_SURVEYS;
  } catch {
    return FALLBACK_WATERFALL_SURVEYS;
  }
}
