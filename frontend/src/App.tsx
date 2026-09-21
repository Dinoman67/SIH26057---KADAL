import { LandingPage } from './components/LandingPage';
import { useState, useEffect } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { ControlBar } from './components/ControlBar';
import { ImageViewer } from './components/ImageViewer';
import { WaterfallSimulator } from './components/WaterfallSimulator';
import { SummaryPanel } from './components/SummaryPanel';
import { MapView } from './components/MapView';
import { DetectionTable } from './components/DetectionTable';
import { MetadataPanel } from './components/MetadataPanel';
import { ReportActions } from './components/ReportActions';

import {
  fetchModelInfo,
  fetchSamples,
  fetchWaterfallSurveys,
  analyzeImage,
  analyzeSample,
} from './services/api';
import type { AnalysisResponse, DetectionRecord, ModelMetadata, SampleItem, WaterfallSurvey, WaterfallTarget } from './types';
import { waterfallTargetToDetection } from './types';
import { AlertTriangle, X } from 'lucide-react';

export function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelMetadata | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active inputs
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Tuning Parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [useTiling, setUseTiling] = useState<boolean>(true);

  // Interactive Highlighting
  const [hoveredDetectionId, setHoveredDetectionId] = useState<number | null>(null);
  const [selectedDetectionId, setSelectedDetectionId] = useState<number | null>(null);

  // Live waterfall (simulated transect; isolated from real analysis/exports)
  const [displayMode, setDisplayMode] = useState<'static' | 'waterfall'>('static');
  const [waterfallSurveys, setWaterfallSurveys] = useState<WaterfallSurvey[]>([]);
  const [activeSurveyId, setActiveSurveyId] = useState<string | null>(null);
  const [simDetections, setSimDetections] = useState<DetectionRecord[]>([]);

  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    setSelectedSampleId(null);
    setError(null);
    setIsAnalyzing(true);

    try {
      const res = await analyzeImage(file, confidenceThreshold, iouThreshold, useTiling);
      setAnalysis(res);
    } catch (err: any) {
      console.error('File analysis failed:', err);
      setError(err.response?.data?.detail || 'Failed to analyze uploaded file.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSampleSelect = async (sampleId: string) => {
    setSelectedSampleId(sampleId);
    setUploadedFile(null);
    setError(null);
    setIsAnalyzing(true);

    try {
      const res = await analyzeSample(sampleId, confidenceThreshold, iouThreshold, useTiling);
      setAnalysis(res);
    } catch (err: any) {
      console.error('Sample analysis failed:', err);
      setError(err.response?.data?.detail || 'Failed to analyze sample image.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReanalyze = async () => {
    if (uploadedFile) {
      await handleFileUpload(uploadedFile);
    } else if (selectedSampleId) {
      await handleSampleSelect(selectedSampleId);
    }
  };

  // Initial load
  useEffect(() => {
    async function init() {
      try {
        const [mInfo, sampleList] = await Promise.all([
          fetchModelInfo(),
          fetchSamples(),
        ]);
        setModelInfo(mInfo);
        setSamples(sampleList);
      } catch (err: any) {
        console.error('Failed initialization:', err);
        setError('Could not connect to backend YOLO-ESI inference service.');
      }
      // Waterfall surveys are optional/demo-only; never block startup.
      try {
        const surveys = await fetchWaterfallSurveys();
        setWaterfallSurveys(surveys);
        if (surveys.length > 0) setActiveSurveyId(surveys[0].survey_id);
      } catch {
        // Fallback already handled inside fetchWaterfallSurveys.
      }
    }
    init();
  }, []);

  const activeSurvey = waterfallSurveys.find((s) => s.survey_id === activeSurveyId) ?? waterfallSurveys[0] ?? null;

  const handleWaterfallPassed = (target: WaterfallTarget) => {
    setSimDetections((prev) => {
      if (prev.some((d) => d.id === target.id)) return prev;
      return [...prev, waterfallTargetToDetection(target)];
    });
  };

  const handleWaterfallLocked = (target: WaterfallTarget) => {
    setSelectedDetectionId(target.id);
  };

  const handleDisplayModeChange = (mode: 'static' | 'waterfall') => {
    setDisplayMode(mode);
    if (mode === 'waterfall') setSimDetections([]);
  };

  const visibleDetections = displayMode === 'waterfall' ? simDetections : analysis?.detections ?? [];

  if (showLanding) {
    return <LandingPage onLaunch={() => setShowLanding(false)} />;
  }

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-200 flex flex-col selection:bg-cyan-500 selection:text-black">
      {/* Error Alert Bar */}
      {error && (
        <div className="max-w-[1800px] w-full mx-auto px-6 pt-3">
          <div className="bg-red-950/80 border border-red-500/60 rounded p-3 text-red-200 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
              <span>{error}</span>
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-200"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Mission Workspace Grid */}
      <main className="max-w-[1800px] w-full mx-auto p-4 md:p-6 flex-1 flex flex-col gap-4">
        {/* Top Control & Tuning Bar */}
        <ControlBar
          confidenceThreshold={confidenceThreshold}
          setConfidenceThreshold={setConfidenceThreshold}
          iouThreshold={iouThreshold}
          setIouThreshold={setIouThreshold}
          useTiling={useTiling}
          setUseTiling={setUseTiling}
          onReanalyze={handleReanalyze}
          isAnalyzing={isAnalyzing}
          canReanalyze={Boolean(uploadedFile || selectedSampleId)}
        />

        {/* Primary Multi-Panel Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* Left Column: Upload & Datasets */}
          <div className="lg:col-span-3 flex flex-col gap-4">
            <UploadPanel
              onFileUpload={handleFileUpload}
              onSampleSelect={handleSampleSelect}
              samples={samples}
              isAnalyzing={isAnalyzing}
              selectedSampleId={selectedSampleId}
            />

            <ReportActions analysis={analysis} />
          </div>

          {/* Center Column: High-Res Imagery Workspace */}
          <div className="lg:col-span-6 flex flex-col gap-4">
            <ImageViewer
              analysis={analysis}
              hoveredDetectionId={hoveredDetectionId}
              selectedDetectionId={selectedDetectionId}
              onSelectDetection={setSelectedDetectionId}
              isAnalyzing={isAnalyzing}
              displayMode={displayMode}
              onDisplayModeChange={handleDisplayModeChange}
              waterfallSlot={
                <WaterfallSimulator
                  survey={activeSurvey}
                  onTargetLocked={handleWaterfallLocked}
                  onTargetPassed={handleWaterfallPassed}
                />
              }
            />

            {/* Target Inventory Table below Image */}
            <DetectionTable
              detections={visibleDetections}
              hoveredDetectionId={hoveredDetectionId}
              selectedDetectionId={selectedDetectionId}
              onHoverDetection={setHoveredDetectionId}
              onSelectDetection={setSelectedDetectionId}
              analysisId={analysis?.analysis_id}
              nmeaExportUrl={analysis?.nmea_export_url}
              kmlExportUrl={analysis?.kml_export_url}
            />
          </div>

          {/* Right Column: Summary KPI & Cartographic Map */}
          <div className="lg:col-span-3 flex flex-col gap-4">
            <SummaryPanel analysis={analysis} />

            <MapView
              analysis={analysis}
              selectedDetectionId={selectedDetectionId}
              onSelectDetection={setSelectedDetectionId}
              simDetections={displayMode === 'waterfall' ? simDetections : null}
              simMode={displayMode === 'waterfall'}
            />

            <MetadataPanel
              fileMeta={analysis?.file_metadata || null}
              geoMeta={analysis?.geospatial_metadata || null}
              modelMeta={analysis?.model_metadata || modelInfo}
            />
          </div>
        </div>
      </main>

      {/* Tactical Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 px-6 py-2.5 text-slate-500 text-[11px] font-mono flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span>YOLO-ESI Debris Intelligence v1.0</span>
          <span>•</span>
          <span>Ultralytics ONNX FP16 Pipeline</span>
          <span>•</span>
          <span>NOAA H11833 Side-Scan Sonar Benchmark</span>
        </div>
        <div className="flex items-center gap-2">
          <span>Model Status:</span>
          <span className="text-emerald-400 font-semibold">IMMUTABLE (VERIFIED)</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
