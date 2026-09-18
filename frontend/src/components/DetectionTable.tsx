import React, { useState } from 'react';
import { Table, Search, Compass, Route } from 'lucide-react';
import type { DetectionRecord } from '../types';

interface DetectionTableProps {
  detections: DetectionRecord[];
  hoveredDetectionId: number | null;
  selectedDetectionId: number | null;
  onHoverDetection: (id: number | null) => void;
  onSelectDetection: (id: number | null) => void;
  analysisId?: string | null;
  nmeaExportUrl?: string | null;
  kmlExportUrl?: string | null;
}

export const DetectionTable: React.FC<DetectionTableProps> = ({
  detections,
  hoveredDetectionId,
  selectedDetectionId,
  onHoverDetection,
  onSelectDetection,
  analysisId,
  nmeaExportUrl,
  kmlExportUrl,
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  // Automatically sort detections descending by threat_score (with confidence tie-breaker)
  const sortedDetections = [...detections].sort((a, b) => {
    const scoreA = a.threat_score ?? 0;
    const scoreB = b.threat_score ?? 0;
    if (scoreB !== scoreA) return scoreB - scoreA;
    return b.confidence - a.confidence;
  });

  const filtered = sortedDetections.filter((det) => {
    const term = searchTerm.toLowerCase();
    const idMatch = det.id.toString().includes(term);
    const classMatch = det.class_name.toLowerCase().includes(term);
    const matMatch = det.material_density?.toLowerCase().includes(term) ?? false;
    const scoreMatch = det.threat_score?.toString().includes(term) ?? false;
    const latMatch = det.geolocation?.latitude?.toString().includes(term) ?? false;
    const lonMatch = det.geolocation?.longitude?.toString().includes(term) ?? false;
    return idMatch || classMatch || matMatch || scoreMatch || latMatch || lonMatch;
  });

  const getThreatBadge = (score: number) => {
    if (score > 75) {
      return {
        label: 'CRITICAL',
        badgeClass: 'bg-red-950/80 text-red-300 border-red-700/80',
        dotClass: 'bg-red-400 animate-pulse',
        scoreClass: 'text-red-400 font-bold',
      };
    }
    if (score >= 40) {
      return {
        label: 'ELEVATED',
        badgeClass: 'bg-amber-950/80 text-amber-300 border-amber-700/80',
        dotClass: 'bg-amber-400',
        scoreClass: 'text-amber-400 font-bold',
      };
    }
    return {
      label: 'MONITOR',
      badgeClass: 'bg-emerald-950/80 text-emerald-300 border-emerald-700/80',
      dotClass: 'bg-emerald-400',
      scoreClass: 'text-emerald-400 font-medium',
    };
  };

  const hasDetections = detections.length > 0;
  const nmeaUrl = nmeaExportUrl || (analysisId ? `/api/export/${analysisId}/nmea` : '/api/export/nmea');
  const kmlUrl = kmlExportUrl || (analysisId ? `/api/export/${analysisId}/kml` : '/api/export/kml');

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded flex flex-col font-mono text-xs overflow-hidden">
      {/* Table Header & Search */}
      <div className="border-b border-slate-800 bg-slate-950/60 p-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Table className="h-4 w-4 text-cyan-400" />
          <span className="font-bold uppercase tracking-wider text-slate-200 font-mono-tech">
            Target Detection & Mensuration Inventory ({detections.length})
          </span>
        </div>

        {/* C2 Operational Export Actions & Search */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Download NMEA Waypoints */}
          <a
            href={hasDetections ? nmeaUrl : '#'}
            download={analysisId ? `waypoints_${analysisId.slice(0, 8)}.txt` : 'waypoints.txt'}
            className={`px-2.5 py-1 rounded text-xs font-bold border inline-flex items-center gap-1.5 transition-all ${
              hasDetections
                ? 'bg-emerald-950/80 border-emerald-500/80 text-emerald-300 hover:bg-emerald-900/60 shadow-[0_0_8px_rgba(16,185,129,0.25)] cursor-pointer'
                : 'bg-slate-950/40 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
            }`}
            title="Download NMEA 0183 $GPWPL Waypoints for Naval ECDIS / GPS"
          >
            <Compass className="h-3.5 w-3.5 text-emerald-400" />
            <span>Download NMEA Waypoints</span>
          </a>

          {/* Download Dive Plan (.KML) */}
          <a
            href={hasDetections ? kmlUrl : '#'}
            download={analysisId ? `dive_plan_${analysisId.slice(0, 8)}.kml` : 'dive_plan.kml'}
            className={`px-2.5 py-1 rounded text-xs font-bold border inline-flex items-center gap-1.5 transition-all ${
              hasDetections
                ? 'bg-cyan-950/80 border-cyan-500/80 text-cyan-300 hover:bg-cyan-900/60 shadow-[0_0_8px_rgba(6,182,212,0.25)] cursor-pointer'
                : 'bg-slate-950/40 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
            }`}
            title="Download Autonomous AUV Dive Route (.KML) for Google Earth / GIS"
          >
            <Route className="h-3.5 w-3.5 text-cyan-400" />
            <span>Download Dive Plan (.KML)</span>
          </a>

          <div className="relative">
            <Search className="h-3.5 w-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filter targets / score..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-2.5 py-1 bg-slate-900 border border-slate-800 rounded text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 text-xs w-44"
            />
          </div>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto max-h-[320px]">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/90 text-slate-400 text-[11px]">
              <th className="py-2 px-3 font-semibold">ID</th>
              <th className="py-2 px-3 font-semibold">THREAT SCORE</th>
              <th className="py-2 px-3 font-semibold">CLASS</th>
              <th className="py-2 px-3 font-semibold">ACOUSTIC MATERIAL</th>
              <th className="py-2 px-3 font-semibold">CONFIDENCE</th>
              <th className="py-2 px-3 font-semibold">EST. HEIGHT</th>
              <th className="py-2 px-3 font-semibold">BOUNDS [X1, Y1, X2, Y2]</th>
              <th className="py-2 px-3 font-semibold">CENTER PIXEL</th>
              <th className="py-2 px-3 font-semibold">LATITUDE</th>
              <th className="py-2 px-3 font-semibold">LONGITUDE</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filtered.length > 0 ? (
              filtered.map((det) => {
                const isHovered = hoveredDetectionId === det.id;
                const isSelected = selectedDetectionId === det.id;
                const isMetallic = det.material_density?.includes('Hard') || det.material_density?.includes('Metallic');
                const threat = det.threat_score ?? 0;
                const { label, badgeClass, dotClass, scoreClass } = getThreatBadge(threat);

                return (
                  <tr
                    key={det.id}
                    onMouseEnter={() => onHoverDetection(det.id)}
                    onMouseLeave={() => onHoverDetection(null)}
                    onClick={() => onSelectDetection(det.id)}
                    className={`cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-cyan-950/60 text-slate-100 border-l-2 border-l-amber-400'
                        : isHovered
                        ? 'bg-slate-800/60 text-slate-200'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-2 px-3 font-bold text-cyan-400">
                      #{det.id.toString().padStart(2, '0')}
                    </td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono inline-flex items-center gap-1.5 border ${badgeClass}`}
                        title={`Tactical Risk Assessment: ${threat}/100 (${label})`}
                      >
                        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
                        <span className={scoreClass}>{threat}</span>
                        <span className="text-[9px] opacity-80 uppercase tracking-tight">[{label}]</span>
                      </span>
                    </td>
                    <td className="py-2 px-3 capitalize font-medium text-slate-200">
                      {det.class_name.replace('_', ' ')}
                    </td>
                    <td className="py-2 px-3">
                      {det.material_density ? (
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold inline-flex items-center gap-1 border ${
                            isMetallic
                              ? 'bg-amber-950/80 text-amber-300 border-amber-700/80'
                              : 'bg-cyan-950/80 text-cyan-300 border-cyan-700/80'
                          }`}
                          title={`Peak Backscatter P95: ${det.peak_backscatter_p95 ?? 'N/A'}/255`}
                        >
                          <span className={`h-1.5 w-1.5 rounded-full ${isMetallic ? 'bg-amber-400 animate-pulse' : 'bg-cyan-400'}`} />
                          {det.material_density}
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      <span className="px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800 text-[11px] font-bold">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      {det.estimated_height_meters !== undefined && det.estimated_height_meters !== null ? (
                        <span className="text-amber-400 font-bold">
                          {det.estimated_height_meters.toFixed(2)} m
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-slate-400">
                      [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}, {det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                    </td>
                    <td className="py-2 px-3 text-slate-400">
                      ({det.center_pixel.x.toFixed(0)}, {det.center_pixel.y.toFixed(0)})
                    </td>
                    <td className="py-2 px-3">
                      {det.geolocation?.latitude !== undefined && det.geolocation?.latitude !== null ? (
                        <span className="text-emerald-400 font-semibold">
                          {det.geolocation.latitude.toFixed(7)}°
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      {det.geolocation?.longitude !== undefined && det.geolocation?.longitude !== null ? (
                        <span className="text-emerald-400 font-semibold">
                          {det.geolocation.longitude.toFixed(7)}°
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={10} className="text-center py-6 text-slate-500">
                  {detections.length === 0 ? 'No objects detected above the confidence threshold.' : 'No matching detections found.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
