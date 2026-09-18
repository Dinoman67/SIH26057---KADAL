import React from 'react';
import { Download, FileText, FileSpreadsheet, Code, Image as ImageIcon, Compass, Route } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface ReportActionsProps {
  analysis: AnalysisResponse | null;
}

export const ReportActions: React.FC<ReportActionsProps> = ({ analysis }) => {
  const isAvailable = Boolean(analysis);
  const analysisId = analysis?.analysis_id;
  const nmeaUrl = analysis?.nmea_export_url || (analysisId ? `/api/export/${analysisId}/nmea` : '/api/export/nmea');
  const kmlUrl = analysis?.kml_export_url || (analysisId ? `/api/export/${analysisId}/kml` : '/api/export/kml');

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded p-3 flex flex-col gap-2 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="font-bold uppercase tracking-wider text-slate-200 font-mono-tech flex items-center gap-1.5">
          <Download className="h-4 w-4 text-cyan-400" />
          Mission Reports & C2 Naval Export
        </span>
        <span className="text-[10px] text-slate-500">PDF • CSV • NMEA • KML</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {/* PDF Report */}
        <a
          href={analysis ? analysis.pdf_report_url : '#'}
          download={analysis ? `debris_report_${analysis.analysis_id.slice(0, 8)}.pdf` : undefined}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-cyan-500/50 hover:bg-cyan-950/40 text-cyan-300 hover:border-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <FileText className="h-4 w-4 text-cyan-400" />
          <span className="font-bold text-[10px]">PDF Report</span>
          <span className="text-[8px] text-slate-400">Formal Brief</span>
        </a>

        {/* CSV Data */}
        <a
          href={analysis ? analysis.csv_export_url : '#'}
          download={analysis ? `detections_${analysis.analysis_id.slice(0, 8)}.csv` : undefined}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-emerald-500/50 hover:bg-emerald-950/40 text-emerald-300 hover:border-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <FileSpreadsheet className="h-4 w-4 text-emerald-400" />
          <span className="font-bold text-[10px]">CSV Table</span>
          <span className="text-[8px] text-slate-400">Hydro Table</span>
        </a>

        {/* NMEA 0183 Waypoints */}
        <a
          href={isAvailable ? nmeaUrl : '#'}
          download={analysis ? `waypoints_${analysis.analysis_id.slice(0, 8)}.txt` : 'waypoints.txt'}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-emerald-500/50 hover:bg-emerald-950/40 text-emerald-300 hover:border-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <Compass className="h-4 w-4 text-emerald-400" />
          <span className="font-bold text-[10px]">NMEA 0183</span>
          <span className="text-[8px] text-slate-400">$GPWPL Bridge</span>
        </a>

        {/* KML Autonomous Dive Route */}
        <a
          href={isAvailable ? kmlUrl : '#'}
          download={analysis ? `dive_plan_${analysis.analysis_id.slice(0, 8)}.kml` : 'dive_plan.kml'}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-cyan-500/50 hover:bg-cyan-950/40 text-cyan-300 hover:border-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <Route className="h-4 w-4 text-cyan-400" />
          <span className="font-bold text-[10px]">Dive Plan</span>
          <span className="text-[8px] text-slate-400">KML Route</span>
        </a>

        {/* JSON Schema */}
        <a
          href={analysis ? analysis.json_export_url : '#'}
          download={analysis ? `results_${analysis.analysis_id.slice(0, 8)}.json` : undefined}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-amber-500/50 hover:bg-amber-950/40 text-amber-300 hover:border-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <Code className="h-4 w-4 text-amber-400" />
          <span className="font-bold text-[10px]">JSON Schema</span>
          <span className="text-[8px] text-slate-400">Raw Telemetry</span>
        </a>

        {/* Annotated Image */}
        <a
          href={analysis ? analysis.annotated_image_url : '#'}
          download={analysis ? `annotated_${analysis.analysis_id.slice(0, 8)}.png` : undefined}
          className={`p-2 rounded border flex flex-col items-center justify-center gap-1 text-center transition-all ${
            isAvailable
              ? 'bg-slate-950/80 border-indigo-500/50 hover:bg-indigo-950/40 text-indigo-300 hover:border-indigo-400 shadow-[0_0_10px_rgba(99,102,241,0.15)] cursor-pointer'
              : 'bg-slate-950/30 border-slate-800 text-slate-600 cursor-not-allowed pointer-events-none'
          }`}
        >
          <ImageIcon className="h-4 w-4 text-indigo-400" />
          <span className="font-bold text-[10px]">Annotated PNG</span>
          <span className="text-[8px] text-slate-400">High-Res Render</span>
        </a>
      </div>
    </div>
  );
};
