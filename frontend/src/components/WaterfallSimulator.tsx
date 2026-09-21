import React, { useEffect, useRef, useState } from 'react';
import { Pause, Play, Volume2, VolumeX, Camera, RotateCcw } from 'lucide-react';
import type { WaterfallColormap, WaterfallSurvey, WaterfallTarget } from '../types';
import { useWaterfallStream } from '../hooks/useWaterfallStream';

interface WaterfallSimulatorProps {
  survey: WaterfallSurvey | null;
  onTargetLocked?: (target: WaterfallTarget) => void;
  onTargetPassed?: (target: WaterfallTarget) => void;
}

const SPEED_OPTIONS = [
  { mult: 1, label: '1x', kts: '3 kts' },
  { mult: 2, label: '2x', kts: '6 kts' },
  { mult: 5, label: '5x', kts: '15 kts' },
];

const COLORMAP_OPTIONS: { id: WaterfallColormap; label: string }[] = [
  { id: 'phosphor', label: 'Phosphor' },
  { id: 'amber', label: 'Amber' },
  { id: 'cyan', label: 'Cyan' },
  { id: 'grayscale', label: 'Gray' },
];

const COLORMAP_TINT: Record<WaterfallColormap, string> = {
  phosphor: 'rgba(34, 197, 94, 0.28)',
  amber: 'rgba(245, 158, 11, 0.30)',
  cyan: 'rgba(6, 182, 212, 0.28)',
  grayscale: 'rgba(0, 0, 0, 0)',
};

function threatColor(className: string): string {
  if (className === 'mine') return '#ef4444';
  if (className === 'wreck') return '#f59e0b';
  if (className === 'airplane') return '#38bdf8';
  return '#22d3ee';
}

export const WaterfallSimulator: React.FC<WaterfallSimulatorProps> = ({
  survey,
  onTargetLocked,
  onTargetPassed,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const lastPingRef = useRef<number>(0);
  const { state, toggle, setSpeed, reset, setColormap } = useWaterfallStream({
    survey,
    onTargetLocked,
    onTargetPassed,
  });

  // Load strip image with graceful fallback.
  useEffect(() => {
    setImageReady(false);
    imageRef.current = null;
    if (!survey) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setImageReady(true);
    };
    img.onerror = () => {
      imageRef.current = null;
      setImageReady(false);
    };
    img.src = survey.strip_image_url;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [survey?.strip_image_url]);

  // Sonar ping on lock (user-gesture gated).
  useEffect(() => {
    if (!audioEnabled || state.lockedTargetId == null) return;
    const now = performance.now();
    if (now - lastPingRef.current < 900) return;
    lastPingRef.current = now;
    try {
      if (!audioCtxRef.current) {
        const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioCtxRef.current = new Ctx();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.25);
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.3);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.32);
    } catch {
      // Audio is best-effort; never break the stream.
    }
  }, [audioEnabled, state.lockedTargetId]);

  // Canvas render loop (draw on state change + rAF-friendly).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !survey) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 640;
    const h = canvas.clientHeight || 520;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Background.
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, '#020617');
    bg.addColorStop(1, '#0b1220');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const STRIP_H = 2000;
    const offset = state.currentY % STRIP_H;
    const img = imageRef.current;
    if (img && imageReady) {
      // Tile the strip vertically for a continuous waterfall.
      const scale = w / img.width;
      const drawH = img.height * scale;
      let y = -(offset % drawH);
      // Center strip with side margins like a real swath.
      const margin = Math.round(w * 0.08);
      const swathW = w - margin * 2;
      for (; y < h; y += drawH) {
        ctx.drawImage(img, margin, y, swathW, drawH);
      }
    } else {
      // Fallback synthetic backscatter texture.
      ctx.fillStyle = '#0f172a';
      const margin = Math.round(w * 0.08);
      ctx.fillRect(margin, 0, w - margin * 2, h);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
      ctx.lineWidth = 1;
      for (let y = -Math.round(offset % 24); y < h; y += 24) {
        ctx.beginPath();
        ctx.moveTo(margin, y);
        ctx.lineTo(w - margin, y);
        ctx.stroke();
      }
    }

    // Colormap tint.
    const tint = COLORMAP_TINT[state.colormap];
    if (tint !== 'rgba(0, 0, 0, 0)') {
      ctx.fillStyle = tint;
      ctx.fillRect(0, 0, w, h);
    }

    // Scanline at 35% height with pulse glow.
    const scanY = Math.round(h * 0.35);
    const pulse = 0.55 + 0.25 * Math.sin(performance.now() / 280);
    ctx.save();
    ctx.shadowColor = 'rgba(34, 211, 238, 0.9)';
    ctx.shadowBlur = 14;
    ctx.strokeStyle = `rgba(34, 211, 238, ${pulse.toFixed(2)})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(w, scanY);
    ctx.stroke();
    ctx.restore();

    // Telemetry overlay.
    ctx.fillStyle = 'rgba(2, 6, 23, 0.82)';
    ctx.fillRect(8, scanY - 30, 250, 20);
    ctx.fillStyle = '#67e8f9';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(
      `ACOUSTIC SWEEP [${survey.ping_rate_hz} Hz] • ${state.vesselSpeedKts.toFixed(1)} KTS`,
      14,
      scanY - 16
    );

    // Target reticles near scanline.
    for (const t of survey.targets) {
      const ty = scanY + (t.y_trigger_px - state.currentY) * 0.35;
      if (ty < 40 || ty > h - 30) continue;
      const near = state.lockedTargetId === t.id;
      const color = threatColor(t.class_name);
      const bw = Math.max(46, t.bbox[2] * w);
      const bh = Math.max(30, t.bbox[3] * h * 0.6);
      const cx = Math.min(Math.max(w / 2 + (t.bbox[0] - 0.5) * w, bw / 2 + 6), w - bw / 2 - 6);
      const x = cx - bw / 2;
      const y = ty - bh / 2;
      ctx.save();
      if (near) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = near ? 2.5 : 1.5;
      const L = 10;
      // Corner brackets.
      ctx.beginPath();
      ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
      ctx.moveTo(x + bw - L, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + L);
      ctx.moveTo(x + bw, y + bh - L); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw - L, y + bh);
      ctx.moveTo(x + L, y + bh); ctx.lineTo(x, y + bh); ctx.lineTo(x, y + bh - L);
      ctx.stroke();
      ctx.restore();
      // Tactical tag.
      const label = `[${t.class_name.toUpperCase()}: ${(t.confidence * 100).toFixed(1)}%] [${t.latitude.toFixed(4)}°, ${t.longitude.toFixed(4)}°]`;
      ctx.font = '10px ui-monospace, monospace';
      const tw = ctx.measureText(label).width + 12;
      const tx = Math.min(Math.max(cx - tw / 2, 6), w - tw - 6);
      const tyTag = Math.max(y - 24, 34);
      ctx.fillStyle = 'rgba(2, 6, 23, 0.88)';
      ctx.fillRect(tx, tyTag, tw, 18);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(tx, tyTag, tw, 18);
      ctx.fillStyle = color;
      ctx.fillText(label, tx + 6, tyTag + 13);
    }

    // SIM watermark.
    ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('SIMULATED TRANSECT', w - 140, h - 12);
  });

  const handleSnapshot = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const url = canvas.toDataURL('image/png');
      setSnapshotUrl(url);
      const a = document.createElement('a');
      a.href = url;
      a.download = `waterfall_contact_${Date.now()}.png`;
      a.click();
    } catch {
      setSnapshotUrl(null);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle]);

  if (!survey) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[420px] bg-slate-950 text-slate-500 text-xs font-mono">
        No waterfall survey available.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Control bar — same language as the static toolbar */}
      <div className="border-b border-slate-800 bg-slate-950/60 px-2.5 py-2 flex flex-wrap items-center gap-2 font-mono text-xs">
        <button
          type="button"
          onClick={toggle}
          className="px-2.5 py-1 rounded flex items-center gap-1.5 bg-cyan-500 text-slate-950 font-bold hover:bg-cyan-400"
          title="Play/Pause (Space)"
        >
          {state.isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {state.isPlaying ? 'Pause' : 'Play'}
        </button>
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-0.5 rounded">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s.label}
              type="button"
              onClick={() => setSpeed(s.mult)}
              title={`${s.kts}`}
              className={`px-2 py-1 rounded transition-all ${
                state.speedMultiplier === s.mult
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-0.5 rounded">
          {COLORMAP_OPTIONS.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => setColormap(c.id)}
              className={`px-2 py-1 rounded transition-all ${
                state.colormap === c.id
                  ? 'bg-purple-500 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setAudioEnabled((v) => !v)}
          className={`px-2 py-1 rounded flex items-center gap-1.5 border ${
            audioEnabled
              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
          title="Sonar ping on contact lock"
        >
          {audioEnabled ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
          Ping
        </button>
        <button
          type="button"
          onClick={handleSnapshot}
          className="px-2 py-1 rounded flex items-center gap-1.5 bg-slate-900 border border-slate-800 text-slate-300 hover:border-slate-600"
          title="Snapshot current waterfall frame"
        >
          <Camera className="h-3.5 w-3.5" />
          Snapshot
        </button>
        <button
          type="button"
          onClick={reset}
          className="px-2 py-1 rounded flex items-center gap-1.5 bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200"
          title="Restart transect"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset
        </button>
        <span className="ml-auto text-[11px] text-slate-500">
          {survey.title} • {state.vesselSpeedKts.toFixed(1)} KTS • {survey.ping_rate_hz} Hz
          {snapshotUrl ? ' • snapshot saved' : ''}
        </span>
      </div>
      <div className="relative flex-1 min-h-[420px] bg-slate-950">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      </div>
    </div>
  );
};
