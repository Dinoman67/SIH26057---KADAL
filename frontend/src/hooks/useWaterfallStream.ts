import { useCallback, useEffect, useRef, useState } from 'react';
import type { WaterfallColormap, WaterfallStreamState, WaterfallSurvey, WaterfallTarget } from '../types';

interface UseWaterfallStreamOptions {
  survey: WaterfallSurvey | null;
  stripHeightPx?: number;
  baseSpeedPxPerSec?: number;
  lockWindowPx?: number;
  onTargetLocked?: (target: WaterfallTarget) => void;
  onTargetPassed?: (target: WaterfallTarget) => void;
}

export function useWaterfallStream({
  survey,
  stripHeightPx = 2000,
  baseSpeedPxPerSec = 90,
  lockWindowPx = 30,
  onTargetLocked,
  onTargetPassed,
}: UseWaterfallStreamOptions) {
  const [state, setState] = useState<WaterfallStreamState>({
    isPlaying: true,
    speedMultiplier: 1,
    currentY: 0,
    vesselSpeedKts: survey?.vessel_speed_knots ?? 4.0,
    colormap: 'phosphor',
    activeTargets: [],
    lockedTargetId: null,
  });
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  const passedRef = useRef<Set<number>>(new Set());
  const callbacksRef = useRef({ onTargetLocked, onTargetPassed });
  callbacksRef.current = { onTargetLocked, onTargetPassed };

  // Reset when survey changes.
  useEffect(() => {
    passedRef.current = new Set();
    setState((s) => ({
      ...s,
      currentY: 0,
      activeTargets: [],
      lockedTargetId: null,
      vesselSpeedKts: survey?.vessel_speed_knots ?? 4.0,
    }));
  }, [survey?.survey_id]);

  useEffect(() => {
    if (!survey) return;
    const tick = (ts: number) => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const dt = Math.min((ts - lastTsRef.current) / 1000, 0.1);
      lastTsRef.current = ts;
      setState((prev) => {
        if (!prev.isPlaying) return prev;
        const nextY = (prev.currentY + baseSpeedPxPerSec * prev.speedMultiplier * dt) % stripHeightPx;
        let lockedId: number | null = null;
        const newlyPassed: WaterfallTarget[] = [];
        for (const t of survey.targets) {
          const dy = Math.abs(t.y_trigger_px - nextY);
          const wrapped = Math.min(dy, stripHeightPx - dy);
          if (wrapped <= lockWindowPx && lockedId == null) lockedId = t.id;
          // Fire passed once per loop when scanline moves past trigger.
          const prevDy = t.y_trigger_px - prev.currentY;
          const crossed = prevDy >= 0 && prevDy < baseSpeedPxPerSec * prev.speedMultiplier * dt + lockWindowPx;
          if (crossed && !passedRef.current.has(t.id)) {
            passedRef.current.add(t.id);
            newlyPassed.push(t);
          }
        }
        // Reset passed set on wrap so demo can loop.
        if (nextY < prev.currentY) passedRef.current = new Set([...passedRef.current].slice(-8));
        if (lockedId != null) {
          const locked = survey.targets.find((t) => t.id === lockedId);
          if (locked) callbacksRef.current.onTargetLocked?.(locked);
        }
        for (const t of newlyPassed) callbacksRef.current.onTargetPassed?.(t);
        return { ...prev, currentY: nextY, lockedTargetId: lockedId };
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTsRef.current = null;
    };
  }, [survey, stripHeightPx, baseSpeedPxPerSec, lockWindowPx]);

  const play = useCallback(() => setState((s) => ({ ...s, isPlaying: true })), []);
  const pause = useCallback(() => setState((s) => ({ ...s, isPlaying: false })), []);
  const toggle = useCallback(() => setState((s) => ({ ...s, isPlaying: !s.isPlaying })), []);
  const setSpeed = useCallback((mult: number) => {
    setState((s) => ({
      ...s,
      speedMultiplier: mult,
      vesselSpeedKts: Math.round(((survey?.vessel_speed_knots ?? 4.0) * mult + Number.EPSILON) * 10) / 10,
    }));
  }, [survey?.vessel_speed_knots]);
  const reset = useCallback(() => {
    passedRef.current = new Set();
    setState((s) => ({ ...s, currentY: 0, activeTargets: [], lockedTargetId: null }));
  }, []);
  const setColormap = useCallback((colormap: WaterfallColormap) => {
    setState((s) => ({ ...s, colormap }));
  }, []);

  return { state, play, pause, toggle, setSpeed, reset, setColormap };
}
