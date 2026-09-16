"use client";

import { type KeyboardEvent, useEffect, useRef } from "react";
import { AlertCircle, Mic, Pause, Play, Square, Trash2 } from "lucide-react";

import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { formatTime } from "@/lib/audio";
import { cn } from "@/lib/utils";

export interface AudioRecorderProps {
  /** Called with a 16 kHz mono WAV blob once a valid recording is stopped. */
  onRecordingComplete?: (blob: Blob) => void;
  /** Maximum recording length in seconds; recording stops automatically when reached. */
  maxDuration?: number;
  /** Minimum recording length in seconds; shorter takes are rejected with an error. */
  minDuration?: number;
  disabled?: boolean;
  className?: string;
}

const WAVE_COLOR = "#22c55e"; // green-500
const IDLE_COLOR = "#94a3b8"; // slate-400

/**
 * In-browser microphone recorder.
 *
 * Space toggles recording while the recorder has focus. Buttons carry ARIA labels
 * and the timer/status area is a polite live region for screen readers.
 */
export default function AudioRecorder({
  onRecordingComplete,
  maxDuration = 10,
  minDuration = 1,
  disabled = false,
  className,
}: AudioRecorderProps) {
  const recorder = useAudioRecorder({ onRecordingComplete, maxDuration, minDuration });
  const { isRecording, recordingTime, audioBlob, isPlaying, error, warning, permission, supported } = recorder;
  const canRecord = supported && !disabled && permission !== "denied";
  const nearLimit = isRecording && maxDuration - recordingTime <= 2;

  function handleKeyDown(event: KeyboardEvent<HTMLElement>) {
    const isSpace = event.code === "Space" || event.key === " " || event.key === "Spacebar";
    if (!isSpace || event.target !== event.currentTarget) return;
    event.preventDefault();
    if (isRecording) recorder.stop();
    else if (canRecord) void recorder.start();
  }

  return (
    <section
      role="group"
      aria-label="Audio recorder"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className={cn("card flex w-full flex-col items-center gap-4 text-center focus:outline-none", className)}
    >
      <header className="flex w-full items-center justify-between">
        <h2 className="text-h3">Record your voice</h2>
        <span className="text-xs text-muted-foreground">
          {minDuration}-{maxDuration} s · Space to start/stop
        </span>
      </header>

      <Waveform recorder={recorder} />

      <div aria-live="polite" className="flex flex-col items-center gap-1">
        <p
          className={cn(
            "font-mono text-4xl font-semibold tabular-nums",
            isRecording ? (nearLimit ? "text-amber-600" : "text-red-600") : "text-foreground",
          )}
          aria-label={`Recording time ${formatTime(recordingTime)}`}
        >
          {formatTime(recordingTime, isRecording)}
        </p>
        <p className="text-xs text-muted-foreground">
          {isRecording
            ? "Recording…"
            : audioBlob
              ? `Recorded ${formatTime(recordingTime, true)} · ${Math.round(audioBlob.size / 1024)} KB WAV`
              : permission === "denied"
                ? "Microphone access is blocked"
                : "Ready"}
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        {!isRecording ? (
          <button
            type="button"
            onClick={() => void recorder.start()}
            disabled={!canRecord}
            aria-label="Start recording"
            className="flex h-16 w-16 items-center justify-center rounded-full bg-red-600 text-white shadow-lg transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Mic className="h-7 w-7" aria-hidden />
          </button>
        ) : (
          <button
            type="button"
            onClick={recorder.stop}
            aria-label="Stop recording"
            aria-pressed="true"
            className="flex h-16 w-16 animate-pulse-ring items-center justify-center rounded-full bg-red-600 text-white shadow-lg transition hover:bg-red-700"
          >
            <Square className="h-6 w-6 animate-pulse" aria-hidden />
          </button>
        )}

        {audioBlob && !isRecording ? (
          <>
            <button
              type="button"
              onClick={isPlaying ? recorder.pause : recorder.play}
              aria-label={isPlaying ? "Pause playback" : "Play recording"}
              className="btn-secondary"
            >
              {isPlaying ? <Pause className="h-4 w-4" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={recorder.clear} aria-label="Clear recording" className="btn-secondary">
              <Trash2 className="h-4 w-4" aria-hidden />
              Clear
            </button>
          </>
        ) : null}
      </div>

      {warning && !error ? (
        <p role="status" className="text-sm text-amber-700 dark:text-amber-400">
          {warning}
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="flex items-center gap-2 text-sm text-red-600">
          <AlertCircle className="h-4 w-4 shrink-0" aria-hidden />
          {error}
        </p>
      ) : null}
      {recorder.checked && !supported ? (
        <p className="text-xs text-muted-foreground">
          Your browser doesn&apos;t support audio recording. Use the file upload instead, or try a recent
          Chrome, Edge, Firefox or Safari.
        </p>
      ) : null}
    </section>
  );
}

/** Canvas waveform: live amplitude while recording, static peaks afterwards, flat line when idle. */
function Waveform({ recorder }: { recorder: ReturnType<typeof useAudioRecorder> }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { isRecording, peaks, getWaveform } = recorder;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    context.scale(dpr, dpr);

    const buffer = new Uint8Array(new ArrayBuffer(2048));
    let frame = 0;

    const drawFlat = (color: string) => {
      context.clearRect(0, 0, width, height);
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
      context.stroke();
    };

    const drawPeaks = (values: number[]) => {
      context.clearRect(0, 0, width, height);
      context.fillStyle = WAVE_COLOR;
      const barWidth = width / values.length;
      values.forEach((value, i) => {
        const barHeight = Math.max(2, value * (height - 4));
        context.fillRect(i * barWidth + 1, (height - barHeight) / 2, Math.max(1, barWidth - 2), barHeight);
      });
    };

    const drawLive = () => {
      if (!getWaveform(buffer)) {
        drawFlat(WAVE_COLOR);
        return;
      }
      context.clearRect(0, 0, width, height);
      context.strokeStyle = WAVE_COLOR;
      context.lineWidth = 2;
      context.beginPath();
      const step = width / buffer.length;
      for (let i = 0; i < buffer.length; i += 1) {
        const y = (buffer[i] / 255) * height;
        if (i === 0) context.moveTo(0, y);
        else context.lineTo(i * step, y);
      }
      context.stroke();
      frame = requestAnimationFrame(drawLive);
    };

    if (isRecording) {
      frame = requestAnimationFrame(drawLive);
    } else if (peaks && peaks.length) {
      drawPeaks(peaks);
    } else {
      drawFlat(IDLE_COLOR);
    }
    return () => cancelAnimationFrame(frame);
  }, [isRecording, peaks, getWaveform]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={isRecording ? "Live microphone waveform" : peaks ? "Recorded waveform" : "Waveform (idle)"}
      className="h-24 w-full rounded bg-muted"
    />
  );
}
