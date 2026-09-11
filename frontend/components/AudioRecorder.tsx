"use client";

import { Mic } from "lucide-react";

import { cn } from "@/lib/utils";

export interface AudioRecorderProps {
  /** Called with the recorded audio once the user stops recording. */
  onRecordingComplete?: (blob: Blob) => void;
  /** Maximum recording length in seconds. */
  maxDurationSeconds?: number;
  disabled?: boolean;
  className?: string;
}

/**
 * In-browser microphone recorder.
 *
 * The live recording behaviour (react-mic waveform, start/stop, duration cap)
 * is wired in the recording phase. This scaffold fixes the component contract
 * so the dashboard layout and the prediction flow can be built against it.
 */
export default function AudioRecorder({
  maxDurationSeconds = 10,
  disabled = false,
  className,
}: AudioRecorderProps) {
  return (
    <section className={cn("card flex flex-col items-center gap-4 text-center", className)}>
      <h2 className="text-h3">Record your voice</h2>
      <button
        type="button"
        disabled
        aria-disabled
        title="Recording is enabled in the recording phase"
        className="flex h-20 w-20 items-center justify-center rounded-full bg-muted text-muted-foreground"
      >
        <Mic className="h-8 w-8" aria-hidden />
        <span className="sr-only">Start recording</span>
      </button>
      <p className="text-sm text-muted-foreground">
        Up to {maxDurationSeconds} seconds. {disabled ? "Recording is currently unavailable." : null}
        In-browser recording arrives in the recording phase; use the file upload for now.
      </p>
    </section>
  );
}
