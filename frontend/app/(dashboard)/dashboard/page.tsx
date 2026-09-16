"use client";

import Link from "next/link";
import { LogIn } from "lucide-react";
import { useCallback } from "react";

import AudioRecorder from "@/components/AudioRecorder";
import FileUploadForm from "@/components/FileUploadForm";
import ResultsDisplay from "@/components/ResultsDisplay";
import { useAuth } from "@/hooks/useAuth";
import { usePrediction } from "@/hooks/usePrediction";
import { useRecorderStore } from "@/store/recorderStore";
import { useUploadStore } from "@/store/uploadStore";

export default function DashboardPage() {
  const { status, result, error, isBusy, predict, reset } = usePrediction();
  const { isAuthenticated, hasHydrated } = useAuth();
  // Predictions require a JWT (backend PREDICTIONS_REQUIRE_AUTH); keep the controls off until
  // the persisted session has been read, and while signed out.
  const signedOut = hasHydrated && !isAuthenticated;
  const controlsDisabled = isBusy || !hasHydrated || signedOut;
  const recorderAudioUrl = useRecorderStore((s) => s.audioUrl);
  const resetRecorder = useRecorderStore((s) => s.reset);
  const uploadedFile = useUploadStore((s) => s.selectedFile);
  const clearUpload = useUploadStore((s) => s.clear);

  const handleTryAnother = useCallback(() => {
    reset();
    resetRecorder();
    clearUpload();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [reset, resetRecorder, clearUpload]);

  // The analysed audio is still in the browser: either the recorder's WAV or the chosen file.
  const audioSource = result?.audio_filename === "recording.wav" ? recorderAudioUrl : uploadedFile;
  const handleDownload = audioSource
    ? () => {
        const url = typeof audioSource === "string" ? audioSource : URL.createObjectURL(audioSource);
        const a = document.createElement("a");
        a.href = url;
        a.download = result?.audio_filename ?? "recording.wav";
        document.body.appendChild(a);
        a.click();
        a.remove();
        if (typeof audioSource !== "string") setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    : undefined;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-h1">Predict</h1>
        <p className="text-sm text-muted-foreground">
          Provide a voice sample and the neural network will estimate the speaker&apos;s gender.
        </p>
      </header>

      {signedOut ? (
        <div role="status" className="card flex flex-wrap items-center justify-between gap-3 border-brand-200 dark:border-brand-800">
          <p className="text-sm">
            Predictions are stored in your account, so the recorder and upload need a signed-in user.
          </p>
          <div className="flex gap-2">
            <Link href="/login?next=/dashboard" className="btn-primary">
              <LogIn className="h-4 w-4" aria-hidden />
              Sign in
            </Link>
            <Link href="/register" className="btn-secondary">
              Create account
            </Link>
          </div>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <AudioRecorder
          onRecordingComplete={(blob) => void predict(blob, "recording.wav")}
          maxDuration={10}
          minDuration={1}
          disabled={controlsDisabled}
        />
        <FileUploadForm disabled={controlsDisabled} />
      </div>

      <ResultsDisplay
        status={status}
        result={result}
        error={error}
        onTryAnother={handleTryAnother}
        onDownload={handleDownload}
      />
    </div>
  );
}
