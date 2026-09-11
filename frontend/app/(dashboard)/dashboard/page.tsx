"use client";

import AudioRecorder from "@/components/AudioRecorder";
import FileUploadForm from "@/components/FileUploadForm";
import ResultsDisplay from "@/components/ResultsDisplay";
import { usePrediction } from "@/hooks/usePrediction";

export default function DashboardPage() {
  const { status, result, error, isBusy, predict } = usePrediction();

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-h1">Predict</h1>
        <p className="text-sm text-muted-foreground">
          Provide a voice sample and the neural network will estimate the speaker&apos;s gender.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <AudioRecorder onRecordingComplete={(blob) => void predict(blob, "recording.webm")} disabled={isBusy} />
        <FileUploadForm onSubmit={(file) => predict(file)} busy={isBusy} />
      </div>

      <ResultsDisplay status={status} result={result} error={error} />
    </div>
  );
}
