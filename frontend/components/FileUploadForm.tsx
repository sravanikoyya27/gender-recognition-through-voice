"use client";

import {
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { AlertCircle, CheckCircle2, FileAudio, RotateCcw, Upload, UploadCloud, X } from "lucide-react";

import { usePrediction } from "@/hooks/usePrediction";
import {
  ACCEPT_ATTRIBUTE,
  MAX_UPLOAD_BYTES,
  describeType,
  formatBytes,
  formatEta,
  formatSpeed,
  validateAudioFile,
} from "@/lib/upload";
import { cn } from "@/lib/utils";
import { usePredictionStore } from "@/store/predictionStore";
import { useUploadStore } from "@/store/uploadStore";
import type { Prediction } from "@/types/models";

export interface FileUploadFormProps {
  /** Called with the prediction once the upload and analysis succeed. */
  onResult?: (prediction: Prediction) => void;
  /** Disable the form while another prediction (e.g. from the recorder) is running. */
  disabled?: boolean;
  className?: string;
}

/**
 * Drag-and-drop audio upload with validation, upload progress, speed, ETA,
 * cancel and retry. The prediction itself goes through `usePrediction`, so the
 * result also reaches the shared `ResultsDisplay`.
 */
export default function FileUploadForm({ onResult, disabled = false, className }: FileUploadFormProps) {
  const inputId = useId();
  const hintId = `${inputId}-hint`;
  const errorId = `${inputId}-error`;
  const inputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const speedRef = useRef<{ time: number; loaded: number; speed: number }>({ time: 0, loaded: 0, speed: 0 });
  const [dragActive, setDragActive] = useState(false);

  const upload = useUploadStore();
  const {
    selectedFile,
    status,
    isUploading,
    uploadProgress,
    uploadedBytes,
    totalBytes,
    uploadSpeed,
    estimatedTime,
    error,
    retryable,
  } = upload;
  const { predict, isBusy } = usePrediction();
  const blocked = disabled || isUploading || status === "processing";

  // Abort an in-flight upload if the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const handleFileSelect = useCallback(
    async (files: FileList | File[] | null) => {
      const file = files?.[0];
      if (!file) return;
      upload.selectFile(file);
      const result = await validateAudioFile(file);
      if (!result.ok) {
        upload.setError(result.error ?? "Invalid file");
        return;
      }
      upload.setStatus("ready");
    },
    [upload],
  );

  const uploadFile = useCallback(async () => {
    if (!selectedFile || blocked) return;
    const controller = new AbortController();
    abortRef.current = controller;
    speedRef.current = { time: performance.now(), loaded: 0, speed: 0 };
    upload.setError(null);
    upload.setStatus("uploading");
    upload.setProgress(0, selectedFile.size, 0, 0);

    const prediction = await predict(selectedFile, selectedFile.name, {
      signal: controller.signal,
      onUploadProgress: (event) => {
        const now = performance.now();
        const total = event.total ?? selectedFile.size;
        const previous = speedRef.current;
        const dt = (now - previous.time) / 1000;
        let speed = previous.speed;
        if (dt > 0.05) {
          const instant = (event.loaded - previous.loaded) / (1024 * 1024) / dt;
          speed = previous.speed ? previous.speed * 0.6 + instant * 0.4 : instant;
          speedRef.current = { time: now, loaded: event.loaded, speed };
        }
        const remainingMb = Math.max(0, total - event.loaded) / (1024 * 1024);
        const eta = speed > 0 ? remainingMb / speed : 0;
        upload.setProgress(event.loaded, total, speed, eta);
        if (event.loaded >= total) upload.setStatus("processing");
      },
    });
    abortRef.current = null;

    if (prediction) {
      upload.setStatus("done");
      onResult?.(prediction);
      return;
    }
    if (controller.signal.aborted) {
      upload.setStatus("ready");
      upload.setProgress(0, selectedFile.size, 0, 0);
      return;
    }
    const apiError = usePredictionStore.getState().error;
    const message = apiError
      ? apiError.status
        ? `${apiError.message} (HTTP ${apiError.status})`
        : apiError.message
      : "Upload failed. Please try again.";
    upload.setError(message, true);
  }, [selectedFile, blocked, predict, upload, onResult]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleRetry = useCallback(() => {
    upload.setError(null);
    upload.setStatus("ready");
    void uploadFile();
  }, [upload, uploadFile]);

  const handleClear = useCallback(() => {
    abortRef.current?.abort();
    upload.clear();
    if (inputRef.current) inputRef.current.value = "";
  }, [upload]);

  // --- drag and drop --------------------------------------------------------
  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (blocked) return;
    event.dataTransfer.dropEffect = "copy";
    setDragActive(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setDragActive(false);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    if (blocked) return;
    void handleFileSelect(event.dataTransfer.files);
  }

  function handleZoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!blocked) inputRef.current?.click();
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    void handleFileSelect(event.target.files);
  }

  const showProgress = status === "uploading" || status === "processing";

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void uploadFile();
      }}
      aria-describedby={error ? `${hintId} ${errorId}` : hintId}
      className={cn("card flex flex-col gap-4", className)}
    >
      <h2 className="text-h3">Upload an audio file</h2>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={blocked ? -1 : 0}
        aria-disabled={blocked}
        aria-label="Drop an audio file here or press Enter to choose one"
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !blocked && inputRef.current?.click()}
        onKeyDown={handleZoneKeyDown}
        className={cn(
          "flex min-h-[300px] cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-6 text-center transition-colors",
          dragActive
            ? "border-brand bg-brand-50 dark:bg-brand-900/30"
            : "border-border bg-muted/40 hover:border-brand-300 hover:bg-muted",
          blocked && "cursor-not-allowed opacity-60",
        )}
      >
        <UploadCloud className={cn("h-12 w-12", dragActive ? "text-brand" : "text-muted-foreground")} aria-hidden />
        <p className="text-base font-medium">
          {dragActive ? "Release to add the file" : "Drag & drop your audio file here"}
        </p>
        <p className="text-sm text-muted-foreground">or click to browse</p>
        <p id={hintId} className="text-xs text-muted-foreground">
          Supported formats: .wav, .mp3 · up to {formatBytes(MAX_UPLOAD_BYTES)} · 1–10 seconds of speech
        </p>
      </div>

      {/* Fallback input */}
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
        <label htmlFor={inputId} className="btn-secondary w-fit">
          <Upload className="h-4 w-4" aria-hidden />
          Choose file
        </label>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT_ATTRIBUTE}
          onChange={handleInputChange}
          disabled={blocked}
          aria-describedby={hintId}
          className="sr-only"
        />
        <span className="text-xs text-muted-foreground">{selectedFile ? "" : "No file selected"}</span>
      </div>

      {/* File preview */}
      {selectedFile ? (
        <div className="flex items-start gap-3 rounded border border-border bg-card p-3 text-sm">
          <FileAudio className="mt-0.5 h-5 w-5 shrink-0 text-brand" aria-hidden />
          <dl className="grid flex-1 grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-muted-foreground">Name</dt>
            <dd className="truncate font-medium" title={selectedFile.name}>
              {selectedFile.name}
            </dd>
            <dt className="text-muted-foreground">Size</dt>
            <dd>{formatBytes(selectedFile.size)}</dd>
            <dt className="text-muted-foreground">Type</dt>
            <dd>{describeType(selectedFile)}</dd>
          </dl>
          {!isUploading ? (
            <button type="button" onClick={handleClear} aria-label="Remove selected file" className="rounded p-1 hover:bg-muted">
              <X className="h-4 w-4" aria-hidden />
            </button>
          ) : null}
        </div>
      ) : null}

      {/* Progress */}
      {showProgress ? (
        <div className="flex flex-col gap-2" aria-live="polite">
          <div
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={uploadProgress}
            aria-label="Upload progress"
            className="h-3 w-full overflow-hidden rounded-full bg-muted"
          >
            <div
              className="h-full rounded-full bg-green-500 transition-[width] duration-300 ease-out"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <div className="flex flex-wrap justify-between gap-x-4 text-xs text-muted-foreground">
            <span>
              {status === "processing"
                ? "Upload complete · analysing…"
                : `${uploadProgress}% · ${formatBytes(uploadedBytes)} of ${formatBytes(totalBytes)}`}
            </span>
            {status === "uploading" ? (
              <span>
                {formatSpeed(uploadSpeed)} · {estimatedTime > 0 ? `${formatEta(estimatedTime)} remaining` : "estimating…"}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Errors */}
      {error ? (
        <p id={errorId} role="alert" className="flex items-start gap-2 text-sm text-red-600">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>{error}</span>
        </p>
      ) : null}

      {status === "done" ? (
        <p role="status" className="flex items-center gap-2 text-sm text-green-700 dark:text-green-400">
          <CheckCircle2 className="h-4 w-4" aria-hidden />
          Analysis complete. See the result below.
        </p>
      ) : null}

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-3">
        {status === "uploading" || status === "processing" ? (
          <button type="button" onClick={handleCancel} className="btn-secondary" disabled={status === "processing"}>
            <X className="h-4 w-4" aria-hidden />
            Cancel
          </button>
        ) : status === "error" && selectedFile && retryable ? (
          <>
            <button type="button" onClick={handleRetry} className="btn-primary" disabled={disabled || isBusy}>
              <RotateCcw className="h-4 w-4" aria-hidden />
              Retry
            </button>
            <button type="button" onClick={handleClear} className="btn-secondary">
              Choose another file
            </button>
          </>
        ) : status === "done" ? (
          <button type="button" onClick={handleClear} className="btn-primary">
            <Upload className="h-4 w-4" aria-hidden />
            Upload another file
          </button>
        ) : (
          <button type="submit" className="btn-primary" disabled={!selectedFile || status !== "ready" || blocked}>
            <Upload className="h-4 w-4" aria-hidden />
            Predict gender
          </button>
        )}
      </div>
    </form>
  );
}

