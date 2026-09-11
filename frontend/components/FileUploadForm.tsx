"use client";

import { type ChangeEvent, type FormEvent, useId, useState } from "react";
import { Upload } from "lucide-react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { cn, formatBytes } from "@/lib/utils";

const ACCEPTED_EXTENSIONS = [".wav", ".webm", ".mp3", ".ogg", ".flac"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

export interface FileUploadFormProps {
  onSubmit: (file: File) => void | Promise<unknown>;
  busy?: boolean;
  className?: string;
}

export default function FileUploadForm({ onSubmit, busy = false, className }: FileUploadFormProps) {
  const inputId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setValidationError(null);
    if (!selected) {
      setFile(null);
      return;
    }
    const ext = selected.name.slice(selected.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setValidationError(`Unsupported format. Use ${ACCEPTED_EXTENSIONS.join(", ")}.`);
      setFile(null);
      return;
    }
    if (selected.size > MAX_SIZE_BYTES) {
      setValidationError(`File is ${formatBytes(selected.size)}; the limit is 10 MB.`);
      setFile(null);
      return;
    }
    setFile(selected);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;
    await onSubmit(file);
  }

  return (
    <form onSubmit={handleSubmit} className={cn("card flex flex-col gap-4", className)}>
      <h2 className="text-h3">Upload an audio file</h2>

      <div>
        <label htmlFor={inputId} className="label">
          Audio file
        </label>
        <input
          id={inputId}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          onChange={handleChange}
          disabled={busy}
          className="input file:mr-3 file:rounded file:border-0 file:bg-muted file:px-3 file:py-1 file:text-sm"
        />
        <p className="mt-1 text-xs text-muted-foreground">
          WAV recommended. {ACCEPTED_EXTENSIONS.join(", ")} up to 10 MB.
        </p>
      </div>

      {file ? (
        <p className="text-sm">
          Selected <span className="font-medium">{file.name}</span> ({formatBytes(file.size)})
        </p>
      ) : null}

      {validationError ? (
        <p role="alert" className="text-sm text-red-600">
          {validationError}
        </p>
      ) : null}

      <button type="submit" disabled={!file || busy} className="btn-primary self-start">
        {busy ? <LoadingSpinner size="sm" /> : <Upload className="h-4 w-4" aria-hidden />}
        {busy ? "Analyzing..." : "Predict gender"}
      </button>
    </form>
  );
}
