/** Upload validation and formatting helpers for the file upload form. */

import { formatBytes } from "@/lib/utils";

export { formatBytes };

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

/** Accepted formats: extension -> MIME types browsers report for it. */
export const ACCEPTED_FORMATS: Record<string, string[]> = {
  ".wav": ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"],
  ".mp3": ["audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mpeg-3"],
};
export const ACCEPTED_EXTENSIONS = Object.keys(ACCEPTED_FORMATS);
export const ACCEPT_ATTRIBUTE = [...ACCEPTED_EXTENSIONS, ...Object.values(ACCEPTED_FORMATS).flat()].join(",");

export type DetectedFormat = "wav" | "mp3" | "unknown";

export interface ValidationResult {
  ok: boolean;
  error?: string;
  extension: string;
  detected: DetectedFormat;
}

export function extensionOf(name: string): string {
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

/** Read the first bytes of a file with FileReader (the "FileReader + type checking" step). */
export function readHeader(file: File, length = 12): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the file"));
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.readAsArrayBuffer(file.slice(0, length));
  });
}

/** Detect WAV (RIFF....WAVE) or MP3 (ID3 tag or MPEG frame sync) from the header bytes. */
export function detectFormat(header: Uint8Array): DetectedFormat {
  const ascii = (start: number, end: number) => String.fromCharCode(...header.slice(start, end));
  if (header.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WAVE") return "wav";
  if (header.length >= 3 && ascii(0, 3) === "ID3") return "mp3";
  if (header.length >= 2 && header[0] === 0xff && (header[1] & 0xe0) === 0xe0) return "mp3";
  return "unknown";
}

/** Extension, MIME and content checks. Returns a specific message when the file is rejected. */
export async function validateAudioFile(file: File): Promise<ValidationResult> {
  const extension = extensionOf(file.name);
  const detectedFallback: DetectedFormat = "unknown";

  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return {
      ok: false,
      extension,
      detected: detectedFallback,
      error: `Unsupported file type "${extension || "none"}". Please choose a .wav or .mp3 file.`,
    };
  }
  if (file.size === 0) {
    return { ok: false, extension, detected: detectedFallback, error: "The selected file is empty." };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      extension,
      detected: detectedFallback,
      error: `File is too large (${formatBytes(file.size)}). The maximum size is 10 MB.`,
    };
  }
  const mime = (file.type || "").toLowerCase();
  const allowedMimes = ACCEPTED_FORMATS[extension];
  if (mime && mime !== "application/octet-stream" && !allowedMimes.includes(mime)) {
    return {
      ok: false,
      extension,
      detected: detectedFallback,
      error: `The file's type "${mime}" does not match its ${extension} extension.`,
    };
  }

  let detected: DetectedFormat = "unknown";
  try {
    detected = detectFormat(await readHeader(file));
  } catch {
    return { ok: false, extension, detected, error: "The file could not be read. Please try another file." };
  }
  const expected: DetectedFormat = extension === ".wav" ? "wav" : "mp3";
  if (detected !== expected) {
    return {
      ok: false,
      extension,
      detected,
      error: `The file does not look like a valid ${expected.toUpperCase()} recording.`,
    };
  }
  return { ok: true, extension, detected };
}

export function formatSpeed(mbPerSecond: number): string {
  if (!Number.isFinite(mbPerSecond) || mbPerSecond <= 0) return "–";
  return mbPerSecond >= 1 ? `${mbPerSecond.toFixed(1)} MB/s` : `${(mbPerSecond * 1024).toFixed(0)} KB/s`;
}

export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  if (seconds < 1) return "less than a second";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.ceil(seconds % 60)} s`;
}

/** Describe the file type for the preview (MIME when present, otherwise from the extension). */
export function describeType(file: File): string {
  if (file.type) return file.type;
  const ext = extensionOf(file.name);
  return ext === ".wav" ? "audio/wav" : ext === ".mp3" ? "audio/mpeg" : "unknown";
}
