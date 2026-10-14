/**
 * Browser-side audio helpers: MediaRecorder format selection, WAV encoding and waveform peaks.
 *
 * Recordings are converted to 16 kHz mono 16-bit PCM WAV before upload because the
 * backend decodes with libsndfile (WAV/FLAC/OGG/MP3) and cannot read WebM/Opus.
 */

export const TARGET_SAMPLE_RATE = 16_000;

const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

export function isRecordingSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof window.MediaRecorder !== "undefined"
  );
}

/** First MediaRecorder MIME type the browser supports, or undefined to let the browser choose. */
export function pickMimeType(): string | undefined {
  if (typeof window === "undefined" || typeof window.MediaRecorder === "undefined") return undefined;
  return CANDIDATE_MIME_TYPES.find((type) => window.MediaRecorder.isTypeSupported(type));
}

/** mm:ss[.t] display for the recording timer. */
export function formatTime(seconds: number, tenths = false): string {
  const clamped = Math.max(0, seconds);
  const mm = Math.floor(clamped / 60)
    .toString()
    .padStart(2, "0");
  const ss = Math.floor(clamped % 60)
    .toString()
    .padStart(2, "0");
  if (!tenths) return `${mm}:${ss}`;
  const t = Math.floor((clamped * 10) % 10);
  return `${mm}:${ss}.${t}`;
}

function getAudioContextClass(): typeof AudioContext | undefined {
  if (typeof window === "undefined") return undefined;
  return window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

/** Decode any browser-produced audio blob into an AudioBuffer. */
export async function decodeBlob(blob: Blob): Promise<AudioBuffer> {
  const Ctx = getAudioContextClass();
  if (!Ctx) throw new Error("Web Audio API is not available");
  const context = new Ctx();
  try {
    const data = await blob.arrayBuffer();
    return await context.decodeAudioData(data);
  } finally {
    await context.close();
  }
}

/** Downmix to mono and resample to `sampleRate` using an OfflineAudioContext. */
export async function toMono16k(buffer: AudioBuffer, sampleRate = TARGET_SAMPLE_RATE): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil((buffer.duration * sampleRate)));
  const offline = new OfflineAudioContext(1, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/** Encode mono float samples as a 16-bit PCM WAV blob. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += bytesPerSample;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Cut `samples` to at most `maxSeconds` of audio. Returns the same array when it already fits.
 *
 * The recorder stops on a timer tick at the maximum duration, so the captured take can run a
 * few hundredths of a second over it; the backend rejects anything above its limit, so the
 * decoded samples (not just the timer display) are trimmed before encoding.
 */
export function trimToMaxDuration(samples: Float32Array, sampleRate: number, maxSeconds?: number): Float32Array {
  if (maxSeconds === undefined || !Number.isFinite(maxSeconds) || maxSeconds <= 0) return samples;
  const maxSamples = Math.floor(maxSeconds * sampleRate);
  return samples.length > maxSamples ? samples.subarray(0, maxSamples) : samples;
}

/** Duration in seconds of a 16-bit mono PCM WAV blob, read from its header-independent size. */
export function wavDurationSeconds(wav: Blob, sampleRate: number): number {
  return Math.max(0, wav.size - 44) / 2 / sampleRate;
}

/**
 * Full conversion used by the recorder: any recorded blob -> 16 kHz mono WAV blob, trimmed to
 * `maxSeconds` when given. `duration` is the length of the audio actually encoded.
 */
export async function blobToWav(
  blob: Blob,
  maxSeconds?: number,
): Promise<{ wav: Blob; duration: number; samples: Float32Array }> {
  const decoded = await decodeBlob(blob);
  const samples = trimToMaxDuration(await toMono16k(decoded), TARGET_SAMPLE_RATE, maxSeconds);
  return { wav: encodeWav(samples, TARGET_SAMPLE_RATE), duration: samples.length / TARGET_SAMPLE_RATE, samples };
}

/** Peak envelope (0..1) with `buckets` entries for drawing a static waveform. */
export function computePeaks(samples: Float32Array, buckets = 120): number[] {
  if (samples.length === 0) return [];
  const size = Math.max(1, Math.floor(samples.length / buckets));
  const peaks: number[] = [];
  let max = 0;
  for (let b = 0; b < buckets; b += 1) {
    let peak = 0;
    const start = b * size;
    const end = Math.min(samples.length, start + size);
    for (let i = start; i < end; i += 1) {
      const v = Math.abs(samples[i]);
      if (v > peak) peak = v;
    }
    peaks.push(peak);
    if (peak > max) max = peak;
  }
  return max > 0 ? peaks.map((p) => p / max) : peaks;
}
