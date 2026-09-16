"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { blobToWav, computePeaks, isRecordingSupported, pickMimeType } from "@/lib/audio";
import { useRecorderStore } from "@/store/recorderStore";

export const ERRORS = {
  unsupported: "Your browser doesn't support audio recording.",
  denied: "Permission denied. Please allow microphone access.",
  noDevice: "No microphone was found. Please connect one and try again.",
  failed: "Failed to record audio. Please try again.",
  tooShort: (min: number) => `Recording too short. Please record at least ${min} second${min === 1 ? "" : "s"}.`,
};

const TIMER_INTERVAL_MS = 100;
const WARN_BEFORE_MAX_SECONDS = 2;

export interface UseAudioRecorderOptions {
  onRecordingComplete?: (blob: Blob) => void;
  maxDuration?: number;
  minDuration?: number;
}

/**
 * MediaRecorder + Web Audio recorder with a Zustand-backed state.
 *
 * - start(): asks for microphone permission, records, auto-stops at `maxDuration`
 * - stop(): enforces `minDuration`, converts the take to 16 kHz mono WAV, calls onRecordingComplete
 * - play()/pause(): playback of the finished recording
 * - clear(): discards the recording
 * - getWaveform(): fills a Uint8Array with live time-domain samples for the canvas
 */
export function useAudioRecorder({
  onRecordingComplete,
  maxDuration = 10,
  minDuration = 1,
}: UseAudioRecorderOptions = {}) {
  const store = useRecorderStore();
  const {
    isRecording,
    recordingTime,
    audioBlob,
    audioUrl,
    isPlaying,
    error,
    warning,
    permission,
    peaks,
    setRecording,
    setRecordingTime,
    setRecordingResult,
    setPlaying,
    setError,
    setWarning,
    setPermission,
    reset,
  } = store;

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const discardRef = useRef(false);
  const completeRef = useRef(onRecordingComplete);
  completeRef.current = onRecordingComplete;

  // Resolved after mount so server and first client render agree (no window during SSR).
  const [supported, setSupported] = useState(false);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    setSupported(isRecordingSupported());
    setChecked(true);
  }, []);

  // Report the permission state up front where the Permissions API exists (Chrome/Edge).
  useEffect(() => {
    if (!checked) return;
    if (!supported) {
      setPermission("unsupported");
      return;
    }
    let cancelled = false;
    navigator.permissions
      ?.query({ name: "microphone" as PermissionName })
      .then((status) => {
        if (cancelled) return;
        setPermission(status.state as "granted" | "denied" | "prompt");
        status.onchange = () => setPermission(status.state as "granted" | "denied" | "prompt");
      })
      .catch(() => {
        /* Permissions API unavailable (Firefox/Safari): state is learned on first use */
      });
    return () => {
      cancelled = true;
    };
  }, [checked, supported, setPermission]);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const releaseStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    analyserRef.current = null;
    if (contextRef.current) {
      void contextRef.current.close().catch(() => undefined);
      contextRef.current = null;
    }
  }, []);

  const finalize = useCallback(
    async (mimeType: string) => {
      const raw = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
      chunksRef.current = [];
      if (discardRef.current || raw.size === 0) {
        discardRef.current = false;
        return;
      }
      try {
        const { wav, samples, duration } = await blobToWav(raw);
        // Enforce the minimum on the audio actually captured, not on the timer:
        // capture can start later than the start() call (device latency).
        if (duration < minDuration) {
          setRecordingResult(null, null, null);
          setRecordingTime(0);
          setError(ERRORS.tooShort(minDuration));
          return;
        }
        setRecordingTime(duration);
        setRecordingResult(wav, URL.createObjectURL(wav), computePeaks(samples));
        completeRef.current?.(wav);
      } catch (err) {
        console.error("WAV conversion failed", err);
        setError(ERRORS.failed);
      }
    },
    [minDuration, setRecordingResult, setRecordingTime, setError],
  );

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    const elapsed = (performance.now() - startedAtRef.current) / 1000;
    stopTimer();
    if (elapsed < minDuration) {
      discardRef.current = true;
      setError(ERRORS.tooShort(minDuration));
      setRecordingTime(0);
    }
    setRecording(false);
    setWarning(null);
    try {
      // onstop -> finalize() -> releaseStream(): the tracks stay open until the encoder
      // has flushed its last chunk, otherwise the tail of the take is lost.
      recorder.stop();
    } catch {
      setError(ERRORS.failed);
      releaseStream();
    }
  }, [minDuration, releaseStream, setError, setRecording, setRecordingTime, setWarning, stopTimer]);

  const stopRef = useRef(stop);
  stopRef.current = stop;

  const start = useCallback(async () => {
    if (!supported) {
      setPermission("unsupported");
      setError(ERRORS.unsupported);
      return;
    }
    if (recorderRef.current && recorderRef.current.state !== "inactive") return;

    setError(null);
    setWarning(null);
    reset();

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      setPermission("granted");
    } catch (err) {
      const name = (err as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        setPermission("denied");
        setError(ERRORS.denied);
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setError(ERRORS.noDevice);
      } else {
        setError(ERRORS.failed);
      }
      return;
    }

    try {
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const context = new Ctx();
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.6;
      context.createMediaStreamSource(stream).connect(analyser);
      contextRef.current = context;
      analyserRef.current = analyser;
      streamRef.current = stream;

      const mimeType = pickMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        releaseStream();
        void finalize(recorder.mimeType);
      };
      recorder.onerror = () => {
        setError(ERRORS.failed);
        stopRef.current();
      };
      recorder.onstart = () => {
        startedAtRef.current = performance.now(); // capture has actually begun
      };
      recorderRef.current = recorder;

      startedAtRef.current = performance.now();
      recorder.start(TIMER_INTERVAL_MS); // gather chunks every 100 ms
      setRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        const elapsed = (performance.now() - startedAtRef.current) / 1000;
        setRecordingTime(Math.min(elapsed, maxDuration));
        const remaining = maxDuration - elapsed;
        if (remaining <= 0) {
          stopRef.current(); // auto-stop at the maximum duration
        } else if (remaining <= WARN_BEFORE_MAX_SECONDS) {
          setWarning(`Recording stops automatically in ${Math.ceil(remaining)} s (maximum ${maxDuration} s).`);
        } else if (elapsed < minDuration) {
          setWarning(`Keep going: at least ${minDuration} s needed.`);
        } else {
          setWarning(null);
        }
      }, TIMER_INTERVAL_MS);
    } catch (err) {
      console.error("Recorder start failed", err);
      releaseStream();
      setError(ERRORS.failed);
    }
  }, [supported, reset, maxDuration, minDuration, finalize, releaseStream, setError, setPermission, setRecording, setRecordingTime, setWarning]);

  const play = useCallback(() => {
    if (!audioUrl) return;
    if (!audioElRef.current || audioElRef.current.src !== audioUrl) {
      audioElRef.current?.pause();
      const el = new Audio(audioUrl);
      el.onended = () => setPlaying(false);
      el.onpause = () => setPlaying(false);
      el.onplay = () => setPlaying(true);
      audioElRef.current = el;
    }
    void audioElRef.current.play().catch(() => setError(ERRORS.failed));
  }, [audioUrl, setPlaying, setError]);

  const pause = useCallback(() => {
    audioElRef.current?.pause();
  }, []);

  const clear = useCallback(() => {
    audioElRef.current?.pause();
    audioElRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      discardRef.current = true;
      recorderRef.current.stop();
    }
    stopTimer();
    releaseStream();
    reset();
  }, [releaseStream, reset, stopTimer]);

  /** Live time-domain samples (0..255, 128 = silence) while recording; false when idle. */
  const getWaveform = useCallback((target: Uint8Array<ArrayBuffer>): boolean => {
    const analyser = analyserRef.current;
    if (!analyser) return false;
    analyser.getByteTimeDomainData(target);
    return true;
  }, []);

  // Release everything when the component unmounts.
  useEffect(() => {
    return () => {
      stopTimer();
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        discardRef.current = true;
        recorderRef.current.stop();
      }
      releaseStream();
      audioElRef.current?.pause();
    };
  }, [releaseStream, stopTimer]);

  return {
    supported,
    checked,
    isRecording,
    recordingTime,
    audioBlob,
    audioUrl,
    isPlaying,
    error,
    warning,
    permission,
    peaks,
    maxDuration,
    minDuration,
    start,
    stop,
    play,
    pause,
    clear,
    getWaveform,
  };
}
