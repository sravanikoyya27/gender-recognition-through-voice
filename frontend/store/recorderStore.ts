import { create } from "zustand";

export type MicPermission = "unknown" | "prompt" | "granted" | "denied" | "unsupported";

export interface RecorderState {
  isRecording: boolean;
  /** Elapsed recording time in seconds (updated every 100 ms while recording). */
  recordingTime: number;
  /** The finished recording as a 16 kHz mono WAV blob. */
  audioBlob: Blob | null;
  /** Object URL for playback of `audioBlob`. */
  audioUrl: string | null;
  isPlaying: boolean;
  error: string | null;
  warning: string | null;
  permission: MicPermission;
  /** Peak-normalised amplitude envelope of the finished recording (for the static waveform). */
  peaks: number[] | null;

  setRecording: (value: boolean) => void;
  setRecordingTime: (seconds: number) => void;
  setRecordingResult: (blob: Blob | null, url: string | null, peaks: number[] | null) => void;
  setPlaying: (value: boolean) => void;
  setError: (message: string | null) => void;
  setWarning: (message: string | null) => void;
  setPermission: (value: MicPermission) => void;
  reset: () => void;
}

const initial = {
  isRecording: false,
  recordingTime: 0,
  audioBlob: null,
  audioUrl: null,
  isPlaying: false,
  error: null,
  warning: null,
  permission: "unknown" as MicPermission,
  peaks: null,
};

export const useRecorderStore = create<RecorderState>()((set, get) => ({
  ...initial,
  setRecording: (value) => set({ isRecording: value }),
  setRecordingTime: (seconds) => set({ recordingTime: seconds }),
  setRecordingResult: (blob, url, peaks) => {
    const previous = get().audioUrl;
    if (previous && previous !== url) {
      URL.revokeObjectURL(previous);
    }
    set({ audioBlob: blob, audioUrl: url, peaks, isPlaying: false });
  },
  setPlaying: (value) => set({ isPlaying: value }),
  setError: (message) => set({ error: message }),
  setWarning: (message) => set({ warning: message }),
  setPermission: (value) => set({ permission: value }),
  reset: () => {
    const previous = get().audioUrl;
    if (previous) {
      URL.revokeObjectURL(previous);
    }
    set({ ...initial, permission: get().permission });
  },
}));
