import { create } from "zustand";

export type UploadStatus = "idle" | "validating" | "ready" | "uploading" | "processing" | "done" | "error";

export interface UploadState {
  selectedFile: File | null;
  status: UploadStatus;
  isUploading: boolean;
  /** 0-100 */
  uploadProgress: number;
  uploadedBytes: number;
  totalBytes: number;
  /** MB/s, smoothed */
  uploadSpeed: number;
  /** seconds remaining, estimated from the current speed */
  estimatedTime: number;
  error: string | null;
  /** True when the error came from the upload/analysis (retry makes sense), false for validation errors. */
  retryable: boolean;

  selectFile: (file: File) => void;
  setStatus: (status: UploadStatus) => void;
  setProgress: (loaded: number, total: number, speedMb: number, etaSeconds: number) => void;
  setError: (message: string | null, retryable?: boolean) => void;
  clear: () => void;
}

const initial = {
  selectedFile: null,
  status: "idle" as UploadStatus,
  isUploading: false,
  uploadProgress: 0,
  uploadedBytes: 0,
  totalBytes: 0,
  uploadSpeed: 0,
  estimatedTime: 0,
  error: null,
  retryable: false,
};

export const useUploadStore = create<UploadState>()((set) => ({
  ...initial,
  selectFile: (file) => set({ ...initial, selectedFile: file, status: "validating", totalBytes: file.size }),
  setStatus: (status) => set({ status, isUploading: status === "uploading" }),
  setProgress: (loaded, total, speedMb, etaSeconds) =>
    set({
      uploadedBytes: loaded,
      totalBytes: total,
      uploadProgress: total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0,
      uploadSpeed: speedMb,
      estimatedTime: etaSeconds,
    }),
  setError: (message, retryable = false) =>
    set((state) => ({
      error: message,
      retryable: message ? retryable : false,
      status: message ? "error" : state.status,
      isUploading: false,
    })),
  clear: () => set({ ...initial }),
}));
