import { create } from "zustand";

import type { ApiError } from "@/types/api";
import type { Prediction, PredictionStatus } from "@/types/models";

interface PredictionState {
  status: PredictionStatus;
  result: Prediction | null;
  error: ApiError | null;
  /** Results from this browser session, newest first. */
  history: Prediction[];
  setStatus: (status: PredictionStatus) => void;
  setResult: (result: Prediction) => void;
  setError: (error: ApiError) => void;
  reset: () => void;
  clearHistory: () => void;
}

export const usePredictionStore = create<PredictionState>()((set) => ({
  status: "idle",
  result: null,
  error: null,
  history: [],
  setStatus: (status) => set({ status }),
  setResult: (result) =>
    set((state) => ({
      status: "success",
      result,
      error: null,
      history: [result, ...state.history].slice(0, 50),
    })),
  setError: (error) => set({ status: "error", error }),
  reset: () => set({ status: "idle", result: null, error: null }),
  clearHistory: () => set({ history: [] }),
}));
