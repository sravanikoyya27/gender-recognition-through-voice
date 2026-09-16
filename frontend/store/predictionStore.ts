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
  setResult: (result) => {
    // The API response carries no timestamp for fresh predictions; record when it arrived.
    const stamped = result.created_at ? result : { ...result, created_at: new Date().toISOString() };
    set((state) => ({
      status: "success",
      result: stamped,
      error: null,
      history: [stamped, ...state.history].slice(0, 50),
    }));
  },
  setError: (error) => set({ status: "error", error }),
  reset: () => set({ status: "idle", result: null, error: null }),
  clearHistory: () => set({ history: [] }),
}));
