"use client";

import { useCallback } from "react";

import { CANCELLED, type UploadOptions, predictAudio, toApiError } from "@/services/api";
import { useAnalyticsStore } from "@/store/analyticsStore";
import { usePredictionStore } from "@/store/predictionStore";

export function usePrediction() {
  const { status, result, error, history, setStatus, setResult, setError, reset } =
    usePredictionStore();

  /**
   * Upload a recording or file and store the prediction.
   * Resolves to the prediction, or null on error / cancellation (a cancelled
   * request returns the store to `idle` without an error).
   */
  const predict = useCallback(
    async (file: File | Blob, filename?: string, options: UploadOptions = {}) => {
      setStatus("uploading");
      try {
        const prediction = await predictAudio(file, filename, {
          ...options,
          onUploadProgress: (event) => {
            options.onUploadProgress?.(event);
            if (event.total && event.loaded >= event.total) {
              setStatus("analyzing"); // bytes are on the server; the model is working
            }
          },
        });
        setResult(prediction);
        // The user's history and statistics changed: let the analytics page refetch on its next visit.
        if (prediction.prediction_id != null) useAnalyticsStore.getState().invalidate();
        return prediction;
      } catch (err) {
        const apiError = toApiError(err);
        if (apiError.detail === CANCELLED) {
          reset();
          return null;
        }
        setError(apiError);
        return null;
      }
    },
    [setStatus, setResult, setError, reset],
  );

  return {
    status,
    result,
    error,
    history,
    isBusy: status === "uploading" || status === "analyzing",
    predict,
    reset,
  };
}
