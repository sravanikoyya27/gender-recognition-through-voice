"use client";

import { useCallback } from "react";

import { predictAudio, toApiError } from "@/services/api";
import { usePredictionStore } from "@/store/predictionStore";

export function usePrediction() {
  const { status, result, error, history, setStatus, setResult, setError, reset } =
    usePredictionStore();

  const predict = useCallback(
    async (file: File | Blob, filename?: string) => {
      setStatus("uploading");
      try {
        setStatus("analyzing");
        const prediction = await predictAudio(file, filename);
        setResult(prediction);
        return prediction;
      } catch (err) {
        setError(toApiError(err));
        return null;
      }
    },
    [setStatus, setResult, setError],
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
