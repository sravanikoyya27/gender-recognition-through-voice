"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";

import { API_BASE_URL, getHealth, toApiError } from "@/services/api";
import type { HealthResponse } from "@/types/api";

type State =
  | { kind: "loading" }
  | { kind: "ok"; health: HealthResponse }
  | { kind: "down"; message: string };

/** Small live indicator that the frontend can reach the backend. */
export default function ApiStatus() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((health) => !cancelled && setState({ kind: "ok", health }))
      .catch((err) => !cancelled && setState({ kind: "down", message: toApiError(err).message }));
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.kind === "loading") {
    return (
      <p className="inline-flex items-center gap-2 text-sm text-muted-foreground">
        <CircleDashed className="h-4 w-4 animate-spin" aria-hidden /> Checking API...
      </p>
    );
  }

  if (state.kind === "down") {
    return (
      <p className="inline-flex items-center gap-2 text-sm text-red-600">
        <XCircle className="h-4 w-4" aria-hidden /> API unreachable at {API_BASE_URL}: {state.message}
      </p>
    );
  }

  const { health } = state;
  return (
    <p className="inline-flex flex-wrap items-center gap-2 text-sm">
      <CheckCircle2 className="h-4 w-4 text-green-600" aria-hidden />
      API v{health.version} {health.status === "ok" ? "healthy" : "reachable (degraded)"}
      <span className="text-muted-foreground">
        &middot; database {health.database} &middot; model {health.model_loaded ? "loaded" : "not loaded"}
      </span>
    </p>
  );
}
