"use client";

import { useEffect, useState } from "react";

/** Returns `value` after it has stayed unchanged for `delayMs` (filter inputs → fetches). */
export function useDebouncedValue<T>(value: T, delayMs = 500): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}
