import { cn } from "@/lib/utils";

/** Grey placeholder block shown while data loads. */
export default function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded bg-muted", className)} />;
}
