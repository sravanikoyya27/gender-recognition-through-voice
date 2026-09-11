import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  label?: string;
  className?: string;
}

const sizes = { sm: "h-4 w-4 border-2", md: "h-6 w-6 border-2", lg: "h-10 w-10 border-4" };

export default function LoadingSpinner({ size = "md", label, className }: LoadingSpinnerProps) {
  return (
    <span role="status" className={cn("inline-flex items-center gap-2", className)}>
      <span
        className={cn(
          "animate-spin rounded-full border-brand-200 border-t-brand-600",
          sizes[size],
        )}
      />
      {label ? <span className="text-sm text-muted-foreground">{label}</span> : null}
      <span className="sr-only">{label ?? "Loading"}</span>
    </span>
  );
}
