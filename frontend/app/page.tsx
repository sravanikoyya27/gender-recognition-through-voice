import Link from "next/link";
import { ArrowRight, BarChart3, BrainCircuit, Mic } from "lucide-react";

import ApiStatus from "@/components/ApiStatus";

const features = [
  {
    icon: Mic,
    title: "Record or upload",
    body: "Capture a few seconds of speech in the browser or upload a WAV file.",
  },
  {
    icon: BrainCircuit,
    title: "20 acoustic features",
    body: "Mean frequency, spectral entropy, fundamental and dominant frequency statistics and more, reproduced in Python from the original R pipeline.",
  },
  {
    icon: BarChart3,
    title: "Neural network verdict",
    body: "A multilayer perceptron returns male or female with a calibrated confidence score.",
  },
];

export default function HomePage() {
  return (
    <div className="flex flex-col gap-16">
      <section className="flex flex-col items-center gap-6 py-12 text-center">
        <span className="rounded-full border border-border bg-card px-3 py-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Neural networks &middot; Acoustic analysis
        </span>
        <h1 className="text-balance text-display sm:text-display-lg">
          Gender Recognition <span className="text-brand">Through Voice</span>
        </h1>
        <p className="max-w-2xl text-balance text-lg text-muted-foreground">
          A modern rebuild of a B.Tech final-year project: the speaker&apos;s voice is reduced to
          twenty acoustic measurements and classified by a neural network.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link href="/dashboard" className="btn-primary">
            Try it now <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link href="/analytics" className="btn-secondary">
            View analytics
          </Link>
        </div>
        <ApiStatus />
      </section>

      <section className="grid gap-6 md:grid-cols-3">
        {features.map(({ icon: Icon, title, body }) => (
          <article key={title} className="card">
            <Icon className="mb-3 h-6 w-6 text-brand" aria-hidden />
            <h2 className="text-h3">{title}</h2>
            <p className="mt-2 text-sm text-muted-foreground">{body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
