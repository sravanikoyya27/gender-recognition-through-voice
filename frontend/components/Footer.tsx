const APP_NAME = process.env.NEXT_PUBLIC_APP_NAME ?? "Gender Recognition Through Voice";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-border py-6 text-sm text-muted-foreground">
      <div className="container flex flex-col items-center justify-between gap-2 sm:flex-row">
        <p>
          {APP_NAME} &middot; Neural network on 20 acoustic features
        </p>
        <p>
          Rebuilt from the B.Tech project by GVPCEW Batch 10 (2020-21)
        </p>
      </div>
    </footer>
  );
}
