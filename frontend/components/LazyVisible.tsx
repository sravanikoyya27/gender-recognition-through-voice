"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";

interface LazyVisibleProps {
  children: ReactNode;
  /** Placeholder height while off-screen, keeps the page from jumping. */
  minHeight?: number | string;
  /** Start rendering this many pixels before the element scrolls into view. */
  rootMargin?: string;
  className?: string;
}

/** Renders its children the first time the wrapper scrolls near the viewport (chart lazy loading). */
export default function LazyVisible({
  children,
  minHeight = 288,
  rootMargin = "200px",
  className,
}: LazyVisibleProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || visible) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [visible, rootMargin]);

  return (
    <div ref={ref} className={className} style={visible ? undefined : { minHeight }}>
      {visible ? children : null}
    </div>
  );
}
