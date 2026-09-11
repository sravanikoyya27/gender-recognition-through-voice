"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const tabs = [
  { href: "/dashboard", label: "Predict" },
  { href: "/analytics", label: "Analytics" },
];

export default function DashboardLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const pathname = usePathname();

  return (
    <div className="flex flex-col gap-6">
      <nav aria-label="Dashboard sections" className="flex gap-1 border-b border-border">
        {tabs.map(({ href, label }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "-mb-px border-b-2 px-4 py-2 text-sm font-medium transition-colors",
                active
                  ? "border-brand text-brand"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              {label}
            </Link>
          );
        })}
      </nav>
      {children}
    </div>
  );
}
