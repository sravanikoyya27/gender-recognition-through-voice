"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Home, LogIn, LogOut, Mic, Waves } from "lucide-react";

import { useAuth } from "@/hooks/useAuth";
import { cn } from "@/lib/utils";

const APP_NAME = process.env.NEXT_PUBLIC_APP_NAME ?? "Gender Recognition Through Voice";

const links = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: Mic },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
];

export default function Navigation() {
  const pathname = usePathname();
  const { isAuthenticated, hasHydrated, logout } = useAuth();

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-card/80 backdrop-blur">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Waves className="h-6 w-6 text-brand" aria-hidden />
          <span className="hidden sm:inline">{APP_NAME}</span>
          <span className="sm:hidden">Voice Gender</span>
        </Link>

        <nav aria-label="Primary" className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "inline-flex items-center gap-2 rounded px-3 py-2 text-sm font-medium transition-colors",
                  active ? "bg-brand-soft text-brand-700 dark:text-brand-200" : "hover:bg-muted",
                )}
                style={active ? { backgroundColor: "var(--color-brand-soft)" } : undefined}
              >
                <Icon className="h-4 w-4" aria-hidden />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}

          {hasHydrated ? (
            isAuthenticated ? (
              <button type="button" onClick={logout} className="btn-secondary ml-2">
                <LogOut className="h-4 w-4" aria-hidden />
                <span className="hidden md:inline">Sign out</span>
              </button>
            ) : (
              <Link href="/login" className="btn-primary ml-2">
                <LogIn className="h-4 w-4" aria-hidden />
                <span className="hidden md:inline">Sign in</span>
              </Link>
            )
          ) : null}
        </nav>
      </div>
    </header>
  );
}
