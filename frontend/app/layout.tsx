import type { Metadata } from "next";
import localFont from "next/font/local";

import Footer from "@/components/Footer";
import Navigation from "@/components/Navigation";
import StoreHydration from "@/components/StoreHydration";
import "@/styles/globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

const APP_NAME = process.env.NEXT_PUBLIC_APP_NAME ?? "Gender Recognition Through Voice";

export const metadata: Metadata = {
  title: {
    default: APP_NAME,
    template: `%s | ${APP_NAME}`,
  },
  description:
    "Predicts a speaker's gender from a short voice recording using a neural network trained on 20 acoustic features.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="flex min-h-screen flex-col">
        <StoreHydration />
        <Navigation />
        <main className="container flex-1 py-8">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
