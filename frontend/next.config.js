// @ts-check

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const apiHost = new URL(apiUrl);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Emit a self-contained server bundle (used by the Docker image in infrastructure/).
  output: "standalone",

  // Public build-time defaults; values in .env / .env.local take precedence.
  env: {
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME ?? "Gender Recognition Through Voice",
    NEXT_PUBLIC_API_URL: apiUrl,
  },

  // Image optimisation: allow visualisations served by the backend (spectrograms, plots)
  // and any S3 bucket configured for the project.
  images: {
    formats: ["image/avif", "image/webp"],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    remotePatterns: [
      {
        protocol: apiHost.protocol.replace(":", "") === "https" ? "https" : "http",
        hostname: apiHost.hostname,
        port: apiHost.port || "",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "*.amazonaws.com",
        pathname: "/**",
      },
    ],
  },

  experimental: {
    // Tree-shake large icon / chart packages to only the symbols we import.
    optimizePackageImports: ["lucide-react", "recharts"],
  },

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          // The recorder needs microphone access on our own origin only.
          { key: "Permissions-Policy", value: "microphone=(self), camera=()" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
