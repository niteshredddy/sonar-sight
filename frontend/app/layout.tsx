import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SonarSight — Marine Debris Detection",
  description:
    "AI-powered detection of man-made marine debris in side-scan sonar imagery. Detect ghost nets, shipwrecks, pipes, and cylinders with geotagged reporting.",
  keywords: ["sonar", "marine debris", "computer vision", "YOLO", "side-scan sonar", "SIH"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </head>
      <body style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
        {/* Header */}
        <header
          style={{
            position: "sticky",
            top: 0,
            zIndex: 50,
            background: "rgba(10, 22, 40, 0.85)",
            backdropFilter: "blur(16px)",
            borderBottom: "1px solid rgba(0, 212, 255, 0.1)",
            padding: "0 24px",
          }}
        >
          <div
            style={{
              maxWidth: 1400,
              margin: "0 auto",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              height: 64,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 28 }}>🌊</span>
              <h1
                className="gradient-text"
                style={{ fontSize: 22, fontWeight: 800, margin: 0, letterSpacing: -0.5 }}
              >
                SonarSight
              </h1>
              <span
                style={{
                  fontSize: 11,
                  color: "var(--color-accent-cyan)",
                  background: "rgba(0, 212, 255, 0.1)",
                  padding: "2px 8px",
                  borderRadius: 20,
                  fontWeight: 600,
                }}
              >
                SIH 2026
              </span>
            </div>
            <nav style={{ display: "flex", gap: 24 }}>
              <a
                href="/"
                style={{
                  color: "var(--color-text-secondary)",
                  textDecoration: "none",
                  fontSize: 14,
                  fontWeight: 500,
                  transition: "color 0.2s",
                }}
              >
                Upload
              </a>
            </nav>
          </div>
        </header>

        {/* Main content */}
        <main style={{ maxWidth: 1400, margin: "0 auto", padding: "24px" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
