import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Ratio — Indian Legal AI Built on Ratio Decidendi";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "1200px",
          height: "630px",
          background: "#09090b",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "64px 72px",
          fontFamily: "Georgia, serif",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Subtle grid lines */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
            backgroundSize: "80px 80px",
            display: "flex",
          }}
        />

        {/* Gold accent top bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 72,
            right: 72,
            height: "1px",
            background:
              "linear-gradient(90deg, transparent, #E5C158 40%, #E5C158 60%, transparent)",
            display: "flex",
          }}
        />

        {/* Top row: wordmark + status pill */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "2px" }}>
            <span
              style={{
                fontSize: "52px",
                fontWeight: "300",
                color: "#F4F2EB",
                letterSpacing: "-1px",
                fontFamily: "Georgia, serif",
              }}
            >
              Ratio
            </span>
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: "#E5C158",
                marginLeft: "4px",
                marginBottom: "10px",
                display: "flex",
              }}
            />
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              padding: "8px 16px",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                background: "#56C48B",
                display: "flex",
              }}
            />
            <span
              style={{
                fontSize: "12px",
                fontFamily: "monospace",
                color: "#71717a",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              INDEX LIVE
            </span>
          </div>
        </div>

        {/* Main headline */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <span
            style={{
              fontSize: "11px",
              fontFamily: "monospace",
              color: "#E5C158",
              letterSpacing: "0.2em",
              textTransform: "uppercase",
            }}
          >
            RATIO DECIDENDI · LEGAL RAG ENGINE
          </span>
          <div
            style={{
              fontSize: "54px",
              fontWeight: "300",
              color: "#F4F2EB",
              lineHeight: 1.15,
              fontFamily: "Georgia, serif",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <span>Indian legal AI that retrieves</span>
            <span style={{ color: "#A09F95" }}>judicial authority, not text strings.</span>
          </div>
        </div>

        {/* Bottom: stat pills */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {[
            { value: "10,588", label: "Judgments" },
            { value: "580,939", label: "Indexed Chunks" },
            { value: "73.5%", label: "Recall@5" },
            { value: "87.2%", label: "MRR" },
          ].map((stat, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.07)",
                padding: "14px 20px",
                borderRadius: "10px",
                minWidth: "140px",
              }}
            >
              <span
                style={{
                  fontSize: "26px",
                  fontWeight: "600",
                  color: "#F4F2EB",
                  fontFamily: "monospace",
                  letterSpacing: "-0.5px",
                }}
              >
                {stat.value}
              </span>
              <span
                style={{
                  fontSize: "11px",
                  color: "#52525b",
                  fontFamily: "monospace",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                }}
              >
                {stat.label}
              </span>
            </div>
          ))}
          <div style={{ flex: 1, display: "flex", justifyContent: "flex-end" }}>
            <span
              style={{
                fontSize: "11px",
                fontFamily: "monospace",
                color: "#3f3f46",
                letterSpacing: "0.1em",
              }}
            >
              github.com/rasinmuhammed/ratio
            </span>
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}
