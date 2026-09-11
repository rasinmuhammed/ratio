"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { RatioWordmark } from "@/components/RatioWordmark";

const RagArchitectureDiagram = dynamic(
  () => import("@/components/RagArchitectureDiagram").then((m) => m.RagArchitectureDiagram),
  { ssr: false, loading: () => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-ash)", fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.15em" }}>
      LOADING PIPELINE...
    </div>
  ) }
);

export default function ArchitecturePage() {
  return (
    <div style={{ background: "var(--color-void)", minHeight: "100vh", display: "flex", flexDirection: "column", color: "var(--color-ivory)" }}>
      {/* Header */}
      <header style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, background: "rgba(7,7,10,0.92)", backdropFilter: "blur(16px) saturate(1.4)", borderBottom: "1px solid var(--color-graphite-border)" }}>
        <div className="ratio-container" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "4.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <Link href="/" style={{ textDecoration: "none" }}><RatioWordmark size="sm" /></Link>
            <span style={{ width: "1px", height: "1.25rem", background: "var(--color-graphite-border)" }} />
            <span style={{ fontSize: "0.7rem", fontWeight: 600, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>RAG Pipeline Architecture</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <Link href="/" style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.15em", color: "var(--color-fog)", textDecoration: "none" }}>
              Back to Overview
            </Link>
            <Link href="/research" style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "var(--color-ivory)", color: "var(--color-void-text)", fontSize: "0.75rem", fontWeight: 600, padding: "0.5rem 1rem", borderRadius: "0.5rem", textDecoration: "none" }}>
              Open Workspace
            </Link>
          </div>
        </div>
      </header>

      {/* Diagram */}
      <div style={{ flex: 1, paddingTop: "4.5rem", position: "relative" }}>
        <div style={{ width: "100%", height: "calc(100vh - 4.5rem)" }}>
          <RagArchitectureDiagram />
        </div>
      </div>
    </div>
  );
}
