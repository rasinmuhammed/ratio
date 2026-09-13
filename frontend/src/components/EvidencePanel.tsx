"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ExternalLink } from "lucide-react";

interface SourceMeta {
  index: number;
  court: string;
  title: string;
  url?: string;
  // Added alongside the citation-signal card: the same stance/authority
  // data ask.py has always printed to a terminal (rag/stance.py,
  // metadata.cited_by), now reaching the web UI. "cited" answers the
  // question a plain source list can't: of the sources retrieved, which
  // ones did the model actually rely on for this answer, per
  // rag.generate.parse_answer against the same [n] markers scored for
  // citation precision.
  cited_by?: number;
  stance?: "holding" | "argument" | "both" | "neither" | string;
  cited?: boolean;
}

interface Props {
  sources: SourceMeta[];
  isLoading: boolean;
  sourceCount: number;
}

// Determine authority tier from court name
function getAuthorityTier(court: string): "supreme" | "high" | "tribunal" {
  const c = court.toLowerCase();
  if (c.includes("supreme")) return "supreme";
  if (c.includes("high court")) return "high";
  return "tribunal";
}

const AUTHORITY_CONFIG = {
  supreme: {
    label: "Supreme Court",
    colorValue: "var(--color-gold)",
    barColorValue: "var(--color-gold)",
    cardClass: "citation-card citation-card-supreme",
    bars: 3,
  },
  high: {
    label: "High Court",
    colorValue: "var(--color-emerald)",
    barColorValue: "var(--color-emerald)",
    cardClass: "citation-card",
    bars: 2,
  },
  tribunal: {
    label: "Tribunal / Other",
    colorValue: "var(--color-fog)",
    barColorValue: "var(--color-mist)",
    cardClass: "citation-card",
    bars: 1,
  },
};

const STANCE_CONFIG: Record<string, { label: string; color: string }> = {
  holding: { label: "Holding", color: "var(--color-emerald)" },
  argument: { label: "Argument", color: "var(--color-gold)" },
  both: { label: "Holding + Argument", color: "var(--color-emerald)" },
  neither: { label: "Procedural", color: "var(--color-ash)" },
};

function CitationCard({ source, delay }: { source: SourceMeta; delay: number }) {
  const tier = getAuthorityTier(source.court);
  const cfg = AUTHORITY_CONFIG[tier];
  const stance = source.stance ? STANCE_CONFIG[source.stance] : undefined;
  // "cited" is only meaningful once the model has actually answered - a
  // source list that arrives before a `cited` field exists (the plain
  // /stream endpoint, or an older cached response) should not visually
  // dim every card, so undefined reads as "not yet known", not "unused".
  const notCited = source.cited === false;

  return (
    <motion.a
      href={source.url || "#"}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: notCited ? 0.5 : 1, y: 0 }}
      transition={{ duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3, opacity: 1, transition: { duration: 0.25, ease: "easeOut" } }}
      className={cfg.cardClass}
      style={{ textDecoration: "none", display: "block", marginBottom: "0.5rem" }}
      title={notCited ? "Retrieved, but not cited in the answer" : undefined}
    >
      <div style={{ padding: "1.25rem" }}>
        {/* Header row */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.75rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
            {/* Authority bars */}
            <div style={{ display: "flex", alignItems: "flex-end", gap: "3px" }}>
              {[1, 2, 3].map((bar) => (
                <div
                  key={bar}
                  style={{
                    width: "3px", borderRadius: "9999px", height: `${bar * 5 + 4}px`,
                    background: bar <= cfg.bars ? cfg.barColorValue : "var(--color-graphite-border)"
                  }}
                />
              ))}
            </div>
            <span style={{ fontSize: "0.625rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.18em", color: cfg.colorValue }}>
              {source.court || cfg.label}
            </span>
          </div>
          <span style={{ flexShrink: 0, fontFamily: "var(--font-mono)", fontSize: "0.625rem", color: "var(--color-ash)", background: "var(--color-graphite-mid)", padding: "0.25rem 0.5rem", borderRadius: "0.375rem" }}>
            [{source.index}]
          </span>
        </div>

        {/* Case title */}
        <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "0.95rem", lineHeight: 1.55, color: "var(--color-ivory)", margin: "0 0 1rem 0", paddingRight: "0.5rem", fontWeight: 400 }}>
          {source.title || "Untitled Document"}
        </h3>

        {/* Signal row: stance + authority count, the same data ask.py
            has always printed to a terminal, given real visual weight */}
        {(stance || typeof source.cited_by === "number") && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", marginBottom: "1rem" }}>
            {stance && (
              <span style={{ fontSize: "0.6875rem", fontWeight: 600, color: stance.color, background: `color-mix(in srgb, ${stance.color} 14%, transparent)`, padding: "0.2rem 0.5rem", borderRadius: "0.3rem", letterSpacing: "0.02em" }}>
                {stance.label}
              </span>
            )}
            {typeof source.cited_by === "number" && (
              <span style={{ fontSize: "0.6875rem", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>
                cited by {source.cited_by} judgment{source.cited_by === 1 ? "" : "s"}
              </span>
            )}
          </div>
        )}

        {/* Footer */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingTop: "1rem", borderTop: "1px solid var(--color-graphite-border)" }}>
          <div style={{ fontSize: "0.625rem", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 500, color: cfg.colorValue, opacity: 0.6 }}>
            {tier === "supreme" ? "Binding Precedent" : tier === "high" ? "Persuasive Authority" : "Tribunal Decision"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {source.cited === true && (
              <span style={{ fontSize: "0.5625rem", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600, color: "var(--color-emerald)" }}>
                Cited in answer
              </span>
            )}
            <ExternalLink size={12} style={{ color: "var(--color-ash)" }} strokeWidth={1.5} />
          </div>
        </div>
      </div>
    </motion.a>
  );
}

function SkeletonCard({ delay }: { delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay }}
      className="citation-card"
      style={{ padding: "1.25rem", marginBottom: "0.5rem" }}
    >
      {/* Mock header */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "flex-end", gap: "3px" }}>
          {[1, 2, 3].map((h) => (
            <motion.div
              key={h}
              style={{ width: "3px", borderRadius: "9999px", background: "var(--color-graphite-border)", height: `${h * 5 + 4}px` }}
              animate={{ opacity: [0.2, 0.5, 0.2] }}
              transition={{ duration: 1.6, repeat: Infinity, delay: h * 0.1 }}
            />
          ))}
        </div>
        <motion.div
          style={{ height: "0.5rem", width: "7rem", borderRadius: "0.25rem", background: "var(--color-graphite-mid)" }}
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 1.6, repeat: Infinity, delay: 0.2 }}
        />
      </div>
      <motion.div
        style={{ height: "1rem", width: "100%", borderRadius: "0.25rem", background: "var(--color-graphite-mid)", marginBottom: "0.5rem" }}
        animate={{ opacity: [0.2, 0.45, 0.2] }}
        transition={{ duration: 1.8, repeat: Infinity }}
      />
      <motion.div
        style={{ height: "1rem", width: "80%", borderRadius: "0.25rem", background: "var(--color-graphite-mid)" }}
        animate={{ opacity: [0.2, 0.45, 0.2] }}
        transition={{ duration: 1.8, repeat: Infinity, delay: 0.1 }}
      />
      <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid var(--color-graphite-border)" }}>
        <motion.div
          style={{ height: "0.5rem", width: "6rem", borderRadius: "0.25rem", background: "var(--color-graphite-border)" }}
          animate={{ opacity: [0.2, 0.4, 0.2] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </div>
    </motion.div>
  );
}

export function EvidencePanel({ sources, isLoading, sourceCount }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Panel header */}
      <div style={{ padding: "1.25rem 1.75rem", borderBottom: "1px solid var(--color-graphite-border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.2em", fontWeight: 600, color: "var(--color-fog)", margin: 0 }}>
            Evidence Stack
          </h2>
        </div>
        <AnimatePresence>
          {sourceCount > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 300 }}
              style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
            >
              <div style={{ width: "0.375rem", height: "0.375rem", borderRadius: "50%", background: "var(--color-emerald)" }} />
              <span style={{ fontSize: "0.625rem", fontFamily: "var(--font-mono)", color: "var(--color-fog)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {sourceCount} sources
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Source list */}
      <div className="hide-scrollbar" style={{ flex: 1, overflowY: "auto", padding: "1.25rem 1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
        <AnimatePresence>
          {isLoading && sources.length === 0
            ? [0, 0.08, 0.16].map((d, i) => (
                <SkeletonCard key={`sk-${i}`} delay={d} />
              ))
            : sources.slice(0, 15).map((src, i) => (
                <CitationCard key={`src-${src.index}`} source={src} delay={0} />
              ))}
        </AnimatePresence>
      </div>

      {/* Panel footer legend */}
      <div style={{ padding: "1rem 1.75rem", borderTop: "1px solid var(--color-graphite-border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
          {(["supreme", "high", "tribunal"] as const).map((tier) => {
            const cfg = AUTHORITY_CONFIG[tier];
            return (
              <div key={tier} style={{ display: "flex", alignItems: "center", gap: "0.375rem" }}>
                <div style={{ display: "flex", alignItems: "flex-end", gap: "2px" }}>
                  {[1, 2, 3].map((h) => (
                    <div
                      key={h}
                      style={{ 
                        width: "2px", borderRadius: "9999px", height: `${h * 3 + 2}px`,
                        background: h <= cfg.bars ? cfg.barColorValue : "var(--color-graphite-border)"
                      }}
                    />
                  ))}
                </div>
                <span style={{ fontSize: "9px", textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--color-ash)" }}>
                  {tier}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
