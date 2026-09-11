"use client";

import { motion } from "framer-motion";

interface Props {
  size?: "sm" | "md" | "xl";
}

export function RatioWordmark({ size = "md" }: Props) {
  const sizes = {
    sm: { text: "text-xl", dot: "w-1.5 h-1.5", gap: "mb-0.5" },
    md: { text: "text-3xl", dot: "w-2 h-2", gap: "mb-1" },
    xl: { text: "text-6xl", dot: "w-2.5 h-2.5", gap: "mb-2" },
  };
  const s = sizes[size];

  return (
    <div className="flex items-baseline gap-1.5">
      <span
        className={`wordmark ${s.text} text-[var(--color-ivory)]`}
        style={{ letterSpacing: size === "xl" ? "0.15em" : "0.1em" }}
      >
        Ratio
      </span>
      {/* Gold accent dot — subtle marker of authority */}
      <motion.span
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        className={`${s.dot} rounded-full bg-[var(--color-gold)] shrink-0`}
        style={{ marginBottom: size === "xl" ? "10px" : "4px" }}
      />
    </div>
  );
}
