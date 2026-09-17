import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, Database, BrainCircuit, CheckCircle2, AlertTriangle, ChevronRight } from "lucide-react";

export interface AgentThought {
  id: string;
  type: "scratchpad" | "search" | "status" | "error";
  content: string;
}

export function AgentTerminal({ thoughts }: { thoughts: AgentThought[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as new thoughts stream in
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [thoughts]);

  if (thoughts.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      className="w-full max-w-4xl mx-auto mb-8 relative rounded-[1.25rem] overflow-hidden"
      style={{
        background: "var(--color-obsidian)",
        boxShadow: "0 24px 48px -12px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
        border: "1px solid var(--color-graphite-border)",
      }}
    >
      {/* Top ambient glow */}
      <div className="absolute top-0 inset-x-0 h-[1px] bg-gradient-to-r from-transparent via-[var(--color-emerald)] to-transparent opacity-20" />

      {/* Header */}
      <div className="flex items-center px-5 py-3 border-b border-[var(--color-graphite-border)] bg-[var(--color-void)]/30 backdrop-blur-md">
        <div className="flex items-center justify-center w-6 h-6 rounded-md bg-[var(--color-graphite-deep)] border border-[var(--color-graphite-border)] mr-3">
          <Terminal size={12} className="text-[var(--color-emerald)]" />
        </div>
        <div className="flex flex-col">
          <span className="text-[0.65rem] font-medium font-sans text-[var(--color-ivory)] uppercase tracking-[0.2em] leading-none mb-1">
            Autonomous Engine
          </span>
          <span className="text-[0.6rem] font-mono text-[var(--color-ash)] uppercase tracking-wider leading-none flex items-center gap-1.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--color-emerald)] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[var(--color-emerald)]"></span>
            </span>
            Live Orchestration
          </span>
        </div>
        
        {/* Mac-style traffic lights */}
        <div className="ml-auto flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-[#FF5F56] border border-black/20 shadow-inner" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#FFBD2E] border border-black/20 shadow-inner" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#27C93F] border border-black/20 shadow-inner" />
        </div>
      </div>

      {/* Terminal Body */}
      <div 
        ref={containerRef}
        className="p-5 max-h-[320px] overflow-y-auto font-mono text-[0.825rem] space-y-4 hide-scrollbar relative"
        style={{ scrollBehavior: "smooth" }}
      >
        <AnimatePresence initial={false}>
          {thoughts.map((t, i) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 40 }}
              className="flex gap-4 group"
            >
              {/* Icon / Status Indicator */}
              <div className="flex flex-col items-center mt-0.5 shrink-0 relative">
                <div className={`flex items-center justify-center w-6 h-6 rounded-full border shadow-sm z-10 transition-colors
                  ${t.type === "search" ? "bg-blue-500/10 border-blue-500/20 text-blue-400" :
                    t.type === "scratchpad" ? "bg-[var(--color-emerald)]/10 border-[var(--color-emerald)]/20 text-[var(--color-emerald)]" :
                    t.type === "error" ? "bg-red-500/10 border-red-500/20 text-red-400" :
                    "bg-purple-500/10 border-purple-500/20 text-purple-400"
                  }`}
                >
                  {t.type === "search" && <Database size={11} strokeWidth={2.5} />}
                  {t.type === "scratchpad" && <BrainCircuit size={11} strokeWidth={2.5} />}
                  {t.type === "status" && <CheckCircle2 size={11} strokeWidth={2.5} />}
                  {t.type === "error" && <AlertTriangle size={11} strokeWidth={2.5} />}
                </div>
                {/* Connecting Line */}
                {i !== thoughts.length - 1 && (
                  <div className="w-px h-full absolute top-6 bg-[var(--color-graphite-border)] group-hover:bg-[var(--color-graphite-light)] transition-colors" />
                )}
              </div>
              
              {/* Content */}
              <div className="flex-1 pb-1">
                {t.type === "search" ? (
                  <div className="flex items-start">
                    <ChevronRight size={14} className="text-blue-500/60 mr-1.5 mt-0.5 shrink-0" />
                    <span className="text-[var(--color-ash)] leading-relaxed">
                      Executing vector search query: <br />
                      <span className="text-[var(--color-ivory)] font-medium">"{t.content}"</span>
                    </span>
                  </div>
                ) : t.type === "scratchpad" ? (
                  <div className="pl-4 py-2 my-1 border-l-[2px] border-[var(--color-emerald)]/30 bg-[var(--color-emerald)]/5 rounded-r-md text-[var(--color-parchment)] leading-relaxed shadow-[inset_1px_0_0_rgba(255,255,255,0.02)]">
                    {t.content}
                  </div>
                ) : t.type === "error" ? (
                  <div className="flex items-start text-red-400/90 leading-relaxed bg-red-500/5 p-3 rounded-md border border-red-500/10">
                    {t.content}
                  </div>
                ) : (
                  <div className="text-purple-300/80 leading-relaxed font-medium">
                    {t.content}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
