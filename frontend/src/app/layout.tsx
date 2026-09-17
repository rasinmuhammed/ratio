import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  display: "swap",
});

const SITE_URL = "https://rasinmuhammed.github.io/ratio";
const OG_IMAGE = `${SITE_URL}/og.jpg`;

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Ratio | Indian Legal AI Built on Ratio Decidendi",
    template: "%s | Ratio",
  },
  description:
    "Ratio is an open-source Indian legal research engine that retrieves judicial authority, not just text strings. Ask any legal question and receive grounded, cited answers from 10,588 court judgments across the Indian Supreme Court, High Courts, and Tribunals.",
  keywords: [
    "Indian legal AI",
    "RAG legal research",
    "ratio decidendi",
    "Indian case law search",
    "legal retrieval augmented generation",
    "Supreme Court India AI",
    "open source legal AI",
    "BM25 legal search",
    "hybrid retrieval",
    "K2-Horizon legal LLM",
    "promissory estoppel India",
    "Indian jurisprudence AI",
    "InJudgements dataset",
    "OpenNyAI",
  ],
  authors: [{ name: "Muhammed Rasin", url: "https://github.com/rasinmuhammed" }],
  creator: "Muhammed Rasin",
  publisher: "Ratio",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "en_IN",
    url: SITE_URL,
    siteName: "Ratio",
    title: "Ratio | Indian Legal AI Built on Ratio Decidendi",
    description:
      "The open-source RAG engine that extracts judicial authority from 10,588 Indian court judgments. 580,939 indexed chunks. 73.5% Recall@5. Built by a developer, measured honestly.",
    images: [
      {
        url: OG_IMAGE,
        width: 1920,
        height: 1080,
        alt: "Ratio - Indian Legal AI. 10,588 Judgments | 580,939 Chunks | 73.5% Recall@5",
        type: "image/jpeg",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Ratio | Indian Legal AI Built on Ratio Decidendi",
    description:
      "Ask any Indian legal question. Get a cited, court-grounded answer from 10,588 judgments. Open source. Every number is measured.",
    images: [OG_IMAGE],
    creator: "@rasinmuhammed",
  },
  alternates: {
    canonical: SITE_URL,
  },
  category: "technology",
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareApplication",
      "@id": `${SITE_URL}/#software`,
      name: "Ratio",
      description:
        "An open-source Indian legal research engine using RAG to extract ratio decidendi from 10,588 court judgments.",
      url: SITE_URL,
      applicationCategory: "LegalApplication",
      operatingSystem: "Web",
      offers: { "@type": "Offer", price: "0", priceCurrency: "INR" },
      author: {
        "@type": "Person",
        name: "Muhammed Rasin",
        url: "https://github.com/rasinmuhammed",
      },
      featureList: [
        "Hybrid BM25 and dense retrieval over Indian case law",
        "Judicial authority weighting by court tier",
        "Stance detection to distinguish holdings from submissions",
        "Corrective RAG with query decomposition",
        "Zero uncited claims via enforced JSON citation schema",
        "GraphRAG citation network traversal",
        "Multi-turn autonomous agent reasoning",
      ],
    },
    {
      "@type": "FAQPage",
      "@id": `${SITE_URL}/#faq`,
      mainEntity: [
        {
          "@type": "Question",
          name: "What is ratio decidendi?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Ratio decidendi is Latin for the reason for the decision. It is the binding legal principle a court articulates when deciding a case. Only the ratio creates enforceable precedent under the Indian doctrine of stare decisis.",
          },
        },
        {
          "@type": "Question",
          name: "How does Ratio differ from keyword-based legal search?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Conventional search returns documents containing keywords. Ratio uses hybrid retrieval (BM25 plus dense vector search), judicial authority weighting, stance detection to isolate holdings, and an autonomous reasoning agent to synthesize grounded answers with verbatim citations from 580,939 indexed passages.",
          },
        },
        {
          "@type": "Question",
          name: "What courts are covered?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "Ratio indexes 10,588 judgments from the Supreme Court of India, all major High Courts, and specialised tribunals (NCLT, NCLAT, ITAT), sourced from the OpenNyAI InJudgements dataset.",
          },
        },
        {
          "@type": "Question",
          name: "What retrieval model powers Ratio?",
          acceptedAnswer: {
            "@type": "Answer",
            text: "BAAI/bge-small-en-v1.5 for dense retrieval across 580,939 FAISS-indexed chunks, combined with case-sensitive BM25 for statutory citations. A cross-encoder reranker (BAAI/bge-reranker-base) promotes highest-authority passages. This achieves 73.5% Recall@5 and 87.2% MRR on a 313-label benchmark.",
          },
        },
      ],
    },
    {
      "@type": "WebSite",
      "@id": `${SITE_URL}/#website`,
      url: SITE_URL,
      name: "Ratio",
      description: "Indian Legal AI Built on Ratio Decidendi",
      inLanguage: "en-IN",
    },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en-IN" className={`${inter.variable} ${playfair.variable} dark`}>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <meta name="theme-color" content="#09090b" />
        <meta name="color-scheme" content="dark" />
      </head>
      <body className="antialiased min-h-screen">
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
