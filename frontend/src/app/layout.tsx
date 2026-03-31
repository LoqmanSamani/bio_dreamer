import type { Metadata } from "next";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import ThemeProvider from "@/components/ThemeProvider";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "BioDreamer. World Models for Biological Design",
  description:
    "An open-source framework that applies JEPA-based world models and Active Inference to molecular dynamics, protein engineering, and cell reprogramming.",
  keywords: [
    "BioDreamer",
    "world models",
    "protein design",
    "model-based RL",
    "active inference",
    "molecular dynamics",
    "cell reprogramming",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col">
        <ThemeProvider>
          <Navbar />
          <main className="flex-1">{children}</main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
