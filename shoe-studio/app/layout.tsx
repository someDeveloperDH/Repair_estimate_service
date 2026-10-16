import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SOLE — 신발 수선 스튜디오",
  description: "여러 장의 사진을 하나의 손상 기록으로. 신발 모형에서 확인하는 중복 없는 수선 견적 UI 데모.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className="antialiased">{children}</body>
    </html>
  );
}
