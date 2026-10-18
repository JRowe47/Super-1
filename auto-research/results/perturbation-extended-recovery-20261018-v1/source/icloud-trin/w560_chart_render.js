#!/usr/bin/env node
"use strict";

const { chromium } = require("playwright");

async function main() {
  const [sourceUrl, destination] = process.argv.slice(2);
  if (!sourceUrl || !destination) {
    throw new Error("usage: w560_chart_render.js <source-url> <destination.png>");
  }
  const browser = await chromium.launch({
    headless: true,
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  try {
    const viewportWidth = Number(process.env.W560_VIEWPORT_WIDTH || "1568");
    const viewportHeight = Number(process.env.W560_VIEWPORT_HEIGHT || "901");
    const context = await browser.newContext({
      viewport: { width: viewportWidth, height: viewportHeight },
      deviceScaleFactor: 2,
      colorScheme: "light",
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    page.on("console", message => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.goto(sourceUrl, { waitUntil: "load" });
    const frame = page.frames().find(candidate => candidate !== page.mainFrame());
    if (!frame) throw new Error("rendered inline visualization frame is missing");
    await frame.waitForSelector(".candidate-point");
    const counts = await frame.evaluate(() => ({
      canonical: document.querySelectorAll(".canonical-point").length,
      provisional: document.querySelectorAll(".provisional-point").length,
      cpu: document.querySelectorAll(".cpu-terminal").length,
      candidates: document.querySelectorAll(".candidate-point").length,
      width: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    if (errors.length || counts.canonical < 1 || counts.provisional > 1 || counts.cpu !== 81 || counts.candidates !== 81 || counts.width !== counts.scrollWidth) {
      throw new Error(`chart render validation failed: ${JSON.stringify({ counts, errors })}`);
    }
    await page.screenshot({ path: destination, fullPage: false });
    console.log(JSON.stringify({ destination, counts, errors }));
    await context.close();
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
