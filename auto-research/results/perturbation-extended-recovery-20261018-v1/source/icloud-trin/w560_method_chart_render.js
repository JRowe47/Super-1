#!/usr/bin/env node
"use strict";

const { chromium } = require("playwright");

async function main() {
  const [sourceUrl, destination] = process.argv.slice(2);
  if (!sourceUrl || !destination) throw new Error("usage: w560_method_chart_render.js <source-url> <destination.png>");
  const browser = await chromium.launch({headless:true,executablePath:"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
  try {
    const viewportWidth=Number(process.env.W560_METHOD_WIDTH||"1568");
    const viewportHeight=Number(process.env.W560_METHOD_HEIGHT||"760");
    const context = await browser.newContext({viewport:{width:viewportWidth,height:viewportHeight},deviceScaleFactor:2,colorScheme:"light"});
    const page = await context.newPage();
    const errors=[];
    page.on("pageerror",error=>errors.push(String(error)));
    page.on("console",message=>{if(message.type()==="error")errors.push(message.text());});
    await page.goto(sourceUrl,{waitUntil:"load"});
    const frame=page.frames().find(candidate=>candidate!==page.mainFrame());
    if(!frame)throw new Error("visualization frame missing");
    await frame.waitForSelector(".block-point");
    const counts=await frame.evaluate(()=>({canonical:document.querySelectorAll(".canonical-point").length,methods:document.querySelectorAll(".method-point").length,width:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth}));
    if(errors.length||counts.canonical!==9||counts.methods!==2||counts.width!==counts.scrollWidth)throw new Error(`method chart validation failed: ${JSON.stringify({counts,errors})}`);
    await page.screenshot({path:destination,fullPage:false});
    console.log(JSON.stringify({destination,counts,errors}));
    await context.close();
  } finally { await browser.close(); }
}

main().catch(error=>{console.error(error);process.exit(1);});
