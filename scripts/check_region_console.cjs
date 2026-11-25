/* Optional bench QA: node + Playwright + a local Chromium. Independent native loss runs are the pixel oracle. */
"use strict";
const fs = require("node:fs"),
  path = require("node:path"),
  { pathToFileURL } = require("node:url");
const { createHash } = require("node:crypto"),
  assert = require("node:assert/strict");
const { chromium } = require("playwright");
const [html, nativeRoot, qualification, browserExe, screenshot, report] =
  process.argv.slice(2);
const sha = (b) => createHash("sha256").update(b).digest("hex");
async function check() {
  const native = JSON.parse(fs.readFileSync(qualification)).cases;
  const browser = await chromium.launch({
    executablePath: browserExe,
    headless: true,
    args: ["--disable-gpu"],
  });
  try {
    const ctx = await browser.newContext({
      offline: true,
      viewport: { width: 1240, height: 850 },
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage(),
      external = [],
      errors = [];
    page.on("request", (r) => {
      if (/^https?:/.test(r.url())) external.push(r.url());
    });
    page.on("pageerror", (e) => errors.push(String(e)));
    await page.goto(pathToFileURL(path.resolve(html)).href);
    const ready = (str) =>
      page.waitForFunction(
        (key) => document.querySelector("#map").dataset.ready === key,
        str,
      );
    const pick = async (name, i, j) => {
      const jobs = await page.locator("#job option").allTextContents(),
        idx = jobs.findIndex((v) => v.startsWith(name + " ·"));
      assert(idx >= 0);
      await page.selectOption("#job", String(idx));
      await page.waitForFunction(
        (job) =>
          document.querySelector("#map").dataset.ready?.startsWith(job + ":"),
        name,
      );
      await page.selectOption("#ref", String(i));
      await page.waitForFunction(
        () => !!document.querySelector("#map").dataset.ready,
      );
      await page.selectOption("#part", String(j));
      await ready(`${name}:${i},${j}`);
    };
    let checked = 0,
      renders = 0,
      cells = 0,
      clicked = 0,
      unknownViews = 0;
    const colors = [
      [49, 72, 88],
      [255, 173, 67],
      [201, 166, 223],
      [117, 194, 220],
    ];
    for (const [name, rec] of Object.entries(native)) {
      const read = (c) => {
        let bytes = fs.readFileSync(
          path.join(nativeRoot, name, `native-${c.pair.join("-")}.json`),
        );
        assert.equal(sha(bytes), c.native_report_sha256);
        let n = JSON.parse(bytes),
          a = new Int32Array(n.rows * n.columns);
        for (const [y, x, l, id] of n.runs)
          a.fill(Math.max(-1, id), y * n.columns + x, y * n.columns + x + l);
        return { n, a };
      };
      const nominalPair = rec.cases[0].pair; // validate_regions places the nominal map first
      const base = read(rec.cases[0]).a;
      for (const c of rec.cases) {
        const [i, j] = c.pair,
          { n, a } = read(c);
        await pick(name, i, j);
        assert.equal(
          await page.locator("#labelhash").textContent(),
          c.labels_i32le_sha256,
        );
        assert.equal(
          await page.locator("#labelhash").textContent(),
          sha(Buffer.from(a.buffer)),
        );
        assert.equal(
          await page.locator("#counts").textContent(),
          `${n.regions.length} 片区域 / ${n.coverage.flagged} 格超限`,
        );
        const w = n.columns,
          h = n.rows;
        for (const [off, overlay] of [
          [0, false],
          [Math.floor(w / 2), true],
        ]) {
          await page.locator("#seam").fill(String(off));
          await page.locator("#overlay").setChecked(overlay);
          let expected = Buffer.alloc(4 * w * h);
          for (let r = 0; r < h; r++)
            for (let x = 0; x < w; x++) {
              const t = r * w + ((x + off) % w),
                id = a[t],
                k = id < 0 ? 2 : id > 0 ? 1 : overlay && base[t] > 0 ? 3 : 0;
              let p = (r * w + x) * 4;
              for (let v = 0; v < 3; v++) expected[p + v] = colors[k][v];
              expected[p + 3] = 255;
            }
          const actual = await page.locator("#map").evaluate(async (el) => {
            const b = el
              .getContext("2d")
              .getImageData(0, 0, el.width, el.height).data;
            return Array.from(
              new Uint8Array(await crypto.subtle.digest("SHA-256", b)),
              (x) => x.toString(16).padStart(2, "0"),
            ).join("");
          });
          assert.equal(actual, sha(expected));
          renders++;
          cells += a.length;
        }
        const t =
          a.findIndex((x) => x < 0) >= 0
            ? a.findIndex((x) => x < 0)
            : a.findIndex((x) => x > 0);
        if (t >= 0) {
          const rect = await page.locator("#map").boundingBox(),
            row = Math.floor(t / w),
            col = t % w,
            x = (col - Math.floor(w / 2) + w) % w;
          await page.mouse.click(
            rect.x + ((x + 0.5) * rect.width) / w,
            rect.y + ((row + 0.5) * rect.height) / h,
          );
          assert(
            (await page.locator("#coord").textContent()).startsWith(
              `行 ${row}，列 ${col}（`,
            ),
          );
          assert(
            (await page.locator("#region").textContent()).includes(
              a[t] < 0 ? "不可测" : `#${a[t]} ·`,
            ),
          );
          clicked++;
          if (a[t] < 0) unknownViews++;
        }
        checked++;
      }
      await page.click("#reset");
      await ready(`${name}:${nominalPair.join(",")}`);
    }
    for (const name of ["slide", "missing-reference"]) {
      const texts = await page.locator("#job option").allTextContents();
      await page.selectOption(
        "#job",
        String(texts.findIndex((t) => t.startsWith(name + " ·"))),
      );
      assert(await page.locator("#work").isHidden());
      assert(await page.locator("#empty").isVisible());
      assert(
        (await page.locator("#reason").textContent()).includes("没有减薄图"),
      );
      assert.equal(await page.locator("#labelhash").textContent(), "—");
    }
    await pick("repeat", 1, 1);
    await page.locator("#map").focus();
    await page.keyboard.press("ArrowLeft");
    assert((await page.locator("#coord").textContent()).includes("列 359"));
    await page.keyboard.press("ArrowRight");
    assert((await page.locator("#coord").textContent()).includes("列 0（"));
    await pick("noisy-change", 1, 1);
    await page.locator("#overlay").check();
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await page.setViewportSize({ width: 1240, height: 850 });
    await page
      .locator("main")
      .screenshot({ path: screenshot, animations: "disabled" });
    assert(
      checked === 22 &&
        renders === 44 &&
        cells === 649440 &&
        unknownViews === 3,
    );
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    // Export a damaged *copy*, not the trusted input. Both bad checksum and malformed RLE must hide the map.
    const fileText = fs.readFileSync(html, "utf8"),
      tmp = fs.mkdtempSync(
        path.join(require("node:os").tmpdir(), "conefdk-console-"),
      );
    const damaged = path.join(tmp, "negative.html");
    try {
      for (const fault of ["digest", "rle"]) {
        const bad = fileText.replace(
          /(<script id="payload" type="application\/json">)([\s\S]*?)(<\/script>)/,
          (_, start, raw, end) => {
            const doc = JSON.parse(raw),
              e = doc.entries.find((x) => x.id === "noisy-change"),
              s = e.scenarios[e.nominal[0] * e.levels.length + e.nominal[1]];
            if (fault === "digest") s.labels_i32le_sha256 = "0".repeat(64);
            else s.pixels = ["rle32le", "AAAAAAAAAAA="];
            return start + JSON.stringify(doc).replaceAll("<", "\\u003c") + end;
          },
        );
        assert.notEqual(bad, fileText);
        fs.writeFileSync(damaged, bad);
        await page.goto(pathToFileURL(damaged).href + "?case=" + fault);
        await page.getByText("标签验证失败", { exact: true }).waitFor();
        assert(await page.locator("#work").isHidden());
        assert.equal(await page.locator("#labelhash").textContent(), "—");
      }
    } finally {
      if (fs.existsSync(damaged)) fs.unlinkSync(damaged);
      fs.rmdirSync(tmp);
    }
    const result = {
      protocol:
        "Actual canvas RGBA digest compared to independently pinned native loss runs at two seam positions, with nominal overlay off/on; material-coordinate picks, unknown precedence, keyboard seam wrap, refusal, offline file URL and mobile width.",
      browser: browser.version(),
      html_sha256: sha(fs.readFileSync(html)),
      native_qualification_sha256: sha(fs.readFileSync(qualification)),
      corrupt_maps_refused: 2,
      maps: checked,
      canvas_comparisons: renders,
      pixels_compared: cells,
      picks: clicked,
      unknown_picks: unknownViews,
      refusals_without_map: 2,
      external_requests: external.length,
      page_errors: errors.length,
      screenshot_sha256: sha(fs.readFileSync(screenshot)),
    };
    fs.writeFileSync(report, JSON.stringify(result, null, 2) + "\n");
    console.log(result);
  } finally {
    await browser.close();
  }
}
check().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
