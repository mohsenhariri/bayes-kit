import { describe, expect, it } from "vitest";

import {
  thresholdSpectrumAtK,
  thresholdSpectrumAtKCi,
  geomAtK,
  geomAtKCi,
  geomDsAtK,
  geomDsAtKCi,
  geoSpectrumAtK,
  geoSpectrumAtKCi,
  geoSpectrumStarAtK,
  geoSpectrumStarAtKCi,
} from "../src/eval/geom.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

describe("geomAtK", () => {
  it("matches doctest", () => {
    // Doctest: round(geom_at_k(R, 2), 6) -> 0.647106
    expect(round(geomAtK(R, 2), 6)).toBe(0.647106);
  });

  it("matches Python reference for extra cases", () => {
    expect(round(geomAtK(R, 3), 6)).toBe(0.474342);
    expect(round(geomAtK(R, 2, 0.3, 0.7), 6)).toBe(0.558242);
  });
});

describe("geomDsAtK", () => {
  it("matches doctest", () => {
    // Doctest: round(geom_ds_at_k(R, 2), 6) -> 0.653835
    expect(round(geomDsAtK(R, 2), 6)).toBe(0.653835);
  });

  it("matches Python reference for extra cases", () => {
    expect(round(geomDsAtK(R, 3), 6)).toBe(0.5);
    expect(round(geomDsAtK(R, 2, 0.3, 0.7), 6)).toBe(0.563074);
  });
});

describe("thresholdSpectrumAtK", () => {
  it("matches Python reference", () => {
    expect(round(thresholdSpectrumAtK(R, 2, [0.3, 0.4]), 6)).toBe(0.465);
    expect(round(thresholdSpectrumAtK(R, 3, [0.1, 0.2, 0.3]), 6)).toBe(0.345);
    expect(round(thresholdSpectrumAtK(R, 3, [0.0, 0.0, 1.0]), 6)).toBe(0.25);
  });
});

describe("geoSpectrumAtK", () => {
  it("matches doctest", () => {
    // Doctest: round(geo_spectrum_at_k(R, 3), 6) -> 0.408248
    expect(round(geoSpectrumAtK(R, 3), 6)).toBe(0.408248);
    // Doctest: round(geo_spectrum_at_k(R, 3, lam=1.0), 6) -> 1.0
    expect(round(geoSpectrumAtK(R, 3, 1.0), 6)).toBe(1.0);
  });

  it("matches Python reference for extra cases", () => {
    expect(round(geoSpectrumAtK(R, 2), 6)).toBe(0.653835);
    expect(round(geoSpectrumAtK(R, 3, 0.0), 6)).toBe(0.166667);
    // lambda_ alias overrides lam
    expect(round(geoSpectrumAtK(R, 3, 0.5, undefined, 0.25), 6)).toBe(0.260847);
    expect(round(geoSpectrumAtK(R, 3, 0.5, [0.1, 0.2, 0.3]), 6)).toBe(0.587367);
  });
});

describe("geoSpectrumStarAtK", () => {
  it("matches Python reference", () => {
    expect(round(geoSpectrumStarAtK(R, 2), 6)).toBe(0.653835);
    expect(round(geoSpectrumStarAtK(R, 3), 6)).toBe(0.408248);
    expect(round(geoSpectrumStarAtK(R, 4), 6)).toBe(0.632456);
  });
});

describe("geomAtKCi", () => {
  it("matches doctest", () => {
    // Doctest: round(mu,6),round(sigma,6),round(lo,4),round(hi,4)
    //   -> (0.610666, 0.133107, 0.3498, 0.8716)
    const [mu, sigma, lo, hi] = geomAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.610666);
    expect(round(sigma, 6)).toBe(0.133107);
    expect(round(lo, 4)).toBe(0.3498);
    expect(round(hi, 4)).toBe(0.8716);
  });

  it("matches Python reference for extra cases", () => {
    const a = geomAtKCi(R, 3);
    expect(round(a[0], 6)).toBe(0.543963);
    expect(round(a[1], 6)).toBe(0.140429);
    expect(round(a[2], 4)).toBe(0.2687);
    expect(round(a[3], 4)).toBe(0.8192);

    const b = geomAtKCi(R, 2, 0.3, 0.7);
    expect(round(b[0], 6)).toBe(0.538442);
    expect(round(b[1], 6)).toBe(0.140713);

    // latent k > N is allowed for CI
    const c = geomAtKCi(R, 6);
    expect(round(c[0], 6)).toBe(0.385915);
    expect(round(c[1], 6)).toBe(0.155094);
    expect(round(c[2], 4)).toBe(0.0819);
    expect(round(c[3], 4)).toBe(0.6899);
  });
});

describe("geomDsAtKCi", () => {
  it("matches doctest", () => {
    // Doctest -> (0.612112, 0.132755, 0.3519, 0.8723)
    const [mu, sigma, lo, hi] = geomDsAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.612112);
    expect(round(sigma, 6)).toBe(0.132755);
    expect(round(lo, 4)).toBe(0.3519);
    expect(round(hi, 4)).toBe(0.8723);
  });

  it("matches Python reference for extra cases", () => {
    const a = geomDsAtKCi(R, 3);
    expect(round(a[0], 6)).toBe(0.547813);
    expect(round(a[1], 6)).toBe(0.139933);
    expect(round(a[2], 4)).toBe(0.2735);
    expect(round(a[3], 4)).toBe(0.8221);

    const b = geomDsAtKCi(R, 2, 0.5, 0.5, 0.9);
    expect(round(b[2], 4)).toBe(0.3937);
    expect(round(b[3], 4)).toBe(0.8305);
  });
});

describe("thresholdSpectrumAtKCi", () => {
  it("matches Python reference", () => {
    const a = thresholdSpectrumAtKCi(R, 2, [0.3, 0.4]);
    expect(round(a[0], 6)).toBe(0.430357);
    expect(round(a[1], 6)).toBe(0.085472);
    expect(round(a[2], 4)).toBe(0.2628);
    expect(round(a[3], 4)).toBe(0.5979);

    const b = thresholdSpectrumAtKCi(R, 3, [0.1, 0.2, 0.3]);
    expect(round(b[0], 6)).toBe(0.326786);
    expect(round(b[1], 6)).toBe(0.079037);
    expect(round(b[2], 4)).toBe(0.1719);
    expect(round(b[3], 4)).toBe(0.4817);
  });
});

describe("geoSpectrumAtKCi", () => {
  it("matches Python reference", () => {
    const a = geoSpectrumAtKCi(R, 2);
    expect(round(a[0], 6)).toBe(0.612112);
    expect(round(a[1], 6)).toBe(0.132755);
    expect(round(a[2], 4)).toBe(0.3519);
    expect(round(a[3], 4)).toBe(0.8723);

    const b = geoSpectrumAtKCi(R, 3);
    expect(round(b[0], 6)).toBe(0.447288);
    expect(round(b[1], 6)).toBe(0.114255);
    expect(round(b[2], 4)).toBe(0.2234);
    expect(round(b[3], 4)).toBe(0.6712);

    const c = geoSpectrumAtKCi(R, 3, 1.0);
    expect(round(c[0], 6)).toBe(0.916667);
    expect(round(c[1], 6)).toBe(0.07321);
    expect(round(c[2], 4)).toBe(0.7732);
    expect(round(c[3], 4)).toBe(1.0);

    const d = geoSpectrumAtKCi(R, 3, 0.0);
    expect(round(d[0], 6)).toBe(0.218254);
    expect(round(d[1], 6)).toBe(0.098816);

    // lambda_ alias
    const e = geoSpectrumAtKCi(R, 3, 0.5, undefined, 0.25);
    expect(round(e[0], 6)).toBe(0.312446);
    expect(round(e[1], 6)).toBe(0.110471);

    // latent k > N
    const f = geoSpectrumAtKCi(R, 7);
    expect(round(f[0], 6)).toBe(0.529225);
    expect(round(f[1], 6)).toBe(0.132827);
    expect(round(f[2], 4)).toBe(0.2689);
    expect(round(f[3], 4)).toBe(0.7896);
  });
});

describe("geoSpectrumStarAtKCi", () => {
  it("matches Python reference", () => {
    const a = geoSpectrumStarAtKCi(R, 2);
    expect(round(a[0], 6)).toBe(0.612112);
    expect(round(a[1], 6)).toBe(0.132755);

    const b = geoSpectrumStarAtKCi(R, 3);
    expect(round(b[0], 6)).toBe(0.447288);
    expect(round(b[1], 6)).toBe(0.114255);
    expect(round(b[2], 4)).toBe(0.2234);
    expect(round(b[3], 4)).toBe(0.6712);
  });
});
