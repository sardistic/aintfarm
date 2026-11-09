// Shared by the browser (field.js) and the server (server.js): the genome layout, decoding, the phenotype
// embedding and the niche archive ("gene bank"). Keeping one copy means the shared bank can't drift from the GPU format.

export const EFFECTS = ['none', 'taxis', 'swirl', 'isoline', 'catalyse', 'jet', 'anchor', 'pressure', 'pulse', 'siphon',
  'relay', 'heat', 'shield', 'drift', 'uptake', 'inject', 'bond', 'edge'];

export function layout(F = 8, HMAX = 12) {
  const FS = F + 1;                // sensed channels: F evolvable + density
  const FX = F + 3;                // field stride: + occupant id + kin energy share slot
  const KE = 4;                    // effect-gene slots per particle
  const NT = EFFECTS.length;
  const NI = 3 * FS + 6;           // value+gradx+grady per sensed channel, energy, vel.xy, mem.xy, clock
  const NO = 6 + F + KE;           // split, share, write gate, write amt, aim[F], mem[2], effect gates[KE]
  const P = NI * HMAX + HMAX + HMAX * NO + NO;
  const HDR = 10;                  // H, range, mutRate, hue, lineage, generation, maxAge, clockFreq, learning rate, bond key
  const EOFF = HDR + 3 * F;        // chemistry: absorb[F], energy gain[F], push[F]
  const WOFF = EOFF + 4 * KE;      // effects: (type|chanA<<8|chanB<<16, p0, p1, p2) × KE
  const WWORDS = Math.ceil(P / 2); // weights as packed f16 pairs
  const GS = WOFF + WWORDS;        // u32 words per genome
  const B1 = NI * HMAX, W2 = B1 + HMAX, B2 = W2 + HMAX * NO;
  return { F, HMAX, FS, FX, KE, EFFECTS, NT, NI, NO, P, HDR, EOFF, WOFF, WWORDS, GS, B1, W2, B2 };
}

export function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const f32view = new DataView(new ArrayBuffer(4));
export function asF32(u) { f32view.setUint32(0, u, true); return f32view.getFloat32(0, true); }

export function genomeToB64(g) {
  const bytes = new Uint8Array(g.buffer, g.byteOffset, g.byteLength);
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
export function b64ToGenome(b64) {
  const bytes = typeof Buffer !== 'undefined' ? Uint8Array.from(Buffer.from(b64, 'base64')) : Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  if (bytes.byteLength % 4) return null;
  return new Uint32Array(bytes.buffer);
}

export function makeGenomeTools(L) {
  const { F, HMAX, KE, NT, NI, NO, P, HDR, EOFF, WOFF, WWORDS, GS, FS } = L;
  const PROJ_D = 8;
  // fixed random ±1 projections of the weight vector: a cheap fingerprint of the net itself
  const projSigns = (() => {
    let s = 0x9e3779b9; const out = new Int8Array(PROJ_D * P);
    for (let i = 0; i < out.length; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; out[i] = (s & 1) ? 1 : -1; }
    return out;
  })();

  function weights(g) {
    const w = new Float32Array(P);
    for (let k = 0; k < WWORDS; k++) { const u = g[WOFF + k]; w[2 * k] = f16(u & 0xffff); if (2 * k + 1 < P) w[2 * k + 1] = f16(u >>> 16); }
    return w;
  }

  function decodeGenome(g) {
    const H = Math.min(g[0], HMAX);
    const chem = { absorb: [], egain: [], push: [] };
    for (let c = 0; c < F; c++) {
      chem.absorb.push(asF32(g[HDR + c])); chem.egain.push(asF32(g[HDR + F + c])); chem.push.push(asF32(g[HDR + 2 * F + c]));
    }
    const effects = [];
    for (let s = 0; s < KE; s++) {
      const tw = g[EOFF + s * 4];
      effects.push({ t: Math.min(tw & 255, NT - 1), a: (tw >> 8) & 255, b: (tw >> 16) & 255,
        p0: asF32(g[EOFF + s * 4 + 1]), p1: asF32(g[EOFF + s * 4 + 2]), p2: asF32(g[EOFF + s * 4 + 3]) });
    }
    return { H, params: NI * H + H + H * NO + NO, range: asF32(g[1]), mut: asF32(g[2]), hue: asF32(g[3]),
      lineage: g[4] >>> 0, gen: g[5], maxAge: asF32(g[6]), clock: asF32(g[7]), eta: asF32(g[8]), key: g[9] >>> 0, chem, effects };
  }

  // Rejects anything the GPU could not have produced: wrong size, out-of-range genes, non-finite numbers.
  function validateGenome(g) {
    if (!(g instanceof Uint32Array) || g.length !== GS) return 'wrong length';
    if (g[0] < 2 || g[0] > HMAX) return 'hidden width';
    const fin = (k, lo, hi) => { const v = asF32(g[k]); return Number.isFinite(v) && v >= lo && v <= hi; };
    if (!fin(1, 1, 10) || !fin(2, 0, 0.31) || !fin(3, 0, 1) || !fin(6, 100, 8001) || !fin(7, 0, 1.01) || !fin(8, 0, 0.21)) return 'header';
    for (let c = 0; c < F; c++) {
      // founders skip mutation's clamps: random init can reach ±1.5 (energy gain) and ±4.5 (push)
      if (!fin(HDR + c, 0, 0.26) || !fin(HDR + F + c, -1.51, 1.51) || !fin(HDR + 2 * F + c, -4.51, 4.51)) return 'chemistry';
    }
    for (let s = 0; s < KE; s++) {
      const tw = g[EOFF + s * 4];
      if ((tw & 255) >= NT || ((tw >> 8) & 255) > FS - 1 || ((tw >> 16) & 255) > F - 1 || tw >>> 24) return 'effect';
      if (!fin(EOFF + s * 4 + 1, -1.81, 1.81) || !fin(EOFF + s * 4 + 2, -2.01, 2.01) || !fin(EOFF + s * 4 + 3, -3.01, 3.01)) return 'effect params';
    }
    const w = weights(g);
    for (let k = 0; k < P; k++) if (!Number.isFinite(w[k]) || Math.abs(w[k]) > 8.01) return 'weights';
    return null;
  }

  function embed(g, d) {
    const v = [];
    for (let c = 0; c < F; c++) v.push(d.chem.egain[c], d.chem.push[c] * 0.25, d.chem.absorb[c] * 4);
    const eff = new Array(NT).fill(0);
    for (const e of d.effects) if (e.t > 0) eff[e.t] += Math.min(Math.abs(e.p0), 1.5);
    v.push(...eff.slice(1).map(x => x * 0.8));
    v.push(d.range / 10, d.H / HMAX, Math.min(d.clock, 1), Math.min(d.eta * 20, 2));
    const w = weights(g), norm = 1 / Math.sqrt(P);
    for (let j = 0; j < PROJ_D; j++) {
      let s = 0; const o = j * P;
      for (let k = 0; k < P; k++) s += w[k] * projSigns[o + k];
      v.push(Math.tanh(s * norm * 0.5));
    }
    return Float32Array.from(v);
  }

  function phenoColor(d) {
    let qx = 0, qy = 0;
    for (let c = 0; c < F; c++) {
      const a = c * 2.39996;
      qx += Math.cos(a) * d.chem.egain[c] - Math.sin(a) * d.chem.push[c] * 0.35;
      qy += Math.sin(a) * d.chem.egain[c] + Math.cos(a) * d.chem.push[c] * 0.35;
    }
    for (const e of d.effects) if (e.t > 0) { const a = e.t * 0.897, m = Math.min(Math.abs(e.p0), 1) * 0.6; qx += Math.cos(a) * m; qy += Math.sin(a) * m; }
    const hue = (((Math.atan2(qy, qx) / 6.28318 + d.hue * 0.35) % 1) + 1) % 1;
    return `hsl(${(hue * 360).toFixed(0)} 80% 62%)`;
  }

  // Nearest-neighbour niche archive. In-radius newcomers replace a niche only if fitter; novel ones open a niche.
  // `halfLifeMs` (server) makes claimed fitness decay with age, so stale or exaggerated entries get displaced.
  class GeneBank {
    constructor({ cap = 400, halfLifeMs = 0, fastPrune = false } = {}) {
      Object.assign(this, { cap, halfLifeMs, fastPrune, entries: [], radius: 0.6, inserts: 0, replaced: 0, nextId: 1 });
    }
    dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
    score(e, now = Date.now()) { return this.halfLifeMs ? e.fitness * 0.5 ** ((now - e.t) / this.halfLifeMs) : e.fitness; }
    nearest(v) {
      let best = -1, bd = Infinity;
      this.entries.forEach((e, i) => { const d = this.dist(v, e.vec); if (d < bd) { bd = d; best = i; } });
      return [best, bd];
    }
    add(genome, meta) { return this.insert(genome, meta.kids + meta.age / 2000, meta.kids, meta.origin || 'local'); }
    insert(genome, fitness, kids, origin = 'local', t = Date.now()) {
      const dec = decodeGenome(genome);
      const vec = embed(genome, dec);
      const [i, d] = this.nearest(vec);
      this.inserts++;
      const entry = { genome, dec, vec, fitness, kids, origin, color: phenoColor(dec), t };
      if (i >= 0 && d < this.radius) {
        if (fitness > this.score(this.entries[i])) {
          Object.assign(this.entries[i], entry);
          this.entries[i].dirty = true;
          this.replaced++;
          return this.entries[i];
        }
        return null;
      }
      const e = { id: this.nextId++, ...entry, dirty: true };
      this.entries.push(e);
      if (this.entries.length > this.cap) this.prune();
      return this.entries.includes(e) ? e : null;
    }
    prune() {
      const E = this.entries;
      if (this.fastPrune) {          // O(n): drop the weakest (age-weighted) niche, and slowly widen niches
        let wi = 0, ws = Infinity; const now = Date.now();
        E.forEach((e, i) => { const s = this.score(e, now); if (s < ws) { ws = s; wi = i; } });
        E.splice(wi, 1);
        this.radius *= 1.002;
        return;
      }
      let bi = 0, bj = 1, bd = Infinity;  // drop the weaker member of the closest pair, and widen the niche radius
      for (let i = 0; i < E.length; i++) for (let j = i + 1; j < E.length; j++) {
        const d = this.dist(E[i].vec, E[j].vec); if (d < bd) { bd = d; bi = i; bj = j; }
      }
      E.splice(E[bi].fitness < E[bj].fitness ? bi : bj, 1);
      this.radius = Math.max(this.radius, bd * 1.02);
    }
    pca() { // top-2 principal components by power iteration
      const E = this.entries; if (E.length < 3) return null;
      const D = E[0].vec.length, mean = new Float64Array(D);
      for (const e of E) for (let i = 0; i < D; i++) mean[i] += e.vec[i] / E.length;
      const X = E.map(e => Array.from(e.vec, (x, i) => x - mean[i]));
      const comps = [];
      for (let k = 0; k < 2; k++) {
        let v = Array.from({ length: D }, (_, i) => Math.sin(i * 7.1 + k * 3.3));
        for (let it = 0; it < 30; it++) {
          const nv = new Array(D).fill(0);
          for (const x of X) { let s = 0; for (let i = 0; i < D; i++) s += x[i] * v[i]; for (let i = 0; i < D; i++) nv[i] += s * x[i]; }
          for (const c of comps) { let s = 0; for (let i = 0; i < D; i++) s += nv[i] * c[i]; for (let i = 0; i < D; i++) nv[i] -= s * c[i]; }
          const n = Math.hypot(...nv) || 1; v = nv.map(x => x / n);
        }
        comps.push(v);
      }
      return X.map(x => comps.map(c => x.reduce((s, xi, i) => s + xi * c[i], 0)));
    }
    toJSON() {
      return { v: 4, F, HMAX, GS, radius: this.radius, entries: this.entries.map(e => ({
        id: e.id, fitness: e.fitness, kids: e.kids, origin: e.origin, t: e.t, g: genomeToB64(e.genome) })) };
    }
    serialize() { return JSON.stringify(this.toJSON()); }
    load(json) {
      const o = typeof json === 'string' ? JSON.parse(json) : json;
      if (o.F !== F || o.HMAX !== HMAX || o.GS !== GS) throw new Error(`Gene bank was saved with F=${o.F}, HMAX=${o.HMAX}; this run uses F=${F}, HMAX=${HMAX}.`);
      this.entries = [];
      for (const e of o.entries) {
        const genome = b64ToGenome(e.g);
        if (!genome || validateGenome(genome)) continue;
        const dec = decodeGenome(genome);
        this.entries.push({ id: e.id, genome, dec, vec: embed(genome, dec), fitness: e.fitness, kids: e.kids,
          origin: e.origin || 'local', color: phenoColor(dec), t: e.t || Date.now() });
      }
      this.radius = o.radius || 0.6;
      this.nextId = Math.max(1, ...this.entries.map(e => e.id + 1));
    }
  }

  return { decodeGenome, validateGenome, embed, phenoColor, weights, GeneBank };
}

// float -> IEEE half (round to nearest), packed two per u32 like WGSL's pack2x16float
export function toHalf(v) {
  f32view.setFloat32(0, v, true);
  const x = f32view.getUint32(0, true);
  const sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) { if (e < -10) return sign; const m = (mant | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
  return sign | (e << 10) | ((mant + 0x1000) >> 13);
}

// A visitor's strain: a stable 32-bit tag (carried in the genome's lineage word), a hue, and a founder genome
// generated deterministically from the tag, with the same distributions as the GPU's random founders.
export function strainTagOf(visitorId) {
  let h = 0x811c9dc5;
  for (let i = 0; i < visitorId.length; i++) { h ^= visitorId.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h | 1) >>> 0;
}
export function strainHue(tag) { return ((tag >>> 8) & 0xffff) / 65536; }
export function makeFounder(L, tag) {
  const { F, HMAX, FS, KE, NT, HDR, EOFF, WOFF, WWORDS, GS } = L;
  let s = tag || 1;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
  const rndn = () => (rnd() + rnd() + rnd() - 1.5) * 2;
  const g = new Uint32Array(GS);
  const f = (k, v) => { f32view.setFloat32(0, v, true); g[k] = f32view.getUint32(0, true); };
  g[0] = 2 + Math.min(Math.floor(rnd() * (HMAX - 1)), HMAX - 2);
  f(1, 1 + rnd() * 6); f(2, 0.01 + rnd() * 0.06); f(3, strainHue(tag));
  g[4] = tag; g[5] = 0; f(6, 500 + rnd() * 2500); f(7, 0.01 + rnd() * 0.2);
  f(8, rnd() < 0.5 ? 0 : rnd() * 0.05);
  g[9] = strainTagOf('bond:' + tag);
  for (let c = 0; c < F; c++) {
    f(HDR + c, c === 0 ? 0.02 + rnd() * 0.1 : rnd() * 0.12);
    f(HDR + F + c, c === 0 ? 0.5 + rnd() * 0.5 : Math.max(-1.5, Math.min(1.5, rndn() * 0.5)));
    f(HDR + 2 * F + c, Math.max(-4.5, Math.min(4.5, rndn() * 1.5)));
  }
  for (let k = 0; k < KE; k++) {
    const t = rnd() < 0.5 ? 0 : 1 + Math.min(Math.floor(rnd() * (NT - 1)), NT - 2);
    g[EOFF + k * 4] = (t | (Math.min(Math.floor(rnd() * FS), F) << 8) | (Math.min(Math.floor(rnd() * F), F - 1) << 16)) >>> 0;
    f(EOFF + k * 4 + 1, Math.max(-1.8, Math.min(1.8, rndn() * 0.6))); f(EOFF + k * 4 + 2, rnd()); f(EOFF + k * 4 + 3, Math.max(-3, Math.min(3, rndn())));
  }
  for (let k = 0; k < WWORDS; k++) g[WOFF + k] = (toHalf(rndn() * 0.45) | (toHalf(rndn() * 0.45) << 16)) >>> 0;
  return g;
}

// ----- strain identity: every strain has a signature effect gene (fixed by its tag), a name, and a body plan -----
// [name noun, what the signature does, body glyph]
export const SIGNATURES = [null,
  ['Seekers', 'climbs chemical gradients', '☄'], ['Vortex', 'spins around what it senses', '☄'],
  ['Contour', 'rides chemical contour lines', '▬'], ['Ferment', 'turns one chemical into another', '●'],
  ['Darters', 'jets forward in bursts', '☄'], ['Holdfast', 'roots itself in place', '◆'],
  ['Crush', 'shoves its neighbours aside', '●'], ['Pulsar', 'broadcasts chemical pulses', '●'],
  ['Leech', 'drains energy from neighbours', '●'], ['Relay', 'passes signals between channels', '●'],
  ['Ember', 'runs hot and restless', '●'], ['Carapace', 'shrugs off toxins', '◆'],
  ['Drifters', 'wanders on its own current', '☄'], ['Scavenger', 'steals genes from cellmates', '✦'],
  ['Spore', 'injects its genes into others', '✦'], ['Colony', 'bonds with kin and shares energy', '◯'],
  ['Membrane', 'settles on borders and builds walls', '▬']];
const ADJ = ['Ochre', 'Velvet', 'Ashen', 'Gilded', 'Pale', 'Crimson', 'Umber', 'Feral', 'Glass', 'Hollow', 'Iron', 'Lunar',
  'Mire', 'Noble', 'Opal', 'Quiet', 'Rust', 'Salt', 'Thorn', 'Verdant', 'Woven', 'Amber', 'Bitter', 'Cinder', 'Dusk',
  'Ember', 'Frost', 'Grey', 'Hazel', 'Ivory', 'Jade', 'Kelp', 'Lichen', 'Moss', 'Night', 'Onyx', 'Pearl', 'Quartz',
  'Russet', 'Sable', 'Tidal', 'Umbral', 'Violet', 'Wild', 'Xeric', 'Yellow', 'Zinc', 'Cobalt'];
export function strainSignature(tag) { return 1 + (strainTagOf('sig:' + (tag >>> 0)) >>> 3) % (EFFECTS.length - 1); }
export function strainName(tag) {
  return ADJ[(strainTagOf('name:' + (tag >>> 0)) >>> 5) % ADJ.length] + ' ' + SIGNATURES[strainSignature(tag)][0];
}
// Makes the signature gene the genome's strongest effect, so the strain's particles carry its body plan and trait.
// Deterministic per tag: every dish applies the same signature to the same strain.
export function applySignature(L, g, tag) {
  const { EOFF, KE, F, FS } = L;
  const out = Uint32Array.from(g);
  const sig = strainSignature(tag);
  let s = strainTagOf('sigp:' + (tag >>> 0));
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
  const get = k => { f32view.setUint32(0, out[k], true); return f32view.getFloat32(0, true); };
  const put = (k, v) => { f32view.setFloat32(0, v, true); out[k] = f32view.getUint32(0, true); };
  let slot = -1, maxOther = 0, weakest = 0, wv = Infinity;
  for (let k = 0; k < KE; k++) {
    const t = out[EOFF + k * 4] & 255, p = Math.abs(get(EOFF + k * 4 + 1));
    if (t === sig && slot < 0) { slot = k; continue; }
    if (t) maxOther = Math.max(maxOther, p);
    const v = t ? p : -1;
    if (v < wv) { wv = v; weakest = k; }
  }
  const a = Math.floor(rnd() * F), b = Math.floor(rnd() * F), sign = rnd() < 0.75 ? 1 : -1, p1 = rnd(), p2 = (rnd() - 0.5) * 2;
  if (slot < 0) {
    slot = weakest;
    out[EOFF + slot * 4] = (sig | (Math.min(a, FS - 1) << 8) | (Math.min(b, F - 1) << 16)) >>> 0;
    put(EOFF + slot * 4 + 2, p1); put(EOFF + slot * 4 + 3, p2);
    put(EOFF + slot * 4 + 1, sign * 0.9);
  }
  const cur = get(EOFF + slot * 4 + 1);
  const mag = Math.min(1.8, Math.max(Math.abs(cur), maxOther + 0.15, 0.9));
  put(EOFF + slot * 4 + 1, (cur < 0 ? -1 : 1) * mag);
  return out;
}
// A readable profile of a strain's champion: signature, body, supporting genes, brain, learning, diet.
export function strainProfile(dec, tag) {
  const sig = strainSignature(tag);
  const others = dec.effects.filter(e => e.t > 0 && e.t !== sig).sort((x, y) => Math.abs(y.p0) - Math.abs(x.p0)).map(e => EFFECTS[e.t]);
  const F = dec.chem.egain.length;
  let diet = 0, dv = -Infinity, poison = 0, pv = Infinity;
  for (let c = 0; c < F; c++) { const v = dec.chem.egain[c] * dec.chem.absorb[c]; if (v > dv) { dv = v; diet = c; } if (v < pv) { pv = v; poison = c; } }
  const ch = c => (c === 0 ? 'energy' : 'c' + c);
  return { name: strainName(tag), sig: EFFECTS[sig], noun: SIGNATURES[sig][0], does: SIGNATURES[sig][1], glyph: SIGNATURES[sig][2],
    also: [...new Set(others)].slice(0, 2), brain: dec.H, learns: dec.eta > 0.0005, gen: dec.gen, diet: ch(diet), poison: ch(poison) };
}
