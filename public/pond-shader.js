import { WIDTH, HEIGHT, CELL } from './evolution.js';
import { SIGNAL_COLS, SIGNAL_ROWS, SIGNAL_CHANNELS } from './signals.js';

const vertexSource = `
attribute vec2 position;
varying vec2 uv;
void main() {
  uv = position * .5 + .5;
  gl_Position = vec4(position, 0., 1.);
}`;

const fragmentSource = `
precision highp float;
varying vec2 uv;
uniform sampler2D terrain;
uniform sampler2D chemistry;
uniform sampler2D reaction;
uniform sampler2D trails;
uniform vec2 worldSize;
uniform vec2 terrainSize;
uniform vec2 chemicalSize;
uniform float time;

float hash21(vec2 p) {
  p = fract(p * vec2(.1031, .1030));
  p += dot(p, p.yx + 33.33);
  return fract((p.x + p.y) * p.x);
}
vec2 hash22(vec2 p) {
  return vec2(hash21(p), hash21(p + 17.71));
}
float field(vec2 p) {
  vec2 q = p * .012;
  return sin(q.x * 1.7 + sin(q.y * 1.25 + time * .32) * 1.9)
       + sin(q.y * 2.0 - cos(q.x * 1.35 - time * .23) * 1.6)
       + .45 * sin((q.x + q.y) * 3.5 + time * .4);
}
void main() {
  vec2 pos = vec2(uv.x, 1. - uv.y) * worldSize;
  vec2 cell = floor(pos * .5);
  vec2 gridUv = (cell + .5) / terrainSize;
  float solid = step(.5, texture2D(terrain, gridUv).r);
  vec4 substrate = texture2D(terrain, pos / worldSize);
  float nutrient = substrate.g;
  vec4 chem = texture2D(chemistry, pos / worldSize);
  float film = texture2D(reaction, pos / worldSize).g;
  float filmEdge = abs(film - texture2D(reaction, (pos + vec2(2., 0.)) / worldSize).g)
    + abs(film - texture2D(reaction, (pos + vec2(0., 2.)) / worldSize).g);
  float concentration = dot(chem, vec4(.65, .65, .65, .65));

  // Each occupied terrain cell has a distinct, slightly offset mineral grain.
  vec2 local = fract(pos * .5);
  float grainSeed = hash21(cell);
  vec2 grainCenter = vec2(.5) + (hash22(cell + 5.3) - .5) * .32;
  float grainDistance = length(local - grainCenter);
  float grain = 1. - smoothstep(.43 + grainSeed * .07, .56 + grainSeed * .04, grainDistance);
  vec3 mineral = mix(vec3(.012, .032, .043),
    mix(vec3(.055, .105, .108), vec3(.17, .23, .20), grainSeed), grain);
  float glint = 1. - smoothstep(.06, .16, length(local - grainCenter - vec2(-.16, -.17)));
  mineral += vec3(.045, .095, .085) * glint * grain * grainSeed;

  // Refracted water, cellular rings, and colored reaction fronts remain visible
  // even before the first organisms have emitted much chemistry.
  float current = field(pos);
  float lace = pow(1. - smoothstep(.04, .32, abs(current)), 2.);
  float ripple = pow(1. - smoothstep(.02, .23, abs(sin(current * 2.1 + time * .13))), 3.);
  vec2 sporeCell = floor(pos / 23.);
  vec2 sporeLocal = fract(pos / 23.);
  vec2 sporeCenter = .18 + .64 * hash22(sporeCell);
  float sporeDistance = length(sporeLocal - sporeCenter);
  float sporeChance = step(.63, hash21(sporeCell + 41.1));
  float spore = sporeChance * (1. - smoothstep(.035, .14, sporeDistance));
  float membrane = sporeChance * (1. - smoothstep(.015, .065, abs(sporeDistance - .24)));
  float sporeLife = .45 + .35 * sin(time * .45 + hash21(sporeCell) * 6.283)
    + min(.5, concentration * .45);
  vec3 water = vec3(.012, .052, .064);
  water += vec3(.025, .13, .15) * lace;
  water += vec3(.014, .055, .075) * ripple;
  water += vec3(.095, .25, .22) * spore * sporeLife;
  water += vec3(.012, .067, .063) * membrane * (.2 + sporeLife);

  vec3 dye = chem.r * vec3(.12, .75, .73)
           + chem.g * vec3(1., .38, .13)
           + chem.b * vec3(.65, .28, .95)
           + chem.a * vec3(.43, .83, .25);
  float edge = 0.;
  edge += length(chem - texture2D(chemistry, (pos + vec2(3., 0.)) / worldSize));
  edge += length(chem - texture2D(chemistry, (pos + vec2(0., 3.)) / worldSize));
  water += dye * (.42 + lace * .55 + edge * .9);
  water += dye * concentration * ripple * .34;

  // The active film seeps through exposed grain edges but leaves grains crisp.
  vec3 color = mix(water, mineral, solid);
  color += solid * (1. - grain) * dye * .08;
  float colony = nutrient * (.78 + .22 * sin(time * .28 + field(pos * 1.4) * 2.));
  float front = 1. - smoothstep(.07, .32,
    abs(sin(nutrient * 13. - time * .25 + field(pos * 1.7) * .7)));
  vec3 colonyColor = mix(vec3(.10, .72, .50), vec3(.75, .27, .67), substrate.b);
  color += solid * colonyColor * colony * (.42 + front * .28) * (grain * .55 + .45);
  color += (1. - solid) * nutrient * colonyColor * .16;
  color += colonyColor * (film * .14 + filmEdge * .3) * (solid * .72 + .28);
  color += texture2D(trails, uv).rgb * (solid * .38 + .5);
  float vessel = smoothstep(.03, .31, min(min(pos.x / worldSize.x,
    1. - pos.x / worldSize.x), min(pos.y / worldSize.y, 1. - pos.y / worldSize.y)));
  color *= mix(.54, 1., vessel);
  color = color / (1. + color * .48);
  gl_FragColor = vec4(pow(max(color, 0.), vec3(.91)), 1.);
}`;

const reactionSource = `
precision highp float;
varying vec2 uv;
uniform sampler2D state;
uniform sampler2D terrain;
uniform sampler2D chemistry;
uniform vec2 pixel;
uniform float time;
void main() {
  vec2 value = texture2D(state, uv).rg;
  vec2 lap = texture2D(state, uv + vec2(pixel.x, 0.)).rg
    + texture2D(state, uv - vec2(pixel.x, 0.)).rg
    + texture2D(state, uv + vec2(0., pixel.y)).rg
    + texture2D(state, uv - vec2(0., pixel.y)).rg - 4. * value;
  float food = texture2D(terrain, uv).g;
  vec4 chemical = texture2D(chemistry, uv);
  float signal = dot(chemical, vec4(.25));
  float feed = .026 + food * .015 + signal * .008;
  float kill = .055 + .003 * sin(uv.x * 34. + uv.y * 21.) - food * .006;
  float reaction = value.r * value.g * value.g;
  float a = clamp(value.r + .85 * lap.r - reaction + feed * (1. - value.r), 0., 1.);
  float b = clamp(value.g + .42 * lap.g + reaction - (kill + feed) * value.g
    + max(0., signal - .08) * .002, 0., 1.);
  gl_FragColor = vec4(a, b, 0., 1.);
}`;

const trailSource = `
precision mediump float;
varying vec2 uv;
uniform sampler2D previousTrail;
uniform vec2 pixel;
void main() {
  vec4 center = texture2D(previousTrail, uv);
  vec4 spread = texture2D(previousTrail, uv + vec2(pixel.x, 0.))
    + texture2D(previousTrail, uv - vec2(pixel.x, 0.))
    + texture2D(previousTrail, uv + vec2(0., pixel.y))
    + texture2D(previousTrail, uv - vec2(0., pixel.y));
  gl_FragColor = (center * .87 + spread * .0325) * .935;
}`;

const moteVertexSource = `
attribute vec2 motePosition;
attribute vec3 moteColor;
attribute float moteAlpha;
uniform vec2 worldSize;
uniform float pointSize;
varying vec3 color;
varying float alpha;
void main() {
  color = moteColor;
  alpha = moteAlpha;
  gl_Position = vec4(motePosition.x / worldSize.x * 2. - 1.,
    1. - motePosition.y / worldSize.y * 2., 0., 1.);
  gl_PointSize = pointSize;
}`;

const moteFragmentSource = `
precision mediump float;
varying vec3 color;
varying float alpha;
uniform float pointMode;
uniform float intensity;
void main() {
  float coverage = 1.;
  if (pointMode > .5) {
    float radius = length(gl_PointCoord - .5) * 2.;
    coverage = 1. - smoothstep(.18, 1., radius);
  }
  gl_FragColor = vec4(color, alpha * coverage * intensity);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(error);
  }
  return shader;
}

function texture(gl, width, height, filter) {
  const handle = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, handle);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA,
    gl.UNSIGNED_BYTE, null);
  return handle;
}

export function createPondShader(canvas) {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) { canvas.dataset.shaderError = 'WebGL context unavailable'; return null; }
  let program;
  try {
    program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
    gl.bindAttribLocation(program, 0, 'position');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  } catch (error) {
    console.warn('Pond shader unavailable:', error);
    canvas.dataset.shaderError = String(error);
    return null;
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const terrainWidth = WIDTH / CELL, terrainHeight = HEIGHT / CELL;
  const terrainPixels = new Uint8Array(terrainWidth * terrainHeight * 4);
  const chemistryPixels = new Uint8Array(SIGNAL_COLS * SIGNAL_ROWS * 4);
  gl.activeTexture(gl.TEXTURE0);
  const terrainTexture = texture(gl, terrainWidth, terrainHeight, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(program, 'terrain'), 0);
  gl.activeTexture(gl.TEXTURE1);
  const chemistryTexture = texture(gl, SIGNAL_COLS, SIGNAL_ROWS, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(program, 'chemistry'), 1);
  const reactionWidth = WIDTH / CELL, reactionHeight = HEIGHT / CELL;
  const reactionTextures = [], reactionBuffers = [];
  for (let i = 0; i < 2; i++) {
    gl.activeTexture(gl.TEXTURE2);
    const handle = texture(gl, reactionWidth, reactionHeight, gl.NEAREST);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, handle, 0);
    reactionTextures.push(handle); reactionBuffers.push(framebuffer);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.uniform1i(gl.getUniformLocation(program, 'reaction'), 2);
  let reactionProgram;
  try {
    reactionProgram = gl.createProgram();
    gl.attachShader(reactionProgram, compile(gl, gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(reactionProgram, compile(gl, gl.FRAGMENT_SHADER, reactionSource));
    gl.bindAttribLocation(reactionProgram, 0, 'position');
    gl.linkProgram(reactionProgram);
    if (!gl.getProgramParameter(reactionProgram, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(reactionProgram));
    for (const framebuffer of reactionBuffers) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Reaction texture framebuffer is incomplete');
    }
  } catch (error) {
    console.warn('Pond reaction shader unavailable:', error);
    canvas.dataset.shaderError = String(error);
    return null;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  gl.useProgram(reactionProgram);
  gl.uniform1i(gl.getUniformLocation(reactionProgram, 'state'), 2);
  gl.uniform1i(gl.getUniformLocation(reactionProgram, 'terrain'), 0);
  gl.uniform1i(gl.getUniformLocation(reactionProgram, 'chemistry'), 1);
  gl.uniform2f(gl.getUniformLocation(reactionProgram, 'pixel'), 1 / reactionWidth, 1 / reactionHeight);
  const reactionTime = gl.getUniformLocation(reactionProgram, 'time');
  let moteProgram = null, moteBuffer = null, motePointMode = null, motePointSize = null, moteIntensity = null;
  try {
    moteProgram = gl.createProgram();
    gl.attachShader(moteProgram, compile(gl, gl.VERTEX_SHADER, moteVertexSource));
    gl.attachShader(moteProgram, compile(gl, gl.FRAGMENT_SHADER, moteFragmentSource));
    gl.bindAttribLocation(moteProgram, 0, 'motePosition');
    gl.bindAttribLocation(moteProgram, 1, 'moteColor');
    gl.bindAttribLocation(moteProgram, 2, 'moteAlpha');
    gl.linkProgram(moteProgram);
    if (!gl.getProgramParameter(moteProgram, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(moteProgram));
    moteBuffer = gl.createBuffer();
    gl.useProgram(moteProgram);
    gl.uniform2f(gl.getUniformLocation(moteProgram, 'worldSize'), WIDTH, HEIGHT);
    motePointMode = gl.getUniformLocation(moteProgram, 'pointMode');
    motePointSize = gl.getUniformLocation(moteProgram, 'pointSize');
    moteIntensity = gl.getUniformLocation(moteProgram, 'intensity');
  } catch (error) {
    console.warn('Motile culture rendering unavailable:', error);
    moteProgram = null;
  }
  const trailWidth = WIDTH, trailHeight = HEIGHT;
  const trailTextures = [], trailBuffers = [];
  let trailProgram;
  try {
    for (let i = 0; i < 2; i++) {
      gl.activeTexture(gl.TEXTURE3);
      const handle = texture(gl, trailWidth, trailHeight, gl.LINEAR);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, handle, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Trail texture framebuffer is incomplete');
      trailTextures.push(handle); trailBuffers.push(framebuffer);
    }
    trailProgram = gl.createProgram();
    gl.attachShader(trailProgram, compile(gl, gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(trailProgram, compile(gl, gl.FRAGMENT_SHADER, trailSource));
    gl.bindAttribLocation(trailProgram, 0, 'position');
    gl.linkProgram(trailProgram);
    if (!gl.getProgramParameter(trailProgram, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(trailProgram));
  } catch (error) {
    console.warn('Pond trails unavailable:', error);
    canvas.dataset.shaderError = String(error);
    return null;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  gl.useProgram(trailProgram);
  gl.uniform1i(gl.getUniformLocation(trailProgram, 'previousTrail'), 3);
  gl.uniform2f(gl.getUniformLocation(trailProgram, 'pixel'), 1 / trailWidth, 1 / trailHeight);
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, 'trails'), 3);
  gl.uniform2f(gl.getUniformLocation(program, 'worldSize'), WIDTH, HEIGHT);
  gl.uniform2f(gl.getUniformLocation(program, 'terrainSize'), terrainWidth, terrainHeight);
  gl.uniform2f(gl.getUniformLocation(program, 'chemicalSize'), SIGNAL_COLS, SIGNAL_ROWS);
  const timeLocation = gl.getUniformLocation(program, 'time');
  let terrainRevision = -1, nutrientClock = -1, lastWorld = null, currentReaction = 0, currentTrail = 0;
  function clearTrails() {
    gl.clearColor(0, 0, 0, 0);
    for (const framebuffer of trailBuffers) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    currentTrail = 0;
  }
  clearTrails();
  function seedReaction() {
    const pixels = new Uint8Array(reactionWidth * reactionHeight * 4);
    for (let y = 0; y < reactionHeight; y++) for (let x = 0; x < reactionWidth; x++) {
      const at = (y * reactionWidth + x) * 4;
      const island = ((Math.floor(x / 4) * 73856093) ^ (Math.floor(y / 4) * 19349663)) >>> 0;
      pixels[at] = pixels[at + 3] = 255;
      pixels[at + 1] = terrainPixels[at + 1] > 45 && island % 11 < 3 ? 155 : 0;
    }
    for (const handle of reactionTextures) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, handle);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, reactionWidth, reactionHeight,
        gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    }
    currentReaction = 0;
  }
  function stepReaction(seconds) {
    const next = 1 - currentReaction;
    gl.bindFramebuffer(gl.FRAMEBUFFER, reactionBuffers[next]);
    gl.viewport(0, 0, reactionWidth, reactionHeight);
    gl.useProgram(reactionProgram);
    gl.uniform1f(reactionTime, seconds);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, reactionTextures[currentReaction]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    currentReaction = next;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function prepareMotes(culture) {
    gl.useProgram(moteProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, moteBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, culture.vertices, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1); gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 8);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 24, 20);
  }
  function paintMotes(culture, pointSize, intensity = 1) {
    gl.useProgram(moteProgram);
    gl.uniform1f(motePointSize, pointSize);
    gl.uniform1f(moteIntensity, intensity);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.uniform1f(motePointMode, 0);
    gl.drawArrays(gl.LINES, 0, culture.count * 2);
    gl.uniform1f(motePointMode, 1);
    gl.drawArrays(gl.POINTS, 0, culture.count * 2);
    gl.disable(gl.BLEND);
  }
  function stepTrails(culture) {
    const next = 1 - currentTrail;
    gl.bindFramebuffer(gl.FRAMEBUFFER, trailBuffers[next]);
    gl.viewport(0, 0, trailWidth, trailHeight);
    gl.useProgram(trailProgram);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, trailTextures[currentTrail]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (culture && moteProgram) {
      prepareMotes(culture);
      paintMotes(culture, 1.8, .08);
      gl.disableVertexAttribArray(1); gl.disableVertexAttribArray(2);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    }
    currentTrail = next;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function resize(width, height) {
    if (canvas.width === width && canvas.height === height) return;
    canvas.width = width; canvas.height = height;
    gl.viewport(0, 0, width, height);
  }
  function updateTerrain(world, seconds) {
    const nextClock = Math.floor(seconds * 2);
    if (terrainRevision === world.packedRevision && nutrientClock === nextClock) return;
    for (let i = 0; i < world.sand.length; i++) {
      const offset = i * 4, filled = world.sand[i] === 1 ? 255 : 0;
      terrainPixels[offset] = filled;
      terrainPixels[offset + 1] = terrainPixels[offset + 2] = 0;
      terrainPixels[offset + 3] = 255;
    }
    for (const food of world.foods) {
      if (food.amount <= 0) continue;
      const radius = food.kind === 'bloom' ? 42 : 34;
      const power = Math.min(1, food.amount / (food.max || 65));
      for (let y = Math.max(0, food.y - radius); y <= Math.min(terrainHeight - 1, food.y + radius); y++)
        for (let x = Math.max(0, food.x - radius); x <= Math.min(terrainWidth - 1, food.x + radius); x++) {
          const distance = Math.hypot(x - food.x, y - food.y) / radius;
          if (distance >= 1) continue;
          const offset = (y * terrainWidth + x) * 4;
          const intensity = Math.round(240 * power * (1 - distance * distance) ** 1.4);
          if (intensity > terrainPixels[offset + 1]) {
            terrainPixels[offset + 1] = intensity;
            terrainPixels[offset + 2] = food.kind === 'bloom' ? 255
              : food.kind === 'carcass' ? 140 : 35 + ((food.x * 17 + food.y * 31) % 150);
          }
        }
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, terrainTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, terrainWidth, terrainHeight,
      gl.RGBA, gl.UNSIGNED_BYTE, terrainPixels);
    terrainRevision = world.packedRevision;
    nutrientClock = nextClock;
  }
  function updateChemistry(world) {
    const values = world.hormones;
    for (let i = 0; i < values.length; i += SIGNAL_CHANNELS) {
      chemistryPixels[i] = 155 * values[i] / (1 + values[i]);
      chemistryPixels[i + 1] = 155 * values[i + 1] / (1 + values[i + 1]);
      chemistryPixels[i + 2] = 155 * values[i + 2] / (1 + values[i + 2]);
      chemistryPixels[i + 3] = 155 * values[i + 3] / (1 + values[i + 3]);
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, chemistryTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SIGNAL_COLS, SIGNAL_ROWS,
      gl.RGBA, gl.UNSIGNED_BYTE, chemistryPixels);
  }
  function draw(world, seconds, updateSignals = true, culture = null) {
    updateTerrain(world, seconds);
    if (updateSignals) updateChemistry(world);
    if (lastWorld !== world) { seedReaction(); clearTrails(); lastWorld = world; }
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    stepReaction(seconds);
    stepTrails(culture);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, reactionTextures[currentReaction]);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, trailTextures[currentTrail]);
    gl.uniform1f(timeLocation, seconds);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (culture && moteProgram) {
      prepareMotes(culture);
      paintMotes(culture, Math.max(1.5, canvas.width / WIDTH * 1.85));
      gl.disableVertexAttribArray(1); gl.disableVertexAttribArray(2);
    }
  }
  return { canvas, resize, draw };
}
