// Four inheritable colony signals. Concentrations drive movement and are also
// rendered by the visual field, so the visible patterns affect agent behavior.
export const SIGNAL_SCALE = 4, SIGNAL_COLS = 250, SIGNAL_ROWS = 163, SIGNAL_CHANNELS = 4;
const LENGTH = SIGNAL_COLS * SIGNAL_ROWS * SIGNAL_CHANNELS;
const diffusion = [.22, .17, .12, .08];
const persistence = [.94, .95, .96, .97];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function makeSignals(world) {
  world.hormones = new Float32Array(LENGTH);
  world.hormonesNext = new Float32Array(LENGTH);
}

export function signalAt(world, px, py, channel) {
  const x = clamp(Math.floor(px / SIGNAL_SCALE), 0, SIGNAL_COLS - 1);
  const y = clamp(Math.floor(py / SIGNAL_SCALE), 0, SIGNAL_ROWS - 1);
  return world.hormones[(y * SIGNAL_COLS + x) * SIGNAL_CHANNELS + channel];
}

export function emitSignals(world, px, py, signature, amount = 1) {
  const x = Math.floor(px / SIGNAL_SCALE), y = Math.floor(py / SIGNAL_SCALE);
  if (x < 1 || x >= SIGNAL_COLS - 1 || y < 1 || y >= SIGNAL_ROWS - 1) return;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const at = ((y + dy) * SIGNAL_COLS + x + dx) * SIGNAL_CHANNELS;
    const strength = amount * (dx === 0 && dy === 0 ? 1 : .27);
    for (let c = 0; c < SIGNAL_CHANNELS; c++)
      world.hormones[at + c] = Math.min(5, world.hormones[at + c] + strength * (signature[c] + 1) * .5);
  }
}

export function stepSignals(world) {
  let values = world.hormones, next = world.hormonesNext;
  next.fill(0);
  for (let y = 1; y < SIGNAL_ROWS - 1; y++) for (let x = 1; x < SIGNAL_COLS - 1; x++) {
    const at = (y * SIGNAL_COLS + x) * SIGNAL_CHANNELS;
    for (let c = 0; c < SIGNAL_CHANNELS; c++) {
      const neighbors = (values[at - SIGNAL_CHANNELS + c] + values[at + SIGNAL_CHANNELS + c]
        + values[at - SIGNAL_COLS * SIGNAL_CHANNELS + c]
        + values[at + SIGNAL_COLS * SIGNAL_CHANNELS + c]) * .25;
      next[at + c] = Math.min(5, (values[at + c] * (1 - diffusion[c])
        + neighbors * diffusion[c]) * persistence[c]);
    }
    // Complementary compounds react into secondary signals. Emission and
    // receptor mutations change which of these local products an agent seeks.
    const reaction02 = Math.min(next[at], next[at + 2]) * .025;
    const reaction13 = Math.min(next[at + 1], next[at + 3]) * .02;
    next[at] -= reaction02; next[at + 2] -= reaction02;
    next[at + 1] += reaction02 * .7 - reaction13;
    next[at + 3] += reaction13 * .8;
  }
  world.hormones = next; world.hormonesNext = values;
  for (const ant of world.ants) emitSignals(world, ant.bodyX, ant.bodyY, ant.signature,
    (ant.foodCargo ? .95 : .55) * (.4 + ant.traits.secretion * 1.2));
}

export function chemicalResponse(receiver, emitter) {
  let response = 0;
  for (let c = 0; c < SIGNAL_CHANNELS; c++) response += receiver.receptor[c] * emitter.signature[c];
  return response / SIGNAL_CHANNELS;
}

export function signalPreference(world, ant, px, py) {
  let force = 0;
  for (let c = 0; c < SIGNAL_CHANNELS; c++)
    force += ant.receptor[c] * signalAt(world, px, py, c);
  return force * .25;
}
