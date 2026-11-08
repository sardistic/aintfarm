import { SIGNAL_COLS, SIGNAL_ROWS, SIGNAL_CHANNELS } from './signals.js';

const palette = [[62, 215, 237], [255, 151, 82], [187, 112, 255], [130, 233, 163]];

export function createVisualField() {
  const canvas = document.createElement('canvas');
  canvas.width = SIGNAL_COLS; canvas.height = SIGNAL_ROWS;
  const context = canvas.getContext('2d');
  const image = context.createImageData(SIGNAL_COLS, SIGNAL_ROWS);

  function update(world) {
    const values = world.hormones, pixels = image.data;
    for (let i = 0; i < values.length; i += SIGNAL_CHANNELS) {
      const a = values[i], b = values[i + 1], c = values[i + 2], d = values[i + 3];
      const sum = a + b + c + d;
      pixels[i] = Math.min(255, a * palette[0][0] + b * palette[1][0] + c * palette[2][0] + d * palette[3][0]);
      pixels[i + 1] = Math.min(255, a * palette[0][1] + b * palette[1][1] + c * palette[2][1] + d * palette[3][1]);
      pixels[i + 2] = Math.min(255, a * palette[0][2] + b * palette[1][2] + c * palette[2][2] + d * palette[3][2]);
      pixels[i + 3] = Math.min(190, sum * 48);
    }
    context.putImageData(image, 0, 0);
  }

  return { canvas, update };
}
