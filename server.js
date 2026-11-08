import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, relative } from 'node:path';
import { randomUUID } from 'node:crypto';

const root = join(process.cwd(), 'public');
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png' };
const questions = {
  chambers: { type: 'noul', instructions: 'Does `artwork` show more than one distinct underground chamber?' },
  branches: { type: 'noul', instructions: 'Do the underground tunnels in `artwork` branch into separate paths?' },
  balance: { type: 'noul', instructions: 'Is the underground nest in `artwork` spread across both sides of the entrance?' }
};

async function critique(req, res) {
  if (!process.env.TYPESAFE_API_KEY) return send(res, 503, { error: 'Set TYPESAFE_API_KEY on the server to enable Jev.' });
  let body = '';
  for await (const chunk of req) {
    body += chunk;
    if (body.length > 2_000_000) return send(res, 413, { error: 'Artwork is too large.' });
  }
  let image;
  try { image = JSON.parse(body).image; } catch { return send(res, 400, { error: 'Invalid JSON.' }); }
  if (typeof image !== 'string' || !/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(image)) return send(res, 400, { error: 'Send a JPEG data URL.' });
  const data = image.split(',')[1];
  if (Buffer.byteLength(data, 'base64') > 1_000_000) return send(res, 413, { error: 'Artwork is too large.' });
  const ref = `artwork-${randomUUID()}`;
  try {
    const response = await fetch('https://api.typesafe.ai/v1/systemone', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${process.env.TYPESAFE_API_KEY}`, 'Content-Type': 'application/json', 'TypeSafe-Danger-Zone-Images': 'true' },
      body: JSON.stringify({ model: 'jev-1.14.0', state: { artwork: ref }, assets: { [ref]: { type: 'image', format: 'jpg', data_base64: data } }, questions }),
      signal: AbortSignal.timeout(30000)
    });
    const result = await response.json();
    if (!response.ok) return send(res, 502, { error: `Jev returned ${response.status}: ${JSON.stringify(result.detail ?? 'Unknown error')}` });
    const scores = Object.fromEntries(Object.keys(questions).map(key => [key, result.answers?.[key]?.noul]));
    if (Object.values(scores).some(value => typeof value !== 'number' || value < 0 || value > 1)) throw new Error('Jev returned incomplete scores.');
    return send(res, 200, { scores, model: result.model, usage: result.usage });
  } catch (error) { return send(res, 502, { error: error.message }); }
}

function send(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(data));
}

const server = createServer(async (req, res) => {
  if (req.url === '/api/critique' && req.method === 'POST') return critique(req, res);
  if (req.url === '/healthz') return send(res, 200, { ok: true });
  if (req.url === '/api/status' && req.method === 'GET') return send(res, 200, { jevConfigured: Boolean(process.env.TYPESAFE_API_KEY) });
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, { error: 'Method not allowed.' });
  const pathname = decodeURIComponent((req.url ?? '/').split('?')[0]);
  // Field Life is the homepage; the original ant farm lives at /farm.
  const routed = pathname === '/' ? '/field/index.html' : pathname === '/farm' || pathname === '/farm/' ? '/index.html' : pathname;
  const file = normalize(join(root, routed.endsWith('/') ? routed + 'index.html' : routed));
  if (relative(root, file).startsWith('..')) return send(res, 403, { error: 'Forbidden.' });
  try {
    const content = await readFile(file);
    res.writeHead(200, {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      // Always revalidate so a deploy shows up immediately through Cloudflare.
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin'
    });
    res.end(content);
  } catch { send(res, 404, { error: 'Not found.' }); }
});

server.listen(Number(process.env.PORT) || 3000, () => console.log(`Art Farm listening on http://localhost:${Number(process.env.PORT) || 3000}`));
