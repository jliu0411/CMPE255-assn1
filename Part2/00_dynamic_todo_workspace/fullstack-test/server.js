import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const root = dirname(fileURLToPath(import.meta.url));
const publicDir = join(root, 'public');
const dataDir = join(root, 'data');
const dataFile = join(dataDir, 'tasks.json');
const port = Number(process.env.PORT || 4173);

const seed = [
  { title: 'Shape the product roadmap', notes: 'Turn customer signals into the next set of bets.', priority: 'high', status: 'in-progress', tags: ['Strategy'], dueDate: new Date().toISOString().slice(0, 10) },
  { title: 'Review research insights', notes: 'Pull the strongest themes into a one-page brief.', priority: 'medium', status: 'todo', tags: ['Research'], dueDate: '' },
  { title: 'Share weekly progress', notes: 'Keep it concise: wins, risks, and next moves.', priority: 'low', status: 'done', tags: ['Team'], dueDate: '' }
];

export function normalizeTask(input = {}, existing = {}) {
  const title = String(input.title ?? existing.title ?? '').trim().slice(0, 160);
  if (!title) throw Object.assign(new Error('A task title is required.'), { status: 400 });
  const allowedStatus = ['todo', 'in-progress', 'done'];
  const allowedPriority = ['low', 'medium', 'high'];
  const now = new Date().toISOString();
  const status = allowedStatus.includes(input.status) ? input.status : (existing.status || 'todo');
  return {
    id: existing.id || randomUUID(), title,
    notes: String(input.notes ?? existing.notes ?? '').trim().slice(0, 2000),
    status, priority: allowedPriority.includes(input.priority) ? input.priority : (existing.priority || 'medium'),
    tags: [...new Set((Array.isArray(input.tags) ? input.tags : existing.tags || []).map(x => String(x).trim().slice(0, 24)).filter(Boolean))].slice(0, 6),
    dueDate: /^\d{4}-\d{2}-\d{2}$/.test(input.dueDate) ? input.dueDate : (input.dueDate === '' ? '' : existing.dueDate || ''),
    createdAt: existing.createdAt || now, updatedAt: now,
    completedAt: status === 'done' ? (existing.completedAt || now) : null
  };
}

async function ensureData() {
  await mkdir(dataDir, { recursive: true });
  if (!existsSync(dataFile)) await writeFile(dataFile, JSON.stringify(seed.map(x => normalizeTask(x)), null, 2));
}
async function getTasks() { await ensureData(); return JSON.parse(await readFile(dataFile, 'utf8')); }
async function saveTasks(tasks) { await writeFile(dataFile, JSON.stringify(tasks, null, 2)); }
function json(res, status, body) { res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(body)); }
async function body(req) {
  const chunks = []; let size = 0;
  for await (const chunk of req) { size += chunk.length; if (size > 1e6) throw Object.assign(new Error('Request is too large.'), { status: 413 }); chunks.push(chunk); }
  try { return JSON.parse(Buffer.concat(chunks).toString() || '{}'); } catch { throw Object.assign(new Error('Invalid JSON.'), { status: 400 }); }
}

export async function handleApi(req, res, url) {
  const match = url.pathname.match(/^\/api\/tasks(?:\/([^/]+))?$/); const id = match?.[1];
  if (!match) return false;
  let tasks = await getTasks();
  if (req.method === 'GET' && !id) { json(res, 200, tasks); return true; }
  if (req.method === 'POST' && !id) { const task = normalizeTask(await body(req)); tasks.unshift(task); await saveTasks(tasks); json(res, 201, task); return true; }
  const index = tasks.findIndex(t => t.id === id);
  if (index < 0) { json(res, 404, { error: 'Task not found.' }); return true; }
  if (req.method === 'PATCH') { tasks[index] = normalizeTask(await body(req), tasks[index]); await saveTasks(tasks); json(res, 200, tasks[index]); return true; }
  if (req.method === 'DELETE') { const [removed] = tasks.splice(index, 1); await saveTasks(tasks); json(res, 200, removed); return true; }
  json(res, 405, { error: 'Method not allowed.' }); return true;
}

const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml' };
export const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    if (url.pathname.startsWith('/api/')) { if (!await handleApi(req, res, url)) json(res, 404, { error: 'Not found.' }); return; }
    const requested = url.pathname === '/' ? 'index-fixed.html' : url.pathname.slice(1);
    if (requested.includes('..')) { res.writeHead(403); return res.end(); }
    try { const file = await readFile(join(publicDir, requested)); res.writeHead(200, { 'Content-Type': mime[extname(requested)] || 'application/octet-stream' }); res.end(file); }
    catch { const file = await readFile(join(publicDir, 'index-fixed.html')); res.writeHead(200, { 'Content-Type': mime['.html'] }); res.end(file); }
  } catch (error) { json(res, error.status || 500, { error: error.status ? error.message : 'Something went wrong.' }); }
});

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await ensureData();
  server.on('error', error => {
    console.error(error.code === 'EADDRINUSE' ? `Port ${port} is already in use. Stop the other process or set PORT to another value.` : error);
    process.exitCode = 1;
  });
  server.listen(port, () => console.log(`Momentum running at http://localhost:${port}`));
}
