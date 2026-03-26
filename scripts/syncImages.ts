import fs from 'fs';
import path from 'path';

const projectRoot = process.cwd();
const srcDir = path.resolve(projectRoot, 'vlab-chatbot/images');
const altSrc = path.resolve(projectRoot, 'images');
const src = fs.existsSync(srcDir) ? srcDir : altSrc;
const destRoots = [
  path.resolve(projectRoot, 'public/images'),
  path.resolve(projectRoot, 'vlab-chatbot/public/images'),
];

for (const dst of destRoots) {
  fs.mkdirSync(dst, { recursive: true });
}

if (!fs.existsSync(src) || !fs.statSync(src).isDirectory()) {
  console.log('No images directory found at', src);
  process.exit(0);
}

const copyToAll = (from: string, rel: string) => {
  for (const root of destRoots) {
    const to = path.join(root, rel);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
  }
};

const walk = (dir: string, base = '') => {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const rel = path.join(base, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, rel);
    else copyToAll(full, rel);
  }
};

walk(src);
console.log('Synced images to:', destRoots.join(', '));

