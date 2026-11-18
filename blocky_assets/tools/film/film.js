// "The Weakest Pet" - blocky 3D Short director. Renders deterministic frames for a timeline.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const W = 720, H = 1280, FPS = 30;
const PI = Math.PI, sin = Math.sin, cos = Math.cos;
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const lerp = (a, b, k) => a + (b - a) * k;
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const vlerp = (a, b, k) => a.clone().lerp(b, k);

// ------------------------------------------------------------------ renderer
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1); renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);
const out = document.createElement('canvas'); out.width = W; out.height = H;
const g2 = out.getContext('2d');
const camera = new THREE.PerspectiveCamera(40, W / H, 0.05, 400);

// ------------------------------------------------------------------ assets
const loader = new GLTFLoader(), tl = new THREE.TextureLoader();
const A = {}, FACE = {}, TEX = {};
let catalog;
async function loadAll() {
  catalog = await (await fetch('/assets/catalog.json')).json();
  const files = {};
  for (const c of catalog.characters) files[c.id] = c.file;
  for (const p of catalog.props) files[p.id] = p.file;
  const need = ['hoodie_boy', 'rival_blond', 'brunette_girl', 'old_man', 'townsfolk_1', 'townsfolk_2', 'townsfolk_3',
    'townsfolk_4', 'townsfolk_5', 'townsfolk_6', 'kid_green', 'farmer', 'academy_student_1', 'academy_student_2',
    'sweater_boy', 'blonde_girl', 'pet_snail', 'pet_dragon', 'pet_wolf', 'pet_owl', 'pet_cat', 'pet_bunny',
    'egg_common', 'egg_rare', 'egg_epic', 'egg_legendary', 'founder_statue', 'stone_guardian', 'fountain',
    'shop_lamps', 'shop_mushrooms', 'shop_cafe', 'shop_blank', 'house_white', 'house_pink', 'house_green',
    'house_navy_night', 'tree_round', 'bush', 'street_lamp', 'string_lights', 'bench', 'car', 'bed', 'window',
    'calendar', 'side_table_lamp', 'tape_measure', 'cloud', 'picket_fence', 'hardwood_floor'];
  await Promise.all(need.map(async id => { A[id] = (await loader.loadAsync('/assets/' + files[id])).scene; }));
  const faces = ['smile', 'happy', 'laugh', 'shocked', 'scared', 'worried', 'smug', 'unimpressed', 'determined',
    'troll_grin', 'sad', 'anime_shocked', 'sleepy_blush', 'neutral', 'screaming', 'love'];
  await Promise.all(faces.flatMap(f => [['', f], ['_lashes', f]]).map(async ([s, f]) => {
    const t = await tl.loadAsync(`/assets/textures/faces/face_${f}${s}.png`);
    t.flipY = false; t.colorSpace = THREE.SRGBColorSpace; FACE[f + s] = t;
  }));
  for (const k of ['sky_day', 'sky_sunset', 'sky_night_stars', 'sky_dusk_purple']) {
    const t = await tl.loadAsync(`/assets/textures/sky_fx/${k}.png`); t.colorSpace = THREE.SRGBColorSpace; TEX[k] = t;
  }
  for (const k of ['calendar_day_01', 'calendar_day_30']) {
    const t = await tl.loadAsync(`/assets/textures/signs/${k}.png`); t.colorSpace = THREE.SRGBColorSpace; t.flipY = false; TEX[k] = t;
  }
  const g = await tl.loadAsync('/assets/textures/surfaces/grass.png');
  g.colorSpace = THREE.SRGBColorSpace; g.wrapS = g.wrapT = THREE.RepeatWrapping; g.repeat.set(40, 40); TEX.grass = g;
  const c = await tl.loadAsync('/assets/textures/surfaces/cobble.png');
  c.colorSpace = THREE.SRGBColorSpace; c.wrapS = c.wrapT = THREE.RepeatWrapping; c.repeat.set(8, 8); TEX.cobble = c;
}

const LASH = new Set(['brunette_girl', 'blonde_girl']);
function inst(id, pos = V(0, 0, 0), rotY = 0, scale = 1) {
  const o = A[id].clone(true);
  o.traverse(m => { if (m.isMesh) { m.castShadow = true; m.receiveShadow = true; } });
  o.position.copy(pos); o.rotation.y = rotY; if (scale !== 1) o.scale.multiplyScalar(scale);
  o.userData.id = id;
  return o;
}
function setFace(o, expr) {
  const f = o.getObjectByName('face'); if (!f) return;
  const key = expr + (LASH.has(o.userData.id) ? '_lashes' : '');
  f.traverse(m => { if (m.isMesh) { if (!m.userData.own) { m.material = m.material.clone(); m.userData.own = 1; } m.material.map = FACE[key]; m.material.emissiveMap = null; } });
}
function glowMats(o, prefix) {
  const mats = [];
  o.traverse(m => {
    if (m.isMesh && m.material.name && m.material.name.startsWith(prefix)) {
      if (!m.userData.own) { m.material = m.material.clone(); m.userData.own = 1; m.userData.base = m.material.emissiveIntensity; }
      mats.push(m);
    }
  });
  return k => mats.forEach(m => { m.material.emissiveIntensity = m.userData.base * k; });
}

// ------------------------------------------------------------------ rigs + actions
function rig(o) {
  const g = n => o.getObjectByName(n);
  const r = { o, hips: g('hips'), armL: g('arm_L'), armR: g('arm_R'), legL: g('leg_L'), legR: g('leg_R'), head: g('head') };
  r.hy = r.hips ? r.hips.position.y : 0;
  return r;
}
function act(r, a, t, k = 1) {
  const z = n => n && n.rotation.set(0, 0, 0);
  [r.armL, r.armR, r.legL, r.legR, r.head, r.hips].forEach(z);
  r.hips.position.y = r.hy; r.hips.position.x = 0;
  const w = (f) => sin(t * f);
  switch (a) {
    case 'idle':
      r.armL.rotation.z = 0.06 + 0.03 * w(2); r.armR.rotation.z = -0.06 - 0.03 * w(2);
      r.head.rotation.y = 0.12 * w(0.8); r.hips.position.y += 0.01 * w(3); break;
    case 'walk': case 'run': {
      const f = a === 'run' ? 13 : 7, am = a === 'run' ? 1.0 : 0.55;
      r.legL.rotation.x = am * w(f); r.legR.rotation.x = -am * w(f);
      r.armL.rotation.x = -am * w(f); r.armR.rotation.x = am * w(f);
      r.hips.position.y += Math.abs(w(f)) * 0.04; if (a === 'run') r.hips.rotation.x = 0.18; break;
    }
    case 'cheer':
      r.armL.rotation.x = r.armR.rotation.x = -2.8 + 0.25 * w(10);
      r.hips.position.y += Math.abs(w(10)) * 0.06; break;
    case 'point':
      r.armR.rotation.x = -1.55; r.head.rotation.y = 0.0; r.armL.rotation.z = 0.08; break;
    case 'point_laugh':
      r.armR.rotation.x = -1.5 + 0.08 * w(14); r.armL.rotation.x = -0.6;
      r.hips.position.y += Math.abs(w(14)) * 0.03; r.head.rotation.x = -0.15; break;
    case 'laugh':
      r.armL.rotation.x = -0.7; r.armR.rotation.x = -0.7; r.armL.rotation.z = -0.25; r.armR.rotation.z = 0.25;
      r.hips.position.y += Math.abs(w(14)) * 0.05; r.head.rotation.x = -0.25 + 0.08 * w(14); break;
    case 'scared':
      r.armL.rotation.x = r.armR.rotation.x = -2.1; r.armL.rotation.z = -0.35; r.armR.rotation.z = 0.35;
      r.hips.position.x = 0.012 * w(45) * k; break;
    case 'freeze':
      r.armL.rotation.x = r.armR.rotation.x = -0.9; r.armL.rotation.z = 0.5; r.armR.rotation.z = -0.5; break;
    case 'shocked':
      r.armL.rotation.z = 0.9; r.armR.rotation.z = -0.9; r.head.rotation.x = -0.1; break;
    case 'hold_out':
      r.armL.rotation.x = r.armR.rotation.x = -1.2; r.armL.rotation.z = -0.15; r.armR.rotation.z = 0.15; break;
    case 'look_down':
      r.head.rotation.x = 0.5; r.armL.rotation.z = 0.05; r.armR.rotation.z = -0.05; break;
    case 'look_around':
      r.head.rotation.y = 0.5 * w(2.2); r.armL.rotation.z = 0.15; r.armR.rotation.z = -0.15; break;
    case 'proud':
      r.armL.rotation.z = 0.5; r.armR.rotation.z = -0.5; r.armL.rotation.x = -0.3; r.armR.rotation.x = -0.3;
      r.head.rotation.x = -0.12; break;
  }
}
function petAnim(o, t, mode = 'idle') {
  const g = n => o.getObjectByName(n);
  const id = o.userData.id;
  if (id === 'pet_dragon') {
    const f = mode === 'fly' ? 16 : 6;
    g('wing_L').rotation.z = 0.7 * sin(t * f); g('wing_R').rotation.z = -0.7 * sin(t * f);
    g('tail').rotation.y = 0.3 * sin(t * 3); g('head').rotation.x = 0.1 * sin(t * 2);
  } else if (id === 'pet_wolf') {
    const f = mode === 'run' ? 16 : 0, a = mode === 'run' ? 0.9 : 0;
    ['FL', 'BR'].forEach(n => g('leg_' + n).rotation.x = a * sin(t * f));
    ['FR', 'BL'].forEach(n => g('leg_' + n).rotation.x = -a * sin(t * f));
    g('tail').rotation.y = 0.4 * sin(t * 6); g('head').rotation.x = mode === 'run' ? 0.15 : 0.05 * sin(t * 2);
  } else if (id === 'pet_snail') {
    g('body').scale.z = 1 + 0.1 * sin(t * 3);
    g('eyestalk_L').rotation.x = 0.25 * sin(t * 2.5); g('eyestalk_R').rotation.x = 0.25 * sin(t * 2.5 + 1);
  } else if (id === 'pet_owl') {
    g('head').rotation.y = 0.6 * sin(t * 1.5);
  } else if (id === 'pet_cat' || id === 'pet_bunny') {
    const h = g('head'); if (h) h.rotation.y = 0.3 * sin(t * 1.7);
  }
}

// ------------------------------------------------------------------ world
function mkScene(tod) {
  const s = new THREE.Scene();
  const L = {
    day: { sky: 'sky_day', hemi: [0xe8f4ff, 0x6a7a55, 1.3], sun: [0xffffff, 2.6], sp: [10, 18, 12] },
    dusk: { sky: 'sky_sunset', hemi: [0xffd2b0, 0x403050, 1.0], sun: [0xffa860, 2.2], sp: [-14, 6, 8] },
    night: { sky: 'sky_night_stars', hemi: [0x5a78c0, 0x101020, 0.55], sun: [0x9ab4ff, 0.7], sp: [-8, 14, 6] },
    morning: { sky: 'sky_day', hemi: [0xfff0d8, 0x707a55, 1.25], sun: [0xffe2b0, 2.4], sp: [14, 10, 10] },
    room: { sky: 'sky_day', hemi: [0xfff6e8, 0x806050, 1.1], sun: [0xfff0d0, 1.6], sp: [4, 6, 6] },
  }[tod];
  s.background = TEX[L.sky];
  s.add(new THREE.HemisphereLight(...L.hemi));
  const d = new THREE.DirectionalLight(...L.sun); d.position.set(...L.sp); d.castShadow = true;
  d.shadow.mapSize.set(1024, 1024); Object.assign(d.shadow.camera, { left: -22, right: 22, top: 22, bottom: -22, near: 1, far: 80 });
  d.shadow.bias = -0.0005; s.add(d);
  if (tod === 'night') s.fog = new THREE.Fog(0x0c1434, 18, 70);
  if (tod === 'dusk') s.fog = new THREE.Fog(0xe0a080, 30, 90);
  if (tod === 'day' || tod === 'morning') s.fog = new THREE.Fog(0xcfe6fa, 40, 120);
  s.userData.tod = tod;
  return s;
}
function town(s, { stage = false } = {}) {
  const tod = s.userData.tod;
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), new THREE.MeshStandardMaterial({ map: TEX.grass, roughness: 1 }));
  grass.rotation.x = -PI / 2; grass.position.y = -0.02; grass.receiveShadow = true; s.add(grass);
  const plaza = new THREE.Mesh(new THREE.PlaneGeometry(34, 30), new THREE.MeshStandardMaterial({ map: TEX.cobble, roughness: 0.95 }));
  plaza.rotation.x = -PI / 2; plaza.position.set(0, 0, 0); plaza.receiveShadow = true; s.add(plaza);
  const add = (id, x, z, ry = 0, sc = 1) => { const o = inst(id, V(x, 0, z), ry, sc); s.add(o); return o; };
  add('fountain', -9, 6);
  add('shop_lamps', -10, -12); add('shop_cafe', -4.5, -12.5); add('shop_mushrooms', 4.5, -12.5); add('shop_blank', 10, -12);
  const night = tod === 'night';
  add(night ? 'house_navy_night' : 'house_white', -19, 2, PI / 2); add('house_pink', -19, -8, PI / 2);
  add(night ? 'house_navy_night' : 'house_green', 19, -6, -PI / 2); add('house_white', 19, 6, -PI / 2);
  for (const [x, z] of [[-14, 10], [14, 12], [-15, -3], [15, 1], [-6, -18], [7, -19], [-22, 14], [22, -14]]) add('tree_round', x, z, x);
  for (const [x, z] of [[-12, 13], [12, 14], [-3, -16]]) add('bush', x, z);
  const lamps = [[-6.5, 3.5], [6.5, 3.5], [-6.5, -6], [6.5, -6]];
  for (const [x, z] of lamps) add('street_lamp', x, z);
  const sl = add('string_lights', 0, -6); sl.position.y = 3.9; sl.scale.set(2.15, 1, 1);
  add('bench', -12, 2, PI / 2); add('bench', 12, -2, -PI / 2);
  for (const [x, y, z, sc] of [[-25, 22, -60, 4], [20, 26, -70, 5], [0, 30, -90, 6]]) { const c = inst('cloud', V(x, y, z), 0, sc); s.add(c); }
  if (night) for (const [x, z] of lamps) {
    const p = new THREE.PointLight(0xffc070, 40, 18, 2); p.position.set(x, 3.6, z); s.add(p);
  }
  const statue = add('founder_statue', 9, 5, -0.75);
  if (stage) buildStage(s);
  return { statue };
}
function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const EGG_X = { egg_legendary: -2.4, egg_epic: -0.8, egg_rare: 0.8, egg_common: 2.4 };
function buildStage(s) {
  const st = new THREE.Group(); s.add(st);
  const plat = new THREE.Mesh(new THREE.BoxGeometry(8, 0.8, 4), new THREE.MeshStandardMaterial({ color: 0x3b2a6a, roughness: 0.6 }));
  plat.position.y = 0.4; plat.castShadow = plat.receiveShadow = true; st.add(plat);
  const trim = new THREE.Mesh(new THREE.BoxGeometry(8.05, 0.08, 4.05), new THREE.MeshStandardMaterial({ color: 0xffd23a, emissive: 0xffb020, emissiveIntensity: 0.6 }));
  trim.position.y = 0.8; st.add(trim);
  const tex = canvasTex(1024, 512, (c, w, h) => {
    const g = c.createLinearGradient(0, 0, 0, h); g.addColorStop(0, '#4a2a9a'); g.addColorStop(1, '#2a1660'); c.fillStyle = g; c.fillRect(0, 0, w, h);
    c.strokeStyle = '#ffd23a'; c.lineWidth = 16; c.strokeRect(14, 14, w - 28, h - 28);
    c.fillStyle = '#ffffff'; c.font = '900 120px Inter'; c.textAlign = 'center'; c.fillText('HATCHING DAY', w / 2, 230);
    c.fillStyle = '#ffd23a'; c.font = '800 60px Inter'; c.fillText('★ AGE 12 ★', w / 2, 340);
  });
  const bd = new THREE.Mesh(new THREE.PlaneGeometry(8, 4), new THREE.MeshStandardMaterial({ map: tex, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0.35 }));
  bd.position.set(0, 2.8, -1.95); st.add(bd);
  for (const x of [-4, 4]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.4, 4.8, 0.4), new THREE.MeshStandardMaterial({ color: 0xd6b04a, metalness: 0.6, roughness: 0.3 }));
    p.position.set(x, 2.4, -1.9); p.castShadow = true; st.add(p);
  }
  for (const id of Object.keys(EGG_X)) {
    const x = EGG_X[id];
    const ped = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.38, 0.6, 16), new THREE.MeshStandardMaterial({ color: 0xf2f0ea, roughness: 0.5 }));
    ped.position.set(x, 1.1, 0.4); ped.castShadow = ped.receiveShadow = true; st.add(ped);
  }
}
function eggs(s, skip = []) {
  const E = {};
  for (const id of Object.keys(EGG_X)) {
    if (skip.includes(id)) continue;
    E[id] = inst(id, V(EGG_X[id], 1.4, 0.4)); s.add(E[id]);
  }
  return E;
}
function eggAnim(e, t, shake = 0, crack = 0) {
  e.rotation.z = shake * 0.18 * sin(t * 28);
  const top = e.getObjectByName('top');
  top.position.y = 0.23 + crack * 0.35; top.rotation.z = crack * 1.1; top.position.x = crack * 0.18;
}
function crowd(s, center, n = 12, faceTo = PI, faces = ['smile', 'happy', 'neutral']) {
  const ids = ['townsfolk_1', 'townsfolk_2', 'townsfolk_3', 'townsfolk_4', 'townsfolk_5', 'townsfolk_6', 'kid_green', 'farmer',
    'academy_student_1', 'academy_student_2', 'sweater_boy', 'blonde_girl'];
  const people = [];
  for (let i = 0; i < n; i++) {
    const row = Math.floor(i / 6), col = i % 6;
    const p = inst(ids[i % ids.length], V(center.x + (col - 2.5) * 1.15 + (row % 2) * 0.5, 0, center.z + row * 1.3), faceTo + (col - 2.5) * 0.06);
    setFace(p, faces[i % faces.length]); s.add(p); people.push(rig(p));
  }
  return people;
}
// a person holding/near a pet
function person(s, id, pos, ry, face) { const o = inst(id, pos, ry); if (face) setFace(o, face); s.add(o); return rig(o); }

// ------------------------------------------------------------------ 2D overlays
function rarityTag(t, label, color, sub) {
  const k = ease(t / 0.25);
  if (k <= 0) return;
  g2.save(); g2.translate(W / 2, 210); g2.scale(0.6 + 0.4 * k, 0.6 + 0.4 * k); g2.globalAlpha = k;
  g2.font = '900 76px Inter'; g2.textAlign = 'center'; g2.textBaseline = 'middle';
  const tw = g2.measureText(label).width + 80;
  g2.fillStyle = 'rgba(10,10,20,0.75)'; roundRect(-tw / 2, -60, tw, sub ? 150 : 120, 28); g2.fill();
  g2.strokeStyle = color; g2.lineWidth = 8; g2.stroke();
  g2.fillStyle = color; g2.fillText(label, 0, 0);
  if (sub) { g2.font = '800 40px Inter'; g2.fillStyle = '#ffffff'; g2.fillText(sub, 0, 62); }
  g2.restore();
}
function roundRect(x, y, w, h, r) { g2.beginPath(); g2.roundRect(x, y, w, h, r); }
function tierLadder(t) {
  const tiers = [['LEGENDARY', '#ffc928', 1.0], ['EPIC', '#b47bff', 0.72], ['RARE', '#4fa8ff', 0.45], ['COMMON', '#c9c9c9', 0.15]];
  g2.save(); g2.translate(W / 2, 170);
  tiers.forEach(([name, col, p], i) => {
    const k = ease((t - i * 0.22) / 0.3); if (k <= 0) return;
    const y = i * 92;
    g2.globalAlpha = k;
    g2.fillStyle = 'rgba(10,10,25,0.78)'; roundRect(-300, y, 600, 76, 20); g2.fill();
    g2.font = '900 40px Inter'; g2.textAlign = 'left'; g2.textBaseline = 'middle'; g2.fillStyle = col; g2.fillText(name, -280, y + 38);
    g2.fillStyle = 'rgba(255,255,255,0.15)'; roundRect(20, y + 24, 260, 28, 14); g2.fill();
    g2.fillStyle = col; roundRect(20, y + 24, 260 * p * k, 28, 14); g2.fill();
  });
  g2.restore();
}
function captions(line, t) {
  const words = line.text.split(' ');
  const chunks = []; let cur = [];
  for (const w of words) { cur.push(w); if (cur.length === 3 || /[.,!?]$/.test(w.replace(/\*/g, ''))) { chunks.push(cur); cur = []; } }
  if (cur.length) chunks.push(cur);
  const total = chunks.reduce((a, c) => a + c.join(' ').length, 0);
  const vt = t - (line.vo_start - line.start);
  let acc = 0, idx = 0;
  for (let i = 0; i < chunks.length; i++) {
    const d = chunks[i].join(' ').length / total * line.vo_dur;
    if (vt >= acc) idx = i;
    acc += d;
  }
  if (vt < 0) idx = 0;
  let cs = 0; for (let i = 0; i < idx; i++) cs += chunks[i].join(' ').length / total * line.vo_dur;
  const pop = 1 + 0.18 * (1 - ease((vt - cs) / 0.12));
  const ch = chunks[idx];
  g2.save(); g2.translate(W / 2, H * 0.64); g2.scale(pop, pop);
  g2.font = '900 64px Inter'; g2.textBaseline = 'middle'; g2.lineJoin = 'round';
  const parts = ch.map(w => ({ w: w.replace(/\*/g, ''), key: w.includes('*') }));
  const space = g2.measureText(' ').width;
  const widths = parts.map(p => g2.measureText(p.w).width);
  let x = -(widths.reduce((a, b) => a + b, 0) + space * (parts.length - 1)) / 2;
  parts.forEach((p, i) => {
    g2.lineWidth = 12; g2.strokeStyle = '#000'; g2.strokeText(p.w, x, 0);
    g2.fillStyle = p.key ? '#3CE63C' : '#ffffff'; g2.fillText(p.w, x, 0);
    x += widths[i] + space;
  });
  g2.restore();
}

// ------------------------------------------------------------------ camera helpers
function look(pos, tgt, shake = 0, t = 0) {
  camera.position.copy(pos);
  if (shake) camera.position.add(V(sin(t * 53) * shake, sin(t * 61 + 1) * shake, 0));
  camera.lookAt(tgt);
}
function worldPos(o, name) { const n = o.getObjectByName(name); return n.getWorldPosition(new THREE.Vector3()); }

// ------------------------------------------------------------------ shots
const SHOTS = {
  s01(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    const elder = person(s, 'old_man', V(0, 0.8, -1.0), 0, 'smile');
    const ppl = crowd(s, V(0, 0, 4.5), 12, PI, ['happy', 'smile', 'laugh']);
    return { s, u(t) {
      for (const k in E) { eggAnim(E[k], t, 0, 0); E[k].position.y = 1.4 + 0.04 * sin(t * 3 + EGG_X[k]); E[k].rotation.y = t * 0.6; }
      act(elder, 'proud', t); ppl.forEach((p, i) => act(p, i % 3 ? 'cheer' : 'idle', t + i));
      const k = ease(t / D); look(vlerp(V(0, 6, 24), V(0, 3.2, 13), k), V(0, 1.5, 0));
    } };
  },
  s02(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    return { s, u(t) {
      for (const k in E) { E[k].rotation.y = t * 0.8; E[k].position.y = 1.4 + 0.03 * sin(t * 4 + EGG_X[k]); }
      const k = ease(t / D); look(V(lerp(-1.2, 1.2, k), 1.9, 4.6), V(lerp(-0.6, 0.6, k), 1.55, 0.4));
    }, o(t) { tierLadder(t); } };
  },
  s03(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    const jake = person(s, 'rival_blond', V(-2.4, 0.8, -0.5), 0, 'smug');
    const dr = inst('pet_dragon', V(-2.4, 1.4, 0.4), 0.3); dr.scale.setScalar(0.01); s.add(dr);
    const glow = new THREE.PointLight(0xffc040, 0, 6, 2); glow.position.set(-2.4, 1.8, 0.8); s.add(glow);
    return { s, u(t) {
      const c = ease((t - 0.5) / 0.35);
      eggAnim(E.egg_legendary, t, t < 0.5 ? 1 : 0, c);
      glow.intensity = 4 * c * (1 + 0.3 * sin(t * 20));
      const g = ease((t - 0.6) / 0.5);
      dr.scale.setScalar(Math.max(0.01, g * 1.1)); dr.position.y = 1.45 + g * 0.6 + 0.08 * sin(t * 6);
      petAnim(dr, t, 'fly');
      if (t > 0.9) { setFace(jake.o, 'happy'); act(jake, 'cheer', t); } else act(jake, 'idle', t);
      const k = ease(t / D); look(vlerp(V(-1.2, 1.5, 5.0), V(-1.7, 1.45, 4.2), k), V(-2.4, 2.0, 0));
    }, o(t) { if (t > 1.0) rarityTag(t - 1.0, 'LEGENDARY', '#ffc928', 'POWER: FIRE'); } };
  },
  s04(D) {
    const s = mkScene('day'); town(s);
    const mia = person(s, 'brunette_girl', V(-8, 0, 5), PI / 2, 'happy');
    const wolf = inst('pet_wolf', V(-7, 0, 5.8), PI / 2, 1.1); s.add(wolf);
    const car = inst('car', V(-6, 0, 9), PI / 2); s.add(car);
    return { s, u(t) {
      const x = -8 + t * 7.5;
      mia.o.position.x = x; wolf.position.x = x + 1.4; car.position.x = -6 + t * 3.5;
      act(mia, 'run', t); petAnim(wolf, t, 'run');
      look(V(x + 2.2, 1.6, 12), V(x + 0.6, 1.0, 5.5));
    }, o(t) { if (t > 0.25) rarityTag(t - 0.25, 'EPIC', '#b47bff', 'POWER: SPEED'); } };
  },
  s05(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    const me = person(s, 'hoodie_boy', V(2.4, 0.8, -0.45), 0, 'worried');
    return { s, u(t) {
      eggAnim(E.egg_common, t, 0.4 + t * 0.8, ease((t - D + 0.25) / 0.25) * 0.4);
      act(me, 'look_down', t);
      const k = ease(t / D); look(vlerp(V(3.9, 2.15, 2.6), V(3.5, 2.1, 2.1), k), V(2.4, 1.85, 0));
    } };
  },
  s06(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    eggAnim(E.egg_common, 0, 0, 1);
    const sn = inst('pet_snail', V(2.32, 1.4, 0.55), -0.2, 1.0); s.add(sn);
    E.egg_common.getObjectByName('bottom').position.x = 0.2; E.egg_common.getObjectByName('bottom').position.z = -0.12;
    return { s, u(t) {
      petAnim(sn, t);
      const k = ease(t / D); look(vlerp(V(2.45, 1.75, 1.6), V(2.4, 1.6, 1.15), k), V(2.33, 1.47, 0.5));
    } };
  },
  s07(D) {
    const s = mkScene('day'); town(s, { stage: true }); const E = eggs(s);
    eggAnim(E.egg_common, 0, 0, 1);
    const sn = inst('pet_snail', V(2.32, 1.4, 0.55), -0.2); s.add(sn);
    const me = person(s, 'hoodie_boy', V(2.4, 0.8, -0.45), 0, 'shocked');
    return { s, u(t) {
      petAnim(sn, t); act(me, 'shocked', t);
      E.egg_common.getObjectByName('top').position.set(0.45, 0.02, 0.25); E.egg_common.getObjectByName('top').rotation.z = 2.4;
      const k = ease(t / D); look(vlerp(V(2.3, 1.8, 2.8), V(2.35, 1.8, 2.3), k), V(2.4, 1.72, 0.2));
    }, o(t) { rarityTag(t, 'COMMON', '#c9c9c9', 'POWER: 0'); } };
  },
  s08(D) {
    const s = mkScene('day'); town(s, { stage: true }); eggs(s, ['egg_common', 'egg_legendary', 'egg_epic']);
    const ppl = crowd(s, V(0, 0, 4.2), 12, PI, ['laugh', 'troll_grin', 'laugh', 'happy']);
    const jake = person(s, 'rival_blond', V(-1.2, 0, 3.0), PI + 0.3, 'laugh');
    const dr = inst('pet_dragon', V(-2.2, 1.8, 3.2), PI); s.add(dr);
    return { s, u(t) {
      ppl.forEach((p, i) => act(p, i % 2 ? 'laugh' : 'point_laugh', t + i * 0.3));
      act(jake, 'point_laugh', t); petAnim(dr, t, 'fly');
      const k = ease(t / D); look(vlerp(V(1.8, 3.4, 0.6), V(1.6, 3.0, 0.9), k), V(-0.2, 1.3, 5.2));
    } };
  },
  s09(D) { return room(D, 'calendar_day_01', 'point', 'determined'); },
  s10(D) { return room(D, 'calendar_day_30', 'unimpressed', 'unimpressed'); },
  s11(D) {
    const s = mkScene('dusk'); const T = town(s);
    const w1 = person(s, 'townsfolk_2', V(3, 0, 9), PI / 2, 'smile'), w2 = person(s, 'farmer', V(14, 0, 8.5), -PI / 2, 'neutral');
    return { s, u(t) {
      w1.o.position.x = 3 + t * 1.3; w2.o.position.x = 14 - t * 1.3; act(w1, 'walk', t); act(w2, 'walk', t + 1);
      const k = ease(t / D); look(vlerp(V(4.5, 1.2, 12), V(5.0, 4.0, 11.5), k), V(9, 2.6, 5));
    } };
  },
  s12(D) {
    const s = mkScene('dusk'); const T = town(s);
    return { s, u(t) {
      const st = T.statue; const fwd = V(sin(st.rotation.y), 0, cos(st.rotation.y));
      const plaque = st.localToWorld(V(0, 0.65, 0.81));
      const k = ease(t / D);
      look(plaque.clone().add(fwd.clone().multiplyScalar(lerp(3.6, 2.6, k))).add(V(0, 0.5, 0)), plaque.clone().add(V(0, 0.5, 0)));
    }, o(t) { warningCard(t); } };
  },
  s13(D) {
    const s = mkScene('night'); town(s);
    const me = person(s, 'hoodie_boy', V(0.5, 0, 7), PI, 'scared');
    const jake = person(s, 'rival_blond', V(-1.8, 0, 6.5), PI - 0.3, 'shocked');
    const mia = person(s, 'brunette_girl', V(2.6, 0, 6.4), PI + 0.3, 'scared');
    const dr = inst('pet_dragon', V(-2.6, 1.6, 6.8), PI); s.add(dr);
    const wolf = inst('pet_wolf', V(3.5, 0, 6.6), PI + 0.4); s.add(wolf);
    return { s, u(t) {
      [me, jake, mia].forEach((p, i) => act(p, 'look_around', t + i)); petAnim(dr, t, 'fly'); petAnim(wolf, t);
      look(V(0.5, 1.8, 11.5), V(0.3, 1.5, 6), 0.06 + 0.04 * sin(t * 3), t);
    } };
  },
  s14(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, -10, -4)); s.add(G); const gl = glowMats(G, 'stone_glow');
    const gr = rig(G);
    return { s, u(t) {
      const k = ease(t / (D * 0.85));
      G.position.y = lerp(-10, 0, k); gl(0.2 + 1.3 * ease((t - 0.5) / 0.6)); act(gr, 'idle', t);
      look(V(0.4, 0.6, 9), V(0, lerp(2, 6, k), -4), 0.08 * (1 - k * 0.5), t);
    } };
  },
  s15(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); glowMats(G, 'stone_glow')(1.4); const gr = rig(G);
    const dr = inst('pet_dragon', V(-6, 3, 6), PI * 0.8, 1.3); s.add(dr);
    const start = V(-6, 3, 6);
    return { s, u(t) {
      act(gr, 'idle', t);
      const swing = ease((t - 0.7) / 0.35);
      gr.armR.rotation.x = -1.3 * swing; gr.armR.rotation.z = 0.25 * swing;
      G.updateMatrixWorld(true);
      const fist = gr.armR.localToWorld(V(0, -0.35, 0));
      if (t < 1.0) dr.position.copy(vlerp(start, V(-0.8, 4.0, -1.2), ease(t / 1.0)));
      else dr.position.copy(fist).add(V(0, 0.15, 0.25));
      dr.lookAt(fist); petAnim(dr, t, 'fly');
      look(V(9, 4.0, 7), V(-0.8, 3.6, -1.5));
    }, o(t) { if (t > 1.05) flash(t - 1.05); } };
  },
  s16(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); glowMats(G, 'stone_glow')(1.4); const gr = rig(G);
    const wolf = inst('pet_wolf', V(-9, 0, 4), PI * 0.75, 1.2); s.add(wolf);
    return { s, u(t) {
      act(gr, 'idle', t);
      const sw = ease((t - 0.55) / 0.3); gr.armL.rotation.x = -0.9 * sw; gr.armL.rotation.z = -0.2 * sw;
      G.updateMatrixWorld(true);
      const fist = gr.armL.localToWorld(V(0, -0.35, 0));
      if (t < 0.8) { wolf.position.copy(vlerp(V(-9, 0, 4), V(0.8, 0, 0.5), t / 0.8)); petAnim(wolf, t, 'run'); }
      else { wolf.position.copy(fist).add(V(0, -0.5, 0.3)); petAnim(wolf, t, 'run'); }
      look(V(-7, 2.2, 10), V(0, 2.2, -1));
    }, o(t) { if (t > 0.8) flash(t - 0.8); } };
  },
  s17(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); glowMats(G, 'stone_glow')(1.4);
    const me = person(s, 'hoodie_boy', V(0.4, 0, 7), PI, 'scared');
    const jake = person(s, 'rival_blond', V(-1.6, 0, 6.6), PI, 'scared');
    const mia = person(s, 'brunette_girl', V(2.3, 0, 6.6), PI, 'anime_shocked');
    const ppl = crowd(s, V(0, 0, 8.6), 6, PI, ['scared', 'shocked', 'worried']);
    return { s, u(t) {
      [me, jake, mia, ...ppl].forEach(p => act(p, 'freeze', t));
      const k = ease(t / D); look(vlerp(V(0.6, 1.0, 1.0), V(0.6, 1.05, 2.2), k), V(0.4, 1.4, 8));
    } };
  },
  s18(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); glowMats(G, 'stone_glow')(1.4);
    const sn = inst('pet_snail', V(0, 0, 6.6), PI, 1.0); s.add(sn);
    const me = person(s, 'hoodie_boy', V(0.5, 0, 7.4), PI, 'scared');
    return { s, u(t) {
      sn.position.z = 6.6 - t * 0.06; petAnim(sn, t); act(me, 'freeze', t);
      look(V(0.55, 0.16, 5.6), V(0, 0.12, 6.6 - t * 0.06));
    } };
  },
  s19(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); glowMats(G, 'stone_glow')(1.4); const gr = rig(G);
    const sn = inst('pet_snail', V(0, 0, 4.0), PI, 1.6); s.add(sn);
    const ppl = [person(s, 'hoodie_boy', V(1.5, 0, 7.4), PI, 'scared'), person(s, 'rival_blond', V(-1.5, 0, 7.4), PI, 'scared')];
    return { s, u(t) {
      act(gr, 'idle', t); gr.head.rotation.y = 0.6 * sin(t * 2.0); gr.head.rotation.x = 0.35;
      petAnim(sn, t); ppl.forEach(p => act(p, 'freeze', t));
      look(V(2.6, 8.2, -6.2), V(0, 0, 4.6));
    } };
  },
  s20(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); const gl = glowMats(G, 'stone_glow'); const gr = rig(G);
    const sn = inst('pet_snail', V(0, 0, 0), 0, 1.6); s.add(sn);
    const qx = new THREE.Quaternion().setFromAxisAngle(V(1, 0, 0), PI / 2), qz = new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), PI);
    sn.quaternion.copy(qz.multiply(qx));
    return { s, u(t) {
      act(gr, 'idle', t * 0.3); gl(0.7 + 0.2 * sin(t * 6));
      G.updateMatrixWorld(true);
      const chest = gr.hips.localToWorld(V(0, 0.36, 0.2));
      const y = lerp(chest.y - 1.8, chest.y - 0.55, ease(t / D));
      sn.position.set(0, y, chest.z + 0.02); petAnim(sn, t);
      look(V(3.2, y + 0.9, chest.z + 3.6), V(0, y + 0.45, chest.z));
    } };
  },
  s21(D) {
    const s = mkScene('night'); town(s);
    const G = inst('stone_guardian', V(0, 0, -4)); s.add(G); const gl = glowMats(G, 'stone_glow'); const gr = rig(G);
    const dr = inst('pet_dragon', V(0, 0, 0), PI * 0.8, 1.3); s.add(dr);
    const wolf = inst('pet_wolf', V(0, 0, 0), PI, 1.2); s.add(wolf);
    return { s, u(t) {
      const k = ease(t / (D * 0.8));
      gl(1.4 * (1 - k)); act(gr, 'idle', 0);
      gr.head.rotation.x = 0.55 * k; gr.hips.rotation.x = 0.12 * k;
      gr.armR.rotation.x = lerp(-1.3, 0, k); gr.armL.rotation.x = lerp(-0.9, 0, k);
      G.position.y = -0.25 * k;
      G.updateMatrixWorld(true);
      const fr = gr.armR.localToWorld(V(0, -0.35, 0)), fl = gr.armL.localToWorld(V(0, -0.35, 0));
      dr.position.copy(fr).add(V(0, 0.15, 0.6)); dr.position.y = Math.max(0.6, dr.position.y);
      wolf.position.copy(fl).add(V(0, -0.5, 0.6)); wolf.position.y = Math.max(0, wolf.position.y);
      petAnim(dr, t, 'fly'); petAnim(wolf, t);
      look(V(7, 3.4, 9), V(0, 3.4, -3.5));
    } };
  },
  s22(D) {
    const s = mkScene('morning'); const T = town(s);
    const me = person(s, 'hoodie_boy', V(4, 0, 10.5), 0, 'smile');
    const sn = inst('pet_snail', V(0, 0, 0), 0, 1.0); s.add(sn);
    const ppl = [person(s, 'rival_blond', V(1.4, 0, 9.4), 0.6, 'happy'), person(s, 'brunette_girl', V(0.2, 0, 10.4), 0.6, 'happy')];
    return { s, u(t) {
      const k = ease(t / D);
      me.o.position.x = lerp(4, 6.3, k); me.o.position.z = lerp(10.5, 8.6, k); me.o.rotation.y = 0.85;
      act(me, k < 0.85 ? 'walk' : 'idle', t);
      me.o.updateMatrixWorld(true);
      sn.position.copy(worldPos(me.o, 'arm_L')).add(V(0, 0.06, 0)); sn.rotation.y = me.o.rotation.y; petAnim(sn, t);
      ppl.forEach((p, i) => act(p, 'cheer', t + i));
      look(vlerp(V(1.5, 2.0, 13.5), V(3.5, 2.1, 12.2), k), V(8.5, 2.2, 5));
    } };
  },
  s23(D) {
    const s = mkScene('morning'); const T = town(s);
    return { s, u(t) {
      const st = T.statue; st.updateMatrixWorld(true);
      const snail = st.localToWorld(V(0.56, 1.345 + 1.5 + 0.12, 0.05));
      const face = st.localToWorld(V(0, 2.9, 0.3));
      const fwd = V(sin(st.rotation.y), 0, cos(st.rotation.y));
      const k = ease(t / (D * 0.75));
      const tgt = vlerp(face, snail, k);
      look(tgt.clone().add(fwd.clone().multiplyScalar(lerp(4.0, 0.9, k))).add(V(0, lerp(-0.4, 0.1, k), 0)), tgt);
    }, o(t) {
      if (t > 1.6) { const k = ease((t - 1.6) / 0.2); sparkle(t, k); }
      if (t > D - 0.45) { g2.fillStyle = `rgba(0,0,0,${ease((t - (D - 0.45)) / 0.45)})`; g2.fillRect(0, 0, W, H); }
    } };
  },
};
function room(D, cal, mood, face) {
  const s = mkScene('room');
  const floor = inst('hardwood_floor', V(0, 0, 0)); s.add(floor);
  const wallM = new THREE.MeshStandardMaterial({ color: 0x9ec9e8, roughness: 0.9 });
  const back = new THREE.Mesh(new THREE.BoxGeometry(6, 3.2, 0.1), wallM); back.position.set(0, 1.6, -3); back.receiveShadow = true; s.add(back);
  const side = new THREE.Mesh(new THREE.BoxGeometry(0.1, 3.2, 6), wallM); side.position.set(-3, 1.6, 0); s.add(side);
  const bed = inst('bed', V(-1.9, 0, -1.6)); s.add(bed);
  const win = inst('window', V(0.6, 1.0, -2.94)); s.add(win);
  const lamp = inst('side_table_lamp', V(2.2, 0, -2.65)); s.add(lamp);
  const c = inst('calendar', V(-0.9, 1.45, -2.94)); s.add(c);
  c.getObjectByName('page').traverse(m => { if (m.isMesh) { m.material = m.material.clone(); m.material.map = TEX[cal]; m.material.emissiveMap = null; } });
  const me = person(s, 'hoodie_boy', V(0.6, 0, -0.6), 0.25, face);
  const sn = inst('pet_snail', V(0.2, 0, 0.6), PI / 2 + 0.2, 1.4); s.add(sn);
  const tape = inst('tape_measure', V(0.6, 0, 0.9), PI); s.add(tape);
  const pl = new THREE.PointLight(0xffe0b0, 6, 6, 2); pl.position.set(2.2, 1.6, -2.3); s.add(pl);
  return { s, u(t) {
    act(me, mood === 'point' ? 'point' : 'idle', t); if (mood !== 'point') me.head.rotation.x = 0.3;
    petAnim(sn, t); sn.position.x = 0.2 - t * 0.02;
    const k = ease(t / D);
    if (mood === 'point') look(vlerp(V(1.9, 1.5, 3.6), V(1.6, 1.4, 3.0), k), V(-0.1, 0.9, -1.0));
    else look(vlerp(V(-0.9, 1.7, 0.9), V(0.4, 0.5, 2.2), k), vlerp(V(-0.9, 1.5, -2.9), V(0.25, 0.1, 0.6), k));
  } };
}
function warningCard(t) {
  const k = ease((t - 0.2) / 0.3); if (k <= 0) return;
  g2.save(); g2.globalAlpha = k; g2.translate(W / 2, 330); g2.rotate(-0.02); g2.scale(0.9 + 0.1 * k, 0.9 + 0.1 * k);
  g2.fillStyle = '#e9d8a6'; roundRect(-300, -150, 600, 300, 18); g2.fill();
  g2.strokeStyle = '#7a5a2a'; g2.lineWidth = 10; g2.stroke();
  g2.fillStyle = '#4a3418'; g2.textAlign = 'center'; g2.textBaseline = 'middle';
  g2.font = '800 34px Inter'; g2.fillText('⚠ WARNING ⚠', 0, -100);
  g2.font = '900 46px Inter';
  ['THE STONE GUARDIAN', 'ONLY SEES WHAT', 'MOVES FAST'].forEach((l, i) => g2.fillText(l, 0, -35 + i * 58));
  g2.restore();
}
function flash(t) { const a = Math.max(0, 0.6 - t * 3); if (a > 0) { g2.fillStyle = `rgba(255,240,220,${a})`; g2.fillRect(0, 0, W, H); } }
function sparkle(t, k) {
  g2.save(); g2.globalAlpha = k;
  for (let i = 0; i < 7; i++) {
    const a = i * 0.9 + t * 1.5, r = 120 + 40 * sin(t * 3 + i);
    const x = W / 2 + cos(a) * r, y = H * 0.42 + sin(a) * r * 0.8, sz = 10 + 6 * sin(t * 6 + i);
    g2.fillStyle = '#fff6c0'; g2.beginPath();
    g2.moveTo(x, y - sz * 2); g2.lineTo(x + sz * 0.5, y); g2.lineTo(x, y + sz * 2); g2.lineTo(x - sz * 0.5, y); g2.closePath(); g2.fill();
    g2.beginPath(); g2.moveTo(x - sz * 2, y); g2.lineTo(x, y + sz * 0.5); g2.lineTo(x + sz * 2, y); g2.lineTo(x, y - sz * 0.5); g2.closePath(); g2.fill();
  }
  g2.restore();
}

// ------------------------------------------------------------------ frame driver
let TL, cur = null, curId = null, composer;
function buildComposer(scene) {
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const night = scene.userData.tod === 'night';
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W / 2, H / 2), night ? 0.55 : 0.3, 0.45, night ? 0.75 : 0.9));
  composer.addPass(new OutputPass());
}
function disposeScene(s) {
  s.traverse(o => { if (o.isMesh && o.userData.own) o.material.dispose(); });
}
window.init = async () => {
  await document.fonts.load('900 64px Inter'); await document.fonts.load('800 40px Inter');
  TL = await (await fetch('/film/timeline.json')).json();
  await loadAll();
  return { frames: Math.ceil(TL[TL.length - 1].end * FPS) };
};
window.frame = (i) => {
  const time = i / FPS;
  let line = TL.find(l => time >= l.start && time < l.end) || TL[TL.length - 1];
  if (line.id !== curId) {
    if (cur) disposeScene(cur.s);
    cur = SHOTS[line.id](line.end - line.start); curId = line.id; buildComposer(cur.s);
  }
  const t = time - line.start;
  cur.u(t);
  composer.render();
  g2.drawImage(renderer.domElement, 0, 0);
  if (cur.o) cur.o(t);
  captions(line, t);
  return out.toDataURL('image/jpeg', 0.9);
};
window.ready = true;
