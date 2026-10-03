'use strict';

/* =====================================================================
 * LoRA Tag Studio — WebUI（純 JavaScript，無需建置）
 * 所有顯示文字都在 app/locales/<lang>.json 的 "ui" 區段。
 * ===================================================================== */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const md = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>');
const h = (html) => { const tp = document.createElement('template'); tp.innerHTML = html.trim(); return tp.content.firstElementChild; };
const debounce = (fn, ms) => { let tm; return (...a) => { clearTimeout(tm); tm = setTimeout(() => fn(...a), ms); }; };
const store = {
  get(k, d) { try { const v = localStorage.getItem('lts.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('lts.' + k, JSON.stringify(v)); } catch { /* 無痕模式 */ } },
};

/* ---------------------------------------------------------------- i18n */
const i18n = { lang: 'en', dict: {}, languages: [] };

function lookup(key) {
  return key.split('.').reduce((o, k) => (o && typeof o === 'object' ? o[k] : undefined), i18n.dict);
}
/** 純文字翻譯（未跳脫，放進 innerHTML 前請用 esc 或改用 th）。 */
function t(key, params = {}) {
  const s = lookup(key);
  if (typeof s !== 'string') return key;
  return s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}
/** 含 HTML 的翻譯（鍵名以 _html 結尾）：模板可信，參數會被跳脫（raw() 包起來的除外）。 */
function th(key, params = {}) {
  const s = lookup(key);
  if (typeof s !== 'string') return esc(key);
  return s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? (params[k]?.__html ?? esc(params[k])) : m));
}
const raw = (html) => ({ __html: html });

function detectLang() {
  const codes = i18n.languages.map((l) => l.code);
  const saved = store.get('lang', null);
  if (saved && codes.includes(saved)) return saved;
  for (const nav of navigator.languages || [navigator.language || 'en']) {
    const c = nav.toLowerCase();
    if (c.startsWith('zh')) {
      const want = /hant|-tw|-hk|-mo/.test(c) ? 'zh-TW' : 'zh-CN';
      if (codes.includes(want)) return want;
    }
    const hit = codes.find((x) => x.toLowerCase() === c) || codes.find((x) => x.toLowerCase().split('-')[0] === c.split('-')[0]);
    if (hit) return hit;
  }
  return 'en';
}

async function loadI18n() {
  const list = await fetch('/api/i18n').then((r) => r.json());
  i18n.languages = list.languages;
  const lang = detectLang();
  const data = await fetch(`/api/i18n/${encodeURIComponent(lang)}`).then((r) => r.json());
  i18n.lang = data.lang;
  i18n.dict = data.ui;
  document.cookie = `lts_lang=${encodeURIComponent(i18n.lang)}; path=/; SameSite=Lax; max-age=31536000`;
  document.documentElement.lang = data.meta.html_lang || i18n.lang;
  document.title = t('app.title');
  $$('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  const sel = $('#langSel');
  sel.innerHTML = i18n.languages.map((l) => `<option value="${esc(l.code)}" ${l.code === i18n.lang ? 'selected' : ''}>${esc(l.name)}</option>`).join('');
  sel.onchange = () => { store.set('lang', sel.value); location.reload(); };
}

const RATINGS = ['general', 'sensitive', 'questionable', 'explicit'];
const STATUSES = ['pending', 'queued', 'processing', 'done', 'error'];
const ratingName = (r) => t(`rating.${r}`);
const statusName = (s) => t(`status.${s}`);

const state = {
  system: null,
  profiles: [],
  project: null,
  images: [],
  selected: new Set(),
  filter: { search: '', status: '', rating: '', blocks: false },
  sideTab: store.get('sideTab', 'settings'),
  pollTimer: null,
  jobId: null,
};

/* ---------------------------------------------------------------- API */
let keyPrompt = null;
async function api(path, { method = 'GET', body, form } = {}) {
  const opts = { method, headers: { 'X-Lang': i18n.lang } };
  if (form) opts.body = form;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers['Content-Type'] = 'application/json'; }
  const res = await fetch('/api' + path, opts);
  if (res.status === 401) { await askApiKey(); return api(path, { method, body, form }); }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* 非 JSON */ }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('json') ? res.json() : res.text();
}

function askApiKey() {
  if (keyPrompt) return keyPrompt;
  keyPrompt = new Promise((resolve) => {
    const m = modal({
      title: t('apikey.title'),
      body: `<p class="muted">${th('apikey.desc_html')}</p>
             <input type="password" id="apiKeyInput" placeholder="API_KEY" autofocus>`,
      closable: false,
      actions: [{ label: t('common.ok'), primary: true, onClick: (close) => {
        const v = $('#apiKeyInput').value.trim();
        if (!v) return;
        document.cookie = `lts_key=${encodeURIComponent(v)}; path=/; SameSite=Lax; max-age=31536000`;
        close(); keyPrompt = null; resolve();
      } }],
    });
    $('#apiKeyInput', m.el).addEventListener('keydown', (e) => { if (e.key === 'Enter') $('.modal-foot .primary', m.el).click(); });
  });
  return keyPrompt;
}

function uploadXHR(pid, files, paths, onProgress) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    files.forEach((f, i) => { fd.append('files', f, f.name); fd.append('paths', paths[i] || f.name); });
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/projects/${pid}/upload`);
    xhr.setRequestHeader('X-Lang', i18n.lang);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded, e.total);
    xhr.onload = async () => {
      if (xhr.status === 401) { await askApiKey(); resolve(uploadXHR(pid, files, paths, onProgress)); return; }
      if (xhr.status >= 400) { let m = xhr.statusText; try { m = JSON.parse(xhr.responseText).detail; } catch { /* */ } reject(new Error(m)); return; }
      resolve(JSON.parse(xhr.responseText));
    };
    xhr.onerror = () => reject(new Error(t('common.network_error')));
    xhr.send(fd);
  });
}

/* ---------------------------------------------------------------- UI helpers */
function toast(msg, type = 'info', ms = 3500) {
  const el = h(`<div class="toast ${type}">${esc(msg)}</div>`);
  $('#toasts').append(el);
  setTimeout(() => el.remove(), ms);
  return el;
}

function modal({ title, body = '', actions = [], size = '', closable = true, onClose } = {}) {
  const el = h(`<div class="modal-backdrop"><div class="modal ${size}">
      <div class="modal-head"><h3>${esc(title)}</h3>${closable ? '<button class="icon-btn" data-x>✕</button>' : ''}</div>
      <div class="modal-body"></div><div class="modal-foot"></div></div></div>`);
  const bodyEl = $('.modal-body', el);
  if (typeof body === 'string') bodyEl.innerHTML = body; else bodyEl.append(body);
  let closed = false;
  const close = () => { if (closed) return; closed = true; el.remove(); document.removeEventListener('keydown', onKey); onClose && onClose(); };
  const onKey = (e) => { if (e.key === 'Escape' && closable && el === $('#modalRoot').lastElementChild) close(); };
  document.addEventListener('keydown', onKey);
  if (closable) {
    $('[data-x]', el).onclick = close;
    el.addEventListener('mousedown', (e) => { if (e.target === el) close(); });
  }
  for (const a of actions) {
    const b = h(`<button class="btn ${a.primary ? 'primary' : ''} ${a.danger ? 'danger' : ''}">${esc(a.label)}</button>`);
    b.onclick = async () => {
      b.disabled = true;
      try { await a.onClick(close, el); } catch (e) { toast(e.message, 'err'); } finally { b.disabled = false; }
    };
    $('.modal-foot', el).append(b);
  }
  $('#modalRoot').append(el);
  setTimeout(() => { const f = $('[autofocus]', el); f && f.focus(); }, 30);
  return { el, close };
}

/** message 為純文字，會被跳脫。 */
function confirmBox(message, { ok = t('common.ok'), danger = false } = {}) {
  return new Promise((resolve) => {
    modal({
      title: t('common.confirm_title'), body: `<p>${esc(message)}</p>`, onClose: () => resolve(false),
      actions: [
        { label: t('common.cancel'), onClick: (close) => close() },
        { label: ok, primary: !danger, danger, onClick: (close) => { resolve(true); close(); } },
      ],
    });
  });
}

function promptBox(title, label, value = '', placeholder = '') {
  return new Promise((resolve) => {
    let done = false;
    const m = modal({
      title,
      body: `<div class="field"><div class="lbl">${esc(label)}</div><input type="text" id="promptInput" value="${esc(value)}" placeholder="${esc(placeholder)}" autofocus></div>`,
      onClose: () => { if (!done) resolve(null); },
      actions: [
        { label: t('common.cancel'), onClick: (close) => close() },
        { label: t('common.ok'), primary: true, onClick: (close) => { done = true; resolve($('#promptInput', m.el).value); close(); } },
      ],
    });
    $('#promptInput', m.el).addEventListener('keydown', (e) => { if (e.key === 'Enter') $('.modal-foot .primary', m.el).click(); });
  });
}

function dropdown(anchor, items) {
  document.querySelectorAll('.menu').forEach((m) => m.remove());
  const menu = h('<div class="menu"></div>');
  for (const it of items) {
    if (it === '-') { menu.append(h('<hr>')); continue; }
    if (it.info) { menu.append(h(`<div class="menu-info ${it.cls || ''}">${esc(it.info)}</div>`)); continue; }
    const b = h(`<button>${esc(it.label)}${it.help ? `<small>${esc(it.help)}</small>` : ''}</button>`);
    b.disabled = !!it.disabled;
    b.onclick = () => { menu.remove(); it.onClick(); };
    menu.append(b);
  }
  anchor.closest('.dropdown').append(menu);
  setTimeout(() => document.addEventListener('click', function off(e) {
    if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener('click', off); }
  }), 0);
}

function copyText(text) {
  navigator.clipboard?.writeText(text).then(() => toast(t('common.copied'), 'ok', 1500), () => toast(t('common.copy_failed'), 'err'));
}

const ratingBadge = (r) => (r ? `<span class="badge r-${esc(r)}">${esc(ratingName(r))}</span>` : '');
const profileOf = (key) => state.profiles.find((p) => p.key === key) || { name: key };
const onOff = (v) => t(v ? 'common.on' : 'common.off');

/* ---------------------------------------------------------------- 系統狀態 */
async function loadSystem() {
  [state.system, state.profiles] = await Promise.all([api('/system'), api('/profiles')]);
  renderSysStatus();
}

function renderSysStatus() {
  const s = state.system;
  if (!s) return;
  const gpu = s.wd14.available_providers.includes('CUDAExecutionProvider');
  const v = s.vlm;
  $('#sysStatus').innerHTML = `
    <span class="pill" title="${esc(t('sys.wd14_title'))}"><span class="dot ${gpu ? 'ok' : 'warn'}"></span>${esc(t(gpu ? 'sys.wd14_gpu' : 'sys.wd14_cpu'))}</span>
    <span class="pill" title="${esc(v.message || '')}"><span class="dot ${v.available ? 'ok' : v.backend === 'none' ? '' : 'err'}"></span>
      VLM · ${esc(v.backend === 'none' ? t('sys.vlm_off') : (v.model || v.backend))}</span>`;
}

/* ---------------------------------------------------------------- Router */
async function route() {
  stopPolling();
  clearTimeout(state.civitaiTimer);
  $('#modalRoot').innerHTML = '';
  const parts = (location.hash || '#/').slice(2).split('/');
  const nav = parts[0] === 'p' ? 'projects' : parts[0] || 'projects';
  $$('.nav a').forEach((a) => a.classList.toggle('active', a.dataset.nav === nav));
  try {
    if (parts[0] === 'p' && parts[1]) {
      await renderProject(parts[1]);
      if (parts[2] === 'civitai') {  // 從首頁的訓練進度點進來
        history.replaceState(null, '', `#/p/${parts[1]}`);
        civitaiDialog({ focusRuns: true });
      }
      return;
    }
    if (parts[0] === 'guide') return await renderGuide(parts[1]);
    if (parts[0] === 'api') return renderApi();
    return await renderProjects();
  } catch (e) {
    $('#view').innerHTML = `<section class="page"><div class="alert err">${esc(e.message)}</div></section>`;
  }
}

/* ================================================================ 專案列表 */
async function renderProjects() {
  const view = $('#view');
  view.innerHTML = `<section class="page">
    <div class="page-head"><h1>${esc(t('projects.title'))}</h1><span class="spacer"></span><button class="btn primary" id="newProj">${esc(t('projects.new'))}</button></div>
    <div id="cvActive"></div>
    <div id="projWrap"><div class="empty">${esc(t('common.loading'))}</div></div></section>`;
  $('#newProj').onclick = newProjectDialog;
  renderActiveTrainings();
  const list = await api('/projects');
  const wrap = $('#projWrap');
  const steps = [1, 2, 3, 4].map((n) => `<div class="step"><span class="n">${n}</span><b>${esc(t(`projects.step${n}_title`))}</b><small>${esc(t(`projects.step${n}_desc`))}</small></div>`).join('');
  const hero = `<div class="hero"><h2>${esc(t('projects.hero_title'))}</h2><p class="muted">${esc(t('projects.hero_desc'))}</p><div class="steps">${steps}</div></div>`;
  if (!list.length) {
    wrap.innerHTML = `${hero}<div class="empty">${esc(t('projects.empty'))}</div>`;
    return;
  }
  wrap.innerHTML = (list.length < 3 ? hero : '') + '<div class="proj-grid"></div>';
  const grid = $('.proj-grid', wrap);
  for (const p of list) {
    const pct = p.image_count ? Math.round((p.done_count / p.image_count) * 100) : 0;
    grid.append(h(`<a class="proj-card" href="#/p/${p.id}" data-pid="${p.id}">
      <div class="cover" ${p.cover_id ? `style="background-image:url('/api/images/${p.cover_id}/thumb')"` : ''}>${p.cover_id ? '' : '◇'}</div>
      <div class="meta">
        <h3>${esc(p.name)}</h3>
        <div class="sub">${esc(p.profile_name)}</div>
        <div class="row" style="font-size:12.5px">
          ${p.settings.trigger ? `<span class="badge accent">${esc(p.settings.trigger)}</span>` : ''}
          <span class="badge cv-badge" hidden></span>
          <span class="muted grow" style="text-align:right">${esc(t('projects.tagged_count', { done: p.done_count, total: p.image_count }))}</span>
        </div>
        <div class="progress"><i style="width:${pct}%"></i></div>
      </div></a>`));
  }
  markProjectCards();
}

/* 首頁：所有專案中正在 Civitai 訓練的任務（與最近 24 小時內結束的） */
function activeRunHTML(run) {
  const st = civitaiRunStatus(run);
  const cls = st === 'succeeded' ? 'ok' : ['failed', 'expired', 'canceled'].includes(st) ? 'err' : 'warn';
  const done = st === 'succeeded' ? t('civitai.finished_ok') : st === 'failed' ? t('civitai.finished_fail')
    : civitaiCanceling(run) ? civitaiCancelNote(run) : '';
  return `<a class="cv-run cv-active-item" href="#/p/${esc(run.project_id)}/civitai">
    <div class="cv-run-head"><span class="badge"><span class="dot ${cls}"></span>${esc(civitaiRunLabel(run))}</span>
      <b>${esc(run.project_name)}</b>
      <span class="muted">${esc(civitaiTypeOf(run.ecosystem)?.label || run.ecosystem)} · ${esc(t('civitai.images', { n: run.image_count }))}
      · ${esc(new Date(run.created_at * 1000).toLocaleString(i18n.lang))}</span></div>
    ${done ? `<div class="muted" style="margin-top:6px;font-size:13px">${esc(done)}</div>` : civitaiProgressHTML(run)}
  </a>`;
}

function markProjectCards() {
  $$('.proj-card[data-pid]').forEach((card) => {
    const run = (state.activeRuns || []).find((r) => r.project_id === card.dataset.pid && CIVITAI_ACTIVE.includes(civitaiRunStatus(r)));
    const badge = $('.cv-badge', card);
    const pct = run?.summary?.progress?.percent;
    badge.hidden = !run;
    badge.textContent = !run ? '' : `☁ ${civitaiCanceling(run) ? t('civitai.st_canceling')
      : pct != null ? t('civitai.badge_pct', { pct: Math.floor(pct) }) : t('civitai.badge_active')}`;
  });
}

async function renderActiveTrainings() {
  clearTimeout(state.civitaiTimer);
  let runs;
  try { runs = await api('/civitai/active'); } catch { return; }
  const box = $('#cvActive');
  if (!box) return;  // 已離開首頁
  state.activeRuns = runs;
  box.innerHTML = runs.length ? `<div class="cv-active"><h2>☁ ${esc(t('civitai.active_title'))}</h2>${runs.map(activeRunHTML).join('')}</div>` : '';
  markProjectCards();
  if (runs.some((r) => CIVITAI_ACTIVE.includes(civitaiRunStatus(r)))) state.civitaiTimer = setTimeout(renderActiveTrainings, 15000);
}

/* 專案頁：Civitai 按鈕上顯示這個專案的訓練進度 */
async function updateCivitaiButton() {
  clearTimeout(state.civitaiTimer);
  const pid = state.project?.id;
  if (!$('#btnCivitai') || !pid) return;
  let runs;
  try { runs = await api('/civitai/active'); } catch { return; }
  const btn = $('#btnCivitai');
  if (!btn || state.project?.id !== pid) return;
  const run = runs.find((r) => r.project_id === pid && CIVITAI_ACTIVE.includes(civitaiRunStatus(r)));
  const pct = run?.summary?.progress?.percent;
  btn.innerHTML = `☁ ${esc(t('ws.civitai'))}${run ? ` <span class="badge accent">${esc(civitaiCanceling(run) ? t('civitai.st_canceling')
    : pct != null ? t('civitai.badge_pct', { pct: Math.floor(pct) }) : t('civitai.badge_active'))}</span>` : ''}`;
  if (run && !$('#cvRuns')) state.civitaiTimer = setTimeout(updateCivitaiButton, 30000);
}

function profileSelectHTML(id, selected) {
  const families = [...new Set(state.profiles.map((p) => p.family))];
  return `<select id="${id}">${families.map((fam) => `<optgroup label="${esc(t(`family.${fam}`))}">${
    state.profiles.filter((p) => p.family === fam).map((p) => `<option value="${p.key}" ${p.key === selected ? 'selected' : ''}>${esc(p.name)}</option>`).join('')
  }</optgroup>`).join('')}</select>`;
}

function newProjectDialog() {
  const lt = state.system.lora_types;
  const body = h(`<div>
    <div class="field"><div class="lbl">${esc(t('newproj.name'))}</div><input type="text" id="npName" placeholder="${esc(t('newproj.name_ph'))}" autofocus></div>
    <div class="field"><div class="lbl">${esc(t('newproj.base_model'))}</div>${profileSelectHTML('npProfile', 'illustrious')}<div class="help" id="npProfileHelp"></div></div>
    <div class="field"><div class="lbl">${esc(t('newproj.lora_type'))}</div><div class="choice-grid">
      ${Object.entries(lt).map(([k, v], i) => `<label class="choice"><input type="radio" name="npType" value="${k}" ${i === 0 ? 'checked' : ''}><b>${esc(v.name)}</b><small class="muted">${esc(v.desc)}</small></label>`).join('')}
    </div></div>
    <div class="row">
      <div class="field grow"><div class="lbl">${esc(t('newproj.trigger'))}</div><input type="text" id="npTrigger" placeholder="${esc(t('newproj.trigger_ph'))}"></div>
      <div class="field grow"><div class="lbl">${esc(t('newproj.class_word'))}</div><input type="text" id="npClass" placeholder="1girl / woman / style"></div>
    </div></div>`);
  const upd = () => { const p = profileOf($('#npProfile', body).value); $('#npProfileHelp', body).textContent = t('newproj.profile_hint', { summary: p.summary, base: p.civitai_base }); };
  $('#npProfile', body).onchange = upd; upd();
  modal({
    title: t('newproj.title'), body, size: 'mid',
    actions: [
      { label: t('common.cancel'), onClick: (c) => c() },
      { label: t('common.create'), primary: true, onClick: async (close) => {
        const p = await api('/projects', { method: 'POST', body: {
          name: $('#npName', body).value, profile: $('#npProfile', body).value,
          lora_type: $('input[name=npType]:checked', body).value,
          trigger: $('#npTrigger', body).value, class_word: $('#npClass', body).value,
        } });
        close(); location.hash = `#/p/${p.id}`;
      } },
    ],
  });
}

/* ================================================================ 專案工作區 */
async function renderProject(pid) {
  const [project, images] = await Promise.all([api(`/projects/${pid}`), api(`/projects/${pid}/images`)]);
  state.project = project;
  state.images = images;
  state.selected.clear();
  const view = $('#view');
  view.innerHTML = `<section class="workspace">
    <div class="ws-head">
      <a href="#/" class="btn ghost" title="${esc(t('ws.back'))}">←</a>
      <div class="ws-title"><h1 id="projName" title="${esc(t('ws.rename_hint'))}"></h1><div class="badges" id="projBadges"></div></div>
      <div class="ws-actions">
        <button class="btn" id="btnFiles">${esc(t('ws.upload_images'))}</button>
        <button class="btn" id="btnFolder">${esc(t('ws.upload_folder'))}</button>
        <div class="dropdown"><button class="btn" id="btnMoreImport">${esc(t('ws.more_import'))} ▾</button></div>
        <div class="dropdown split"><button class="btn primary" id="btnTag">▶ ${esc(t('ws.tag_remaining'))}</button><button class="btn primary" id="btnTagMenu">▾</button></div>
        <button class="btn" id="btnExport">⬇ ${esc(t('ws.export'))}</button>
        <button class="btn" id="btnCivitai">☁ ${esc(t('ws.civitai'))}</button>
        <div class="dropdown"><button class="icon-btn" id="btnProjMenu" title="${esc(t('ws.project_menu'))}">⋯</button></div>
      </div>
    </div>
    <div class="jobbar hidden" id="jobBar"></div>
    <div class="ws-body">
      <aside class="side">
        <div class="tabs">
          <button data-tab="settings">${esc(t('ws.tab_settings'))}</button><button data-tab="stats">${esc(t('ws.tab_stats'))}</button><button data-tab="guide">${esc(t('ws.tab_guide'))}</button>
        </div>
        <div class="tab-body" id="sideBody"></div>
      </aside>
      <div class="main" id="mainArea">
        <div class="toolbar">
          <label class="check" title="${esc(t('ws.select_all'))}"><input type="checkbox" id="selAll"></label>
          <input type="search" id="fSearch" placeholder="${esc(t('ws.search_ph'))}" title="${esc(t('ws.search_help'))}">
          <select id="fStatus"><option value="">${esc(t('ws.all_status'))}</option>${STATUSES.map((k) => `<option value="${k}">${esc(statusName(k))}</option>`).join('')}</select>
          <select id="fRating"><option value="">${esc(t('ws.all_ratings'))}</option>${RATINGS.map((k) => `<option value="${k}">${esc(ratingName(k))}</option>`).join('')}<option value="none">${esc(t('rating.none'))}</option></select>
          <div class="dropdown"><button class="btn sm" id="btnBlocks" title="${esc(t('blocks.button_help'))}"></button></div>
          <label class="check" title="${esc(t('ws.blur_help'))}"><input type="checkbox" id="fBlur"> ${esc(t('ws.blur'))}</label>
          <input type="range" id="fSize" min="130" max="340" step="10" style="width:100px" title="${esc(t('ws.thumb_size'))}">
          <span class="count" id="galCount"></span>
        </div>
        <div class="bulkbar hidden" id="bulkBar"></div>
        <div class="gallery" id="gallery"></div>
      </div>
    </div></section>`;

  renderProjectHead();
  const f = state.filter;
  f.search = ''; f.status = ''; f.rating = ''; f.blocks = false;
  $('#fSearch').oninput = debounce((e) => { f.search = e.target.value; renderGallery(); }, 200);
  $('#fStatus').onchange = (e) => { f.status = e.target.value; renderGallery(); };
  $('#fRating').onchange = (e) => { f.rating = e.target.value; renderGallery(); };
  $('#btnBlocks').onclick = (e) => blocksMenu(e.currentTarget);
  $('#fBlur').checked = store.get('blur', true);
  $('#fBlur').onchange = (e) => { store.set('blur', e.target.checked); $('#gallery').classList.toggle('blur-nsfw', e.target.checked); };
  $('#gallery').classList.toggle('blur-nsfw', $('#fBlur').checked);
  $('#fSize').value = store.get('cardSize', 190);
  const setSize = (v) => { document.documentElement.style.setProperty('--card-size', v + 'px'); store.set('cardSize', +v); };
  setSize($('#fSize').value);
  $('#fSize').oninput = (e) => setSize(e.target.value);
  $('#selAll').onchange = (e) => {
    const ids = filteredImages().map((i) => i.id);
    if (e.target.checked) ids.forEach((id) => state.selected.add(id)); else state.selected.clear();
    renderGallery();
  };

  $('#btnFiles').onclick = () => $('#fileInput').click();
  $('#btnFolder').onclick = () => $('#folderInput').click();
  $('#fileInput').onchange = (e) => { uploadFiles([...e.target.files].map((x) => [x, x.name])); e.target.value = ''; };
  $('#folderInput').onchange = (e) => { uploadFiles([...e.target.files].map((x) => [x, x.webkitRelativePath || x.name])); e.target.value = ''; };
  $('#btnMoreImport').onclick = (e) => dropdown(e.currentTarget, [
    { label: t('menu.import_zip'), help: t('menu.import_zip_help'), onClick: () => $('#fileInput').click() },
    { label: t('menu.import_server'), help: t('menu.import_server_help'), onClick: serverImportDialog },
    { label: t('menu.import_url'), help: t('menu.import_url_help'), onClick: urlImportDialog },
  ]);
  $('#btnTag').onclick = () => startTagging({ only_untagged: true });
  $('#btnTagMenu').onclick = (e) => dropdown(e.currentTarget, [
    { label: t('menu.tag_remaining'), help: t('menu.tag_remaining_help'), onClick: () => startTagging({ only_untagged: true }) },
    { label: t('menu.tag_selected'), help: t('menu.tag_selected_help'), onClick: () => tagSelected() },
    { label: t('menu.retag_all'), help: t('menu.retag_all_help'), onClick: async () => { if (await confirmBox(t('menu.retag_all_confirm'))) startTagging({}); } },
  ]);
  $('#btnExport').onclick = exportDialog;
  $('#btnCivitai').onclick = () => civitaiDialog();
  updateCivitaiButton();
  $('#btnProjMenu').onclick = (e) => dropdown(e.currentTarget, [
    { label: t('menu.rename'), onClick: renameProject },
    { label: t('menu.copy_api'), help: t('menu.copy_api_help'), onClick: () => copyText(`curl ${location.origin}/api/projects/${state.project.id}/captions`) },
    '-',
    { label: t('menu.delete_project'), help: t('menu.delete_project_help'), onClick: deleteProject },
  ]);
  $('#projName').onclick = renameProject;

  setupDrop($('#mainArea'));
  $$('.tabs button').forEach((b) => { b.onclick = () => { state.sideTab = b.dataset.tab; store.set('sideTab', b.dataset.tab); renderSide(); }; });
  renderSide();
  renderGallery();
  if (project.active_job) watchJob(project.active_job.id);
}

function renderProjectHead() {
  const p = state.project;
  const s = p.settings;
  $('#projName').textContent = p.name;
  $('#projBadges').innerHTML = `
    <span class="badge accent">${esc(p.profile_name)}</span>
    <span class="badge">${esc(state.system.lora_types[s.lora_type]?.name || s.lora_type)}</span>
    <span class="badge">${esc(state.system.caption_modes[s.caption_mode] || s.caption_mode)}</span>
    ${s.trigger ? `<span class="badge">${esc(t('ws.trigger_badge', { trigger: s.trigger }))}</span>` : `<span class="badge r-questionable">${esc(t('ws.no_trigger'))}</span>`}`;
  const applied = (state.images || []).filter((i) => i.block_tag_applied).length;
  if (applied) {
    $('#projBadges').insertAdjacentHTML('beforeend', `<span class="badge ok" title="${esc(t('blocks.status_on', { n: applied, tag: s.block_tag }))}">${
      esc(t('blocks.head_badge', { tag: s.block_tag, n: applied }))}</span>`);
  }
}

async function renameProject() {
  const name = await promptBox(t('project.rename_title'), t('project.rename_label'), state.project.name);
  if (name === null || !name.trim()) return;
  state.project = await api(`/projects/${state.project.id}`, { method: 'PATCH', body: { name } });
  renderProjectHead();
}

async function deleteProject() {
  if (!await confirmBox(t('project.delete_confirm', { name: state.project.name }), { ok: t('common.delete'), danger: true })) return;
  await api(`/projects/${state.project.id}`, { method: 'DELETE' });
  toast(t('project.deleted'), 'ok');
  location.hash = '#/';
}

async function reloadImages() {
  state.images = await api(`/projects/${state.project.id}/images`);
  const ids = new Set(state.images.map((i) => i.id));
  [...state.selected].forEach((id) => ids.has(id) || state.selected.delete(id));
  renderGallery();
  if (state.sideTab === 'stats') renderSide();
}

/* ---------------------------------------------------------------- 圖庫 */
function filteredImages() {
  const { search, status, rating, blocks } = state.filter;
  const pats = search.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
  const hit = (pat, text) => {
    if (!pat.includes('*')) return text.includes(pat);
    const re = new RegExp('^' + pat.split('*').map((x) => x.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$');
    return re.test(text);
  };
  return state.images.filter((i) => {
    if (status && i.status !== status) return false;
    if (rating && (rating === 'none' ? i.rating : i.rating !== rating)) return false;
    if (blocks && !i.has_blocks) return false;
    return pats.every((p) => i.tags.some((tg) => hit(p, tg.toLowerCase())) || hit(p, (i.nl_caption || '').toLowerCase()) || hit(p, i.original_name.toLowerCase()));
  });
}

const cardCache = new Map();
function cardFor(img) {
  const sig = [img.thumb_url, img.status, img.caption, img.rating, img.flag, img.error, img.has_blocks, img.block_tag_applied,
    state.selected.has(img.id)].join('|');
  const cached = cardCache.get(img.id);
  if (cached && cached.sig === sig) return cached.el;
  const el = h(`<div class="card ${state.selected.has(img.id) ? 'selected' : ''}" data-id="${img.id}" data-rating="${esc(img.rating || '')}">
      <div class="thumb"><img loading="lazy" src="${img.thumb_url}" alt=""></div>
      <div class="sel"><input type="checkbox" ${state.selected.has(img.id) ? 'checked' : ''}></div>
      <div class="tl">${img.has_blocks ? `<span class="badge blk ${img.block_tag_applied ? 'ok' : ''}" title="${esc(img.block_tag_applied
    ? t('blocks.card_applied', { tag: state.project.settings.block_tag }) : t('blocks.card_title'))}">▭${img.block_tag_applied ? '✓' : ''}</span>` : ''}${img.rating && img.rating !== 'general' ? ratingBadge(img.rating) : ''}<span class="st ${img.status}" title="${esc(statusName(img.status))}${img.error ? ': ' + esc(img.error) : ''}"></span></div>
      ${img.flag ? `<div class="flag" title="${esc(img.flag)}">⚠ ${esc(t('gallery.will_exclude'))}</div>` : ''}
      <div class="cap" title="${esc(img.caption)}">${img.caption ? esc(img.caption) : `<i>${esc(img.original_name)}</i>`}</div>
    </div>`);
  $('.sel', el).onclick = (e) => { e.stopPropagation(); toggleSelect(img.id); };
  el.onclick = (e) => {
    if (e.shiftKey || e.ctrlKey || e.metaKey || state.selected.size) { toggleSelect(img.id); return; }
    openEditor(img.id);
  };
  cardCache.set(img.id, { sig, el });
  return el;
}

function toggleSelect(id) {
  if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
  renderGallery();
}

function renderGallery() {
  const gal = $('#gallery');
  if (!gal) return;
  const list = filteredImages();
  if (!state.images.length) {
    gal.innerHTML = `<div class="upload-empty">
      <h2>${esc(t('gallery.empty_title'))}</h2>
      <p>${esc(t('gallery.empty_desc'))}</p>
      <div class="row"><button class="btn primary" id="emptyFiles">${esc(t('gallery.choose_images'))}</button><button class="btn" id="emptyFolder">${esc(t('gallery.choose_folder'))}</button><button class="btn" id="emptyServer">${esc(t('gallery.server_folder'))}</button></div></div>`;
    $('#emptyFiles').onclick = () => $('#fileInput').click();
    $('#emptyFolder').onclick = () => $('#folderInput').click();
    $('#emptyServer').onclick = serverImportDialog;
  } else {
    gal.replaceChildren(...list.map(cardFor));
    if (!list.length) gal.append(h(`<div class="upload-empty">${esc(t('gallery.no_match'))}</div>`));
  }
  gal.classList.toggle('selecting', state.selected.size > 0);
  const done = state.images.filter((i) => i.status === 'done').length;
  const flagged = state.images.filter((i) => i.flag).length;
  $('#galCount').textContent = t('gallery.count', { shown: list.length, total: state.images.length, done })
    + (flagged ? t('gallery.count_flagged', { n: flagged }) : '');
  $('#selAll').checked = list.length > 0 && list.every((i) => state.selected.has(i.id));
  renderBulkBar();
  renderBlocksButton();
}

function renderBulkBar() {
  const bar = $('#bulkBar');
  const n = state.selected.size;
  bar.classList.toggle('hidden', n === 0);
  if (!n) return;
  bar.innerHTML = `<b>${esc(t('bulk.selected', { n }))}</b>
    <button class="btn sm" data-a="add">＋ ${esc(t('bulk.add'))}</button>
    <button class="btn sm" data-a="remove">－ ${esc(t('bulk.remove'))}</button>
    <button class="btn sm" data-a="replace">⇄ ${esc(t('bulk.replace'))}</button>
    <button class="btn sm" data-a="tag">▶ ${esc(t('bulk.retag'))}</button>
    <button class="btn sm danger" data-a="delete">${esc(t('common.delete'))}</button>
    <span class="grow"></span><button class="btn sm ghost" data-a="clear">${esc(t('bulk.clear'))}</button>`;
  const ids = () => [...state.selected];
  const pid = state.project.id;
  bar.onclick = async (e) => {
    const a = e.target.closest('button')?.dataset.a;
    if (!a) return;
    try {
      if (a === 'clear') { state.selected.clear(); renderGallery(); }
      if (a === 'add') {
        const v = await promptBox(t('bulk.add_title'), t('bulk.add_label'), '', 'smile, outdoors');
        if (v) { const r = await api(`/projects/${pid}/bulk`, { method: 'POST', body: { action: 'add', ids: ids(), tags: v } }); toast(t('bulk.updated', { n: r.changed }), 'ok'); reloadImages(); }
      }
      if (a === 'remove') {
        const v = await promptBox(t('bulk.remove_title'), t('bulk.remove_label'), '', 'watermark, *hair');
        if (v) { const r = await api(`/projects/${pid}/bulk`, { method: 'POST', body: { action: 'remove', ids: ids(), tags: v } }); toast(t('bulk.updated', { n: r.changed }), 'ok'); reloadImages(); }
      }
      if (a === 'replace') replaceDialog('', ids());
      if (a === 'tag') tagSelected();
      if (a === 'delete') {
        if (await confirmBox(t('bulk.delete_confirm', { n }), { ok: t('common.delete'), danger: true })) {
          await api(`/projects/${pid}/images/delete`, { method: 'POST', body: { ids: ids() } });
          state.selected.clear(); toast(t('bulk.deleted'), 'ok'); reloadImages();
        }
      }
    } catch (err) { toast(err.message, 'err'); }
  };
}

/* ---------------------------------------------------------------- 白色色塊 */
// 遮擋用的白色色塊（例如蓋掉其他人）沒寫進 caption 會被 LoRA 學進去；開啟關鍵字後，有色塊的圖片 caption 最後會加上它
function renderBlocksButton() {
  const btn = $('#btnBlocks');
  if (!btn) return;
  const n = state.images.filter((i) => i.has_blocks).length;
  const applied = state.images.filter((i) => i.block_tag_applied).length;
  btn.classList.toggle('active', state.filter.blocks);
  btn.title = applied ? t('blocks.status_on', { n: applied, tag: state.project.settings.block_tag }) : t('blocks.button_help');
  btn.innerHTML = `▭ ${esc(t('blocks.button'))}${applied ? ` <span class="badge ok">✓ ${esc(t('blocks.button_on', { n: applied }))}</span>`
    : n ? ` <span class="badge">${n}</span>` : ''} ▾`;
  renderProjectHead();
}

async function scanBlocks(force) {
  const busy = toast(t('blocks.scanning'), 'info', 600000);
  try {
    const r = await api(`/projects/${state.project.id}/blocks/scan`, { method: 'POST', body: { force } });
    toast(t('blocks.scan_done', { scanned: r.scanned, n: r.with_blocks }), 'ok', 6000);
    await reloadImages();
  } catch (e) { toast(e.message, 'err'); } finally { busy.remove(); }
}

async function setBlockTagAuto(on) {
  const s = state.project.settings;
  state.project = await api(`/projects/${state.project.id}`, { method: 'PATCH', body: { settings: { block_tag_auto: on } } });
  renderProjectHead(); renderSide(); await reloadImages();
  const n = state.images.filter((i) => i.has_blocks).length;
  toast(on ? t('blocks.auto_on_done', { n, tag: s.block_tag }) : t('blocks.auto_off_done', { tag: s.block_tag }), 'ok', 9000);
}

function blocksMenu(anchor) {
  const s = state.project.settings;
  const withBlocks = state.images.filter((i) => i.has_blocks);
  const n = withBlocks.length;
  const unscanned = state.images.filter((i) => i.blocks == null).length;
  const tag = s.block_tag;
  const applied = state.images.filter((i) => i.block_tag_applied).length;
  const status = applied ? { info: t('blocks.status_on', { n: applied, tag }), cls: 'ok' }
    : n ? { info: t('blocks.status_off', { n }), cls: 'warn' } : null;
  dropdown(anchor, [
    ...(status ? [status, '-'] : []),
    unscanned ? { label: t('blocks.scan_n', { n: unscanned }), help: t('blocks.scan_help'), onClick: () => scanBlocks(false) }
      : { label: t('blocks.rescan'), help: t('blocks.rescan_help'), onClick: () => scanBlocks(true) },
    '-',
    { label: s.block_tag_auto ? t('blocks.auto_off', { tag }) : t('blocks.auto_on', { tag, n }),
      help: s.block_tag_auto ? t('blocks.auto_off_help') : t('blocks.auto_on_help', { tag }),
      onClick: () => setBlockTagAuto(!s.block_tag_auto).catch((e) => toast(e.message, 'err')) },
    { label: state.filter.blocks ? t('blocks.show_all') : t('blocks.show_only', { n }), disabled: !n && !state.filter.blocks,
      onClick: () => { state.filter.blocks = !state.filter.blocks; renderGallery(); } },
    { label: t('blocks.select_all', { n }), help: t('blocks.select_help'), disabled: !n, onClick: () => {
      state.selected = new Set(withBlocks.map((i) => i.id));
      state.filter.blocks = true;
      renderGallery();
    } },
    { label: t('blocks.delete_all', { n }), disabled: !n, onClick: async () => {
      if (!await confirmBox(t('bulk.delete_confirm', { n }), { ok: t('common.delete'), danger: true })) return;
      try {
        await api(`/projects/${state.project.id}/images/delete`, { method: 'POST', body: { ids: withBlocks.map((i) => i.id) } });
        state.selected.clear(); state.filter.blocks = false;
        toast(t('bulk.deleted'), 'ok'); reloadImages();
      } catch (e) { toast(e.message, 'err'); }
    } },
  ]);
}

/** 新匯入的圖片裡有白色色塊、但還沒開啟關鍵字時提醒。 */
function blockHint(ids) {
  if (!ids?.length || state.project.settings.block_tag_auto) return;
  const set = new Set(ids);
  const n = state.images.filter((i) => set.has(i.id) && i.has_blocks).length;
  if (n) toast(t('blocks.import_hint', { n }), 'warn', 9000);
}

/* ---------------------------------------------------------------- 上傳 / 匯入 */
function setupDrop(area) {
  let depth = 0, overlay = null;
  const show = (on) => {
    if (on && !overlay) { overlay = h(`<div class="dropzone">${esc(t('upload.drop'))}</div>`); area.append(overlay); }
    if (!on && overlay) { overlay.remove(); overlay = null; }
  };
  area.addEventListener('dragenter', (e) => { if ([...e.dataTransfer.types].includes('Files')) { depth++; show(true); e.preventDefault(); } });
  area.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) show(false); });
  area.addEventListener('dragover', (e) => { if ([...e.dataTransfer.types].includes('Files')) e.preventDefault(); });
  area.addEventListener('drop', async (e) => {
    e.preventDefault(); depth = 0; show(false);
    const entries = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    const files = [];
    const walk = async (entry, prefix) => {
      if (entry.isFile) {
        const f = await new Promise((res, rej) => entry.file(res, rej));
        files.push([f, prefix + f.name]);
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          for (const ch of batch) await walk(ch, prefix + entry.name + '/');
        } while (batch.length);
      }
    };
    if (entries.length) { for (const en of entries) await walk(en, ''); } else { [...e.dataTransfer.files].forEach((f) => files.push([f, f.name])); }
    uploadFiles(files);
  });
}

function importToast(r) {
  const msg = t('upload.added', { n: r.added }) + (r.skipped.length ? t('upload.skipped', { n: r.skipped.length, reason: r.skipped[0].reason }) : '');
  toast(msg, r.added ? 'ok' : 'warn', 6000);
}

async function uploadFiles(pairs) {
  const ok = /\.(png|jpe?g|jfif|webp|bmp|gif|tiff?|avif|txt|caption|zip)$/i;
  pairs = pairs.filter(([, p]) => ok.test(p) && !p.split('/').some((seg) => seg.startsWith('.')));
  if (!pairs.length) { toast(t('upload.none'), 'warn'); return; }
  // 同名的圖片與 .txt 要放在同一批，依路徑排序後分批
  pairs.sort((a, b) => a[1].localeCompare(b[1]));
  const batches = [];
  let cur = [], size = 0;
  for (const pair of pairs) {
    const stem = pair[1].replace(/\.[^.]+$/, '');
    const prevStem = cur.length ? cur[cur.length - 1][1].replace(/\.[^.]+$/, '') : null;
    if (cur.length && (cur.length >= 40 || size > 80e6) && stem !== prevStem) { batches.push(cur); cur = []; size = 0; }
    cur.push(pair); size += pair[0].size;
  }
  if (cur.length) batches.push(cur);
  const total = pairs.reduce((s, p) => s + p[0].size, 0) || 1;
  let sent = 0;
  const result = { added: 0, skipped: [], added_ids: [] };
  const bar = $('#jobBar');
  bar.classList.remove('hidden');
  const draw = (loaded) => {
    const pct = Math.round(((sent + loaded) / total) * 100);
    bar.innerHTML = `<span>${esc(t('upload.progress', { pct }))}</span><div class="progress"><i style="width:${pct}%"></i></div>`;
  };
  draw(0);
  try {
    for (const b of batches) {
      const r = await uploadXHR(state.project.id, b.map((x) => x[0]), b.map((x) => x[1]), draw);
      sent += b.reduce((s, p) => s + p[0].size, 0);
      result.added += r.added; result.skipped.push(...r.skipped); result.added_ids.push(...(r.added_ids || []));
      draw(0);
    }
    importToast(result);
  } catch (e) {
    toast(t('upload.failed', { error: e.message }), 'err');
  } finally {
    if (!state.jobId) bar.classList.add('hidden');
    await reloadImages();
    if (result.added && !state.project.settings.trigger) toast(t('upload.no_trigger'), 'warn');
    blockHint(result.added_ids);
  }
}

async function serverImportDialog() {
  const dirs = await api('/import-dirs');
  const body = dirs.length
    ? `<p class="muted">${th('server_import.desc_html')}</p>
       <table><tr><th>${esc(t('server_import.folder'))}</th><th>${esc(t('server_import.images'))}</th><th></th></tr>${dirs.map((d) => `<tr><td class="mono">/${esc(d.path)}</td><td>${d.images}</td>
       <td style="text-align:right"><button class="btn sm" data-path="${esc(d.path)}">${esc(t('common.import'))}</button></td></tr>`).join('')}</table>
       <label class="check" style="margin-top:10px"><input type="checkbox" id="impRec" checked> ${esc(t('server_import.recursive'))}</label>`
    : `<div class="alert warn">${th('server_import.none_html')}</div>`;
  const m = modal({ title: t('server_import.title'), body, size: 'mid' });
  $$('[data-path]', m.el).forEach((b) => {
    b.onclick = async () => {
      b.disabled = true; b.textContent = t('server_import.importing');
      try {
        const r = await api(`/projects/${state.project.id}/import-server`, { method: 'POST', body: { path: b.dataset.path, recursive: $('#impRec', m.el).checked } });
        importToast(r); m.close(); reloadImages().then(() => blockHint(r.added_ids));
      } catch (e) { toast(e.message, 'err'); b.disabled = false; b.textContent = t('common.import'); }
    };
  });
}

function urlImportDialog() {
  modal({
    title: t('url_import.title'), size: 'mid',
    body: '<textarea id="urlList" rows="8" placeholder="https://example.com/a.png&#10;https://example.com/b.jpg" autofocus></textarea>',
    actions: [{ label: t('common.cancel'), onClick: (c) => c() }, { label: t('common.import'), primary: true, onClick: async (close, el) => {
      const urls = $('#urlList', el).value.split(/\s+/).filter(Boolean);
      if (!urls.length) return;
      const r = await api(`/projects/${state.project.id}/import-urls`, { method: 'POST', body: { urls } });
      importToast(r); close(); reloadImages().then(() => blockHint(r.added_ids));
    } }],
  });
}

/* ---------------------------------------------------------------- 標註工作 */
async function startTagging(body) {
  try {
    const job = await api(`/projects/${state.project.id}/tag`, { method: 'POST', body });
    if (!job.total) { toast(job.message || t('job.nothing'), 'warn'); return; }
    toast(t('job.started', { n: job.total }), 'ok');
    watchJob(job.id);
  } catch (e) { toast(e.message, 'err', 8000); }
}

async function tagSelected() {
  const ids = [...state.selected];
  if (!ids.length) { toast(t('job.select_first'), 'warn'); return; }
  if (await confirmBox(t('job.retag_selected_confirm', { n: ids.length }))) startTagging({ ids });
}

function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; state.jobId = null; }

function watchJob(jobId) {
  stopPolling();
  state.jobId = jobId;
  let lastDone = -1, ticks = 0;
  const tick = async () => {
    if (state.jobId !== jobId) return;
    let job;
    try { job = await api(`/jobs/${jobId}`); } catch { state.pollTimer = setTimeout(tick, 3000); return; }
    const bar = $('#jobBar');
    if (!bar) return;
    const pct = job.total ? (job.done / job.total) * 100 : 0;
    const finished = ['done', 'cancelled', 'error'].includes(job.status);
    bar.classList.remove('hidden');
    bar.innerHTML = `<span>${esc(job.message || '')} ${job.done}/${job.total}${job.failed ? esc(t('job.failed_suffix', { n: job.failed })) : ''}</span>
      <div class="progress"><i style="width:${pct}%"></i></div><button class="btn sm" id="cancelJob">${esc(t('common.cancel'))}</button>`;
    $('#cancelJob').onclick = () => api(`/jobs/${jobId}/cancel`, { method: 'POST' });
    ticks++;
    if (job.done !== lastDone && (ticks % 2 === 0 || finished)) { lastDone = job.done; await reloadImages(); refreshEditor(); }
    if (finished) {
      state.jobId = null;
      bar.classList.add('hidden');
      await reloadImages(); refreshEditor();
      const type = job.status === 'done' && !job.failed ? 'ok' : job.status === 'error' ? 'err' : 'warn';
      toast(job.message + (job.errors?.length ? `: ${job.errors[0].error}` : ''), type, 7000);
      if (state.sideTab === 'stats') renderSide();
      return;
    }
    state.pollTimer = setTimeout(tick, 1200);
  };
  tick();
}

/* ---------------------------------------------------------------- 側欄 */
function renderSide() {
  $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === state.sideTab));
  const body = $('#sideBody');
  if (!body) return;
  if (state.sideTab === 'stats') return renderStats(body);
  if (state.sideTab === 'guide') return renderSideGuide(body);
  return renderSettings(body);
}

const SETTINGS_SECTIONS = () => {
  const sys = state.system;
  const shortModel = sys.wd14.default_model.split('/').pop();
  const vlmStatus = sys.vlm.backend === 'none' ? t('settings.vlm_server_off') : `${sys.vlm.model || sys.vlm.backend} (${sys.vlm.message})`;
  return [
    { id: 'basic', open: true, fields: [
      { key: 'profile', type: 'profile' },
      { key: 'lora_type', type: 'select', options: Object.entries(sys.lora_types).map(([k, v]) => [k, v.name]) },
      { key: 'trigger', type: 'text', placeholder: 'mychar' },
      { key: 'class_word', type: 'text', placeholder: '1girl / woman / style' },
      { key: 'caption_mode', type: 'select', options: Object.entries(sys.caption_modes) },
    ] },
    { id: 'wd14', open: true, fields: [
      { key: 'use_wd14', type: 'check' },
      { key: 'wd14_model', type: 'select', options: [['', t('settings.wd14_default', { model: shortModel })], ...Object.entries(sys.wd14.models)] },
      { key: 'general_threshold', type: 'range', min: 0.1, max: 0.9, step: 0.01 },
      { key: 'character_threshold', type: 'range', min: 0.3, max: 1, step: 0.01 },
      { key: 'max_tags', type: 'number', min: 0 },
      { key: 'include_character_tags', type: 'check' },
      { key: 'vlm_tags', type: 'select', options: [
        ['off', t('settings.vlm_tags_off')], ['extra', t('settings.vlm_tags_extra')],
        ['merge', t('settings.vlm_tags_merge')], ['only', t('settings.vlm_tags_only')],
      ] },
      { key: 'vlm_artist_tags', type: 'check' },
    ] },
    { id: 'vlm', fields: [
      { key: 'use_vlm', type: 'check', helpParams: { status: vlmStatus } },
      { key: 'vlm_detail', type: 'select', options: ['short', 'medium', 'detailed'].map((k) => [k, t(`settings.detail_${k}`)]) },
      { key: 'vlm_nsfw', type: 'check' },
      { key: 'nl_position', type: 'select', options: [['before_tags', t('settings.nl_before')], ['after_tags', t('settings.nl_after')]] },
      { key: 'vlm_extra_prompt', type: 'textarea', placeholder: 'e.g. Always mention the camera angle.' },
    ] },
    { id: 'format', fields: [
      { key: 'prefix_tags', type: 'text', placeholder: 'source_anime' },
      { key: 'append_tags', type: 'text' },
      { key: 'block_tag_auto', type: 'check' },
      { key: 'block_tag', type: 'text', placeholder: 'white rectangle' },
      { key: 'include_rating', type: 'check' },
      { key: 'add_quality_tags', type: 'check' },
      { key: 'quality_tags', type: 'text' },
      { key: 'underscore_to_space', type: 'check' },
      { key: 'escape_parentheses', type: 'check' },
    ] },
    { id: 'prune', open: true, fields: [
      { key: 'prune_groups', type: 'multi', options: Object.entries(sys.prune_groups) },
      { key: 'blacklist', type: 'textarea', placeholder: 'watermark, *username, simple background' },
    ] },
  ];
};

function fieldHTML(f, s) {
  const v = s[f.key];
  const label = esc(t(`settings.f.${f.key}`));
  const helpText = lookup(`settings.h.${f.key}`) ? t(`settings.h.${f.key}`, f.helpParams || {}) : '';
  const help = helpText ? `<div class="help">${esc(helpText)}</div>` : '';
  switch (f.type) {
    case 'profile':
      return `<div class="field"><div class="lbl">${label}</div>${profileSelectHTML('s_' + f.key, v)}<div class="help">${esc(profileOf(v).summary || '')}</div></div>`;
    case 'select':
      return `<div class="field"><div class="lbl">${label}</div><select id="s_${f.key}">${f.options.map(([k, l]) => `<option value="${esc(k)}" ${k === v ? 'selected' : ''}>${esc(l)}</option>`).join('')}</select>${help}</div>`;
    case 'text':
      return `<div class="field"><div class="lbl">${label}</div><input type="text" id="s_${f.key}" value="${esc(v)}" placeholder="${esc(f.placeholder || '')}">${help}</div>`;
    case 'textarea':
      return `<div class="field"><div class="lbl">${label}</div><textarea id="s_${f.key}" rows="2" placeholder="${esc(f.placeholder || '')}">${esc(v)}</textarea>${help}</div>`;
    case 'number':
      return `<div class="field"><div class="lbl">${label}</div><input type="number" id="s_${f.key}" value="${esc(v)}" min="${f.min ?? ''}">${help}</div>`;
    case 'range':
      return `<div class="field"><div class="lbl">${label}<b id="s_${f.key}_v">${Number(v).toFixed(2)}</b></div><input type="range" id="s_${f.key}" value="${v}" min="${f.min}" max="${f.max}" step="${f.step}">${help}</div>`;
    case 'check':
      return `<div class="field"><label class="check"><input type="checkbox" id="s_${f.key}" ${v ? 'checked' : ''}> ${label}</label>${help}</div>`;
    case 'multi':
      return `<div class="field"><div class="lbl">${label}</div>${f.options.map(([k, l]) => `<label class="check"><input type="checkbox" data-multi="${f.key}" value="${k}" ${v.includes(k) ? 'checked' : ''}> ${esc(l)}</label>`).join('')}${help}</div>`;
    default: return '';
  }
}

function readSettings(body) {
  const out = {};
  for (const sec of SETTINGS_SECTIONS()) {
    for (const f of sec.fields) {
      if (f.type === 'multi') { out[f.key] = $$(`[data-multi="${f.key}"]:checked`, body).map((c) => c.value); continue; }
      const el = $(`#s_${f.key}`, body);
      if (!el) continue;
      if (f.type === 'check') out[f.key] = el.checked;
      else if (f.type === 'range') out[f.key] = parseFloat(el.value);
      else if (f.type === 'number') out[f.key] = parseInt(el.value || '0', 10);
      else out[f.key] = el.value;
    }
  }
  return out;
}

// 與後端 pipeline.config_problem 相同的規則：設定組合不會產生內容時擋下，其他可疑組合提醒
function settingsWarnings(s) {
  if (s.caption_mode === 'trigger_only') return [];
  const vlmUsed = s.use_vlm && ['natural', 'hybrid'].includes(s.caption_mode);
  const vlmTags = s.vlm_tags && s.vlm_tags !== 'off';
  if (!s.use_wd14 && !vlmUsed && !vlmTags) return [['err', t('settings.warn_nothing')]];
  const out = [];
  if ((vlmUsed || vlmTags) && state.system.vlm.backend === 'none') out.push(['warn', t('settings.vlm_server_off')]);
  if (s.use_vlm && !vlmUsed) out.push(['warn', t('settings.warn_vlm_ignored')]);
  if (s.include_rating && !s.use_wd14) out.push(['warn', t('settings.warn_rating_needs_wd14')]);
  if (s.caption_mode === 'natural' && vlmUsed && (s.use_wd14 || vlmTags)) out.push(['info', t('settings.warn_natural_tags_unused')]);
  return out;
}

function renderSettings(body) {
  const s = state.project.settings;
  const lt = state.system.lora_types[s.lora_type];
  body.innerHTML = '<div id="setWarn"></div>' + SETTINGS_SECTIONS().map((sec) => `<details class="section" ${sec.open ? 'open' : ''}><summary>${esc(t(`settings.sec.${sec.id}`))}</summary><div class="inner">
      ${sec.fields.map((f) => fieldHTML(f, s)).join('')}
      ${sec.id === 'basic' && lt ? `<ul class="tips">${lt.tips.map((x) => `<li>${md(x)}</li>`).join('')}</ul>` : ''}
    </div></details>`).join('') + `
    <div class="sticky-save"><button class="btn primary grow" id="saveSettings">${esc(t('settings.save'))}</button><button class="btn" id="applyMenuBtn" title="${esc(t('settings.apply_title'))}">${esc(t('settings.apply_existing'))} ▾</button></div>`;
  $$('input[type=range]', body).forEach((r) => { r.oninput = () => { $(`#${r.id}_v`, body).textContent = Number(r.value).toFixed(2); }; });
  const showWarnings = () => {
    $('#setWarn', body).innerHTML = settingsWarnings(readSettings(body)).map(([k, m]) => `<div class="alert ${k}">${esc(m)}</div>`).join('');
  };
  showWarnings();
  let dirty = false;
  body.oninput = () => { dirty = true; $('#saveSettings', body).textContent = t('settings.save') + ' •'; showWarnings(); };

  $('#s_profile', body).onchange = async (e) => {
    const p = profileOf(e.target.value);
    if (await confirmBox(t('settings.switch_profile_confirm', { name: p.name }))) {
      state.project = await api(`/projects/${state.project.id}`, { method: 'PATCH', body: { settings: { profile: p.key }, reset_to_profile: true } });
      toast(t('settings.profile_applied'), 'ok'); renderProjectHead(); renderSide(); reloadImages();
    } else { e.target.value = s.profile; }
  };
  $('#s_lora_type', body).onchange = async (e) => {
    const tp = state.system.lora_types[e.target.value];
    const groups = tp.prune_groups.map((g) => state.system.prune_groups[g]).join(', ') || t('settings.none');
    if (await confirmBox(t('settings.lora_type_confirm', { name: tp.name, groups }), { ok: t('settings.apply') })) {
      $$('[data-multi=prune_groups]', body).forEach((c) => { c.checked = tp.prune_groups.includes(c.value); });
    }
    dirty = true;
  };

  const save = async () => {
    state.project = await api(`/projects/${state.project.id}`, { method: 'PATCH', body: { settings: readSettings(body) } });
    dirty = false;
    renderProjectHead(); renderSide(); await reloadImages();
    toast(t('settings.saved'), 'ok', 5000);
  };
  $('#saveSettings', body).onclick = () => save().catch((e) => toast(e.message, 'err'));
  $('#applyMenuBtn', body).onclick = (e) => {
    e.currentTarget.parentElement.classList.add('dropdown');
    dropdown(e.currentTarget, [
      { label: t('settings.reapply'), help: t('settings.reapply_help'), onClick: () => applyBulk('reapply', dirty ? save : null) },
      { label: t('settings.filter'), help: t('settings.filter_help'), onClick: () => applyBulk('filter', dirty ? save : null) },
      { label: t('settings.clear_nl'), onClick: () => applyBulk('clear_nl', null) },
    ]);
  };
}

async function applyBulk(action, saveFirst) {
  try {
    if (saveFirst) await saveFirst();
    if (action !== 'filter' && !await confirmBox(t(action === 'reapply' ? 'settings.reapply_confirm' : 'settings.clear_nl_confirm'))) return;
    const r = await api(`/projects/${state.project.id}/bulk`, { method: 'POST', body: { action } });
    toast(t('bulk.updated', { n: r.changed }), 'ok');
    reloadImages();
  } catch (e) { toast(e.message, 'err'); }
}

async function renderStats(body) {
  body.innerHTML = `<div class="muted">${esc(t('common.loading'))}</div>`;
  const st = await api(`/projects/${state.project.id}/stats`);
  const max = st.tags[0]?.count || 1;
  body.innerHTML = `
    <div class="stat-top">${Object.entries(st.ratings).map(([k, v]) => k === 'unknown' ? `<span class="badge">${esc(t('rating.none'))} ${v}</span>` : `<span class="badge r-${k}">${esc(ratingName(k))} ${v}</span>`).join('')}</div>
    <div class="row" style="margin-bottom:8px"><input type="search" id="statFilter" placeholder="${esc(t('stats.filter_ph', { n: st.unique_tags }))}"></div>
    <div class="muted" style="font-size:12px;margin-bottom:6px">${esc(t('stats.help'))}</div>
    <div id="statList"></div>`;
  const list = $('#statList', body);
  const draw = (q) => {
    const rows = st.tags.filter((x) => !q || x.tag.toLowerCase().includes(q)).slice(0, 400);
    list.innerHTML = rows.map((x) => `<div class="stat-row" data-tag="${esc(x.tag)}">
      <span class="t" title="${esc(x.tag)}">${esc(x.tag)}</span><span class="bar"><i style="width:${(x.count / max) * 100}%"></i></span>
      <span class="c">${x.count}</span>
      <span class="a"><button class="icon-btn" data-a="rm" title="${esc(t('stats.remove_all'))}">🗑</button><button class="icon-btn" data-a="rep" title="${esc(t('stats.replace'))}">✎</button><button class="icon-btn" data-a="bl" title="${esc(t('stats.blacklist'))}">⛔</button></span></div>`).join('')
      || `<div class="muted">${esc(t('stats.no_tags'))}</div>`;
  };
  draw('');
  $('#statFilter', body).oninput = (e) => draw(e.target.value.toLowerCase());
  list.onclick = async (e) => {
    const row = e.target.closest('.stat-row');
    if (!row) return;
    const tag = row.dataset.tag;
    const a = e.target.closest('button')?.dataset.a;
    const pid = state.project.id;
    try {
      if (!a) { $('#fSearch').value = tag; state.filter.search = tag; renderGallery(); return; }
      if (a === 'rm' && await confirmBox(t('stats.remove_confirm', { tag }))) {
        const r = await api(`/projects/${pid}/bulk`, { method: 'POST', body: { action: 'remove', tags: [tag] } });
        toast(t('stats.removed', { n: r.changed }), 'ok'); reloadImages();
      }
      if (a === 'rep') replaceDialog(tag, null);
      if (a === 'bl') {
        const bl = [state.project.settings.blacklist, tag].filter(Boolean).join(', ');
        state.project = await api(`/projects/${pid}`, { method: 'PATCH', body: { settings: { blacklist: bl } } });
        await api(`/projects/${pid}/bulk`, { method: 'POST', body: { action: 'remove', tags: [tag] } });
        toast(t('stats.blacklisted', { tag }), 'ok'); reloadImages();
      }
    } catch (err) { toast(err.message, 'err'); }
  };
}

function replaceDialog(find, ids) {
  modal({
    title: t('replace.title'), size: 'mid',
    body: `<div class="row"><div class="field grow"><div class="lbl">${esc(t('replace.find'))}</div><input type="text" id="rpFind" value="${esc(find)}" ${find ? '' : 'autofocus'}></div>
           <div class="field grow"><div class="lbl">${esc(t('replace.with'))}</div><input type="text" id="rpRepl" ${find ? 'autofocus' : ''}></div></div>
           <p class="muted">${esc(ids ? t('replace.scope_selected', { n: ids.length }) : t('replace.scope_all'))}</p>`,
    actions: [{ label: t('common.cancel'), onClick: (c) => c() }, { label: t('replace.action'), primary: true, onClick: async (close, el) => {
      const r = await api(`/projects/${state.project.id}/bulk`, { method: 'POST', body: { action: 'replace', ids, find: $('#rpFind', el).value, replace: $('#rpRepl', el).value } });
      toast(t('bulk.updated', { n: r.changed }), 'ok'); close(); reloadImages();
    } }],
  });
}

async function renderSideGuide(body) {
  const p = await api(`/profiles/${state.project.settings.profile}`);
  body.innerHTML = guideHTML(p, true);
  bindCopy(body);
}

function bindCopy(root) {
  $$('[data-copy]', root).forEach((b) => { b.onclick = () => copyText(b.parentElement.querySelector('code').innerText); });
}

/* ---------------------------------------------------------------- 單張編輯器 */
let editor = null;

function openEditor(id) {
  const list = filteredImages();
  let idx = list.findIndex((i) => i.id === id);
  if (idx < 0) return;
  const body = h(`<div class="editor">
      <div class="ed-img"><img id="edImg" alt=""></div>
      <div class="ed-panel">
        <div class="ed-meta" id="edMeta"></div>
        <div id="edAlerts"></div>
        <div><div class="field" style="margin:0"><div class="lbl">${esc(t('editor.tags'))} <small>${esc(t('editor.tags_help'))}</small></div></div>
          <div class="chips" id="edChips"></div></div>
        <div class="field" style="margin:0"><div class="lbl">${esc(t('editor.nl'))} <small>${esc(t('editor.nl_help'))}</small></div><textarea id="edNl" rows="4" placeholder="${esc(t('editor.nl_ph'))}"></textarea></div>
        <div class="field" style="margin:0"><div class="lbl">${esc(t('editor.final'))}<span class="tok" id="edTok"></span></div><div class="preview" id="edPreview"></div></div>
        <div class="row" style="flex-wrap:wrap">
          <button class="btn" id="edRetag">▶ ${esc(t('editor.retag'))}</button>
          <button class="btn" id="edCopy">${esc(t('editor.copy'))}</button>
          <button class="btn danger" id="edDelete">${esc(t('editor.delete'))}</button>
          <span class="grow"></span>
          <button class="btn" id="edPrev" title="←">‹ ${esc(t('editor.prev'))}</button><button class="btn" id="edNext" title="→">${esc(t('editor.next'))} ›</button>
        </div>
        <div class="muted" style="font-size:12px">${esc(t('editor.autosave'))}</div>
      </div></div>`);
  const m = modal({ title: t('editor.title_simple'), body, size: 'wide', onClose: () => { document.removeEventListener('keydown', keys); editor = null; } });
  const title = $('.modal-head h3', m.el);
  let tags = [], cur = null;
  const tagSuggest = h('<datalist id="tagSuggest"></datalist>');
  body.append(tagSuggest);
  api(`/projects/${state.project.id}/stats`).then((st) => { tagSuggest.innerHTML = st.tags.slice(0, 2000).map((x) => `<option value="${esc(x.tag)}">`).join(''); }).catch(() => {});

  const persist = async (fields) => {
    try {
      const updated = await api(`/projects/${state.project.id}/images/${cur.id}`, { method: 'PATCH', body: fields });
      const i = state.images.findIndex((x) => x.id === updated.id);
      if (i >= 0) state.images[i] = updated;
      cur = updated;
      showPreview();
      renderGallery();
    } catch (e) { toast(t('editor.save_failed', { error: e.message }), 'err'); }
  };
  const saveTags = debounce(() => persist({ tags }), 350);
  const saveNl = debounce(() => persist({ nl_caption: $('#edNl', body).value }), 700);

  function showPreview() {
    $('#edPreview', body).textContent = cur.caption || t('editor.empty');
    const limit = profileOf(state.project.settings.profile).token_limit || 75;
    const over = cur.token_estimate > limit;
    $('#edTok', body).className = 'tok' + (over ? ' over' : '');
    $('#edTok', body).textContent = t('editor.tokens', { n: cur.token_estimate }) + (over && limit === 75 ? t('editor.tokens_over') : '');
    const alerts = [];
    if (cur.flag) alerts.push(`<div class="alert err">⚠ ${esc(cur.flag)}</div>`);
    if (cur.error) alerts.push(`<div class="alert warn">${esc(cur.error)}</div>`);
    $('#edAlerts', body).innerHTML = alerts.join('');
    const manual = cur.blocks?.override != null;
    $('#edMeta', body).innerHTML = `<b style="color:var(--text)">${esc(cur.original_name)}</b> · ${cur.width}×${cur.height} ${ratingBadge(cur.rating)}
      <span class="badge"><span class="st ${cur.status}" style="width:8px;height:8px;border:0;border-radius:50%;display:inline-block"></span>${esc(statusName(cur.status))}</span>
      <button class="btn sm ${cur.has_blocks ? 'active' : ''}" id="edBlocks" title="${esc(t('blocks.ed_help'))}">▭ ${esc(t(cur.has_blocks ? 'blocks.ed_has' : 'blocks.ed_none'))}${manual ? ` · ${esc(t('blocks.ed_manual'))}` : ''}</button>
      ${manual ? `<button class="btn sm ghost" id="edBlocksReset" title="${esc(t('blocks.ed_reset'))}">↺</button>` : ''}`;
    $('#edBlocks', body).onclick = () => persist({ has_blocks: !cur.has_blocks });
    if (manual) $('#edBlocksReset', body).onclick = () => persist({ clear_blocks_override: true });
  }

  function drawChips() {
    const wrap = $('#edChips', body);
    wrap.innerHTML = '';
    let dragFrom = -1;
    tags.forEach((tg, i) => {
      const c = h(`<span class="chip" draggable="true">${esc(tg)}<button title="${esc(t('editor.remove_tag'))}">×</button></span>`);
      $('button', c).onclick = () => { tags.splice(i, 1); drawChips(); saveTags(); };
      c.ondragstart = (e) => { dragFrom = i; c.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; };
      c.ondragend = () => c.classList.remove('dragging');
      c.ondragover = (e) => { e.preventDefault(); c.classList.add('over'); };
      c.ondragleave = () => c.classList.remove('over');
      c.ondrop = (e) => {
        e.preventDefault(); c.classList.remove('over');
        if (dragFrom < 0 || dragFrom === i) return;
        const [moved] = tags.splice(dragFrom, 1);
        tags.splice(i, 0, moved); drawChips(); saveTags();
      };
      c.ondblclick = async () => {
        const v = await promptBox(t('editor.edit_tag_title'), t('editor.edit_tag_label'), tg);
        if (v !== null && v.trim()) { tags[i] = v.trim(); drawChips(); saveTags(); }
      };
      wrap.append(c);
    });
    const input = h(`<input class="chip-input" list="tagSuggest" placeholder="${esc(t('editor.add_ph'))}">`);
    input.onkeydown = (e) => {
      if ((e.key === 'Enter' || e.key === ',') && input.value.trim()) {
        e.preventDefault();
        const add = input.value.split(',').map((x) => x.trim()).filter(Boolean).filter((x) => !tags.some((tg) => tg.toLowerCase() === x.toLowerCase()));
        tags.push(...add); drawChips(); saveTags();
        $('.chip-input', body).focus();
      } else if (e.key === 'Backspace' && !input.value && tags.length) {
        tags.pop(); drawChips(); saveTags(); $('.chip-input', body).focus();
      }
    };
    wrap.append(input);
    wrap.onclick = (e) => { if (e.target === wrap) input.focus(); };
  }

  function load(i) {
    idx = i;
    cur = list[idx] && state.images.find((x) => x.id === list[idx].id);
    if (!cur) { m.close(); return; }
    tags = [...cur.tags];
    $('#edImg', body).src = cur.image_url;
    $('#edNl', body).value = cur.nl_caption || '';
    title.textContent = t('editor.title', { i: idx + 1, n: list.length });
    drawChips(); showPreview();
    $('#edPrev', body).disabled = idx === 0;
    $('#edNext', body).disabled = idx === list.length - 1;
  }

  $('#edNl', body).oninput = saveNl;
  $('#edPrev', body).onclick = () => idx > 0 && load(idx - 1);
  $('#edNext', body).onclick = () => idx < list.length - 1 && load(idx + 1);
  $('#edCopy', body).onclick = () => copyText(cur.caption);
  $('#edDelete', body).onclick = async () => {
    if (!await confirmBox(t('editor.delete_confirm'), { ok: t('common.delete'), danger: true })) return;
    await api(`/projects/${state.project.id}/images/delete`, { method: 'POST', body: { ids: [cur.id] } });
    list.splice(idx, 1);
    state.images = state.images.filter((x) => x.id !== cur.id);
    renderGallery();
    if (!list.length) m.close(); else load(Math.min(idx, list.length - 1));
  };
  $('#edRetag', body).onclick = async () => {
    if (!await confirmBox(t('editor.retag_confirm'))) return;
    startTagging({ ids: [cur.id] });
  };
  const keys = (e) => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    if ($('#modalRoot').lastElementChild !== m.el) return;
    if (e.key === 'ArrowLeft') $('#edPrev', body).click();
    if (e.key === 'ArrowRight') $('#edNext', body).click();
  };
  document.addEventListener('keydown', keys);
  editor = {
    refresh: () => {
      const fresh = state.images.find((x) => x.id === cur?.id);
      if (fresh && (fresh.caption !== cur.caption || fresh.status !== cur.status || fresh.error !== cur.error)) {
        cur = fresh; tags = [...cur.tags]; $('#edNl', body).value = cur.nl_caption || ''; drawChips(); showPreview();
      }
    },
  };
  load(idx);
}

function refreshEditor() {
  if (editor && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) editor.refresh();
}

/* ---------------------------------------------------------------- 匯出 */
async function exportDialog() {
  const s = state.project.settings;
  const prof = await api(`/profiles/${s.profile}`);
  const imgs = state.images;
  const pending = imgs.filter((i) => i.status !== 'done').length;
  const flagged = imgs.filter((i) => i.flag).length;
  const sample = imgs.find((i) => i.caption && !i.flag);
  const hints = state.project.training_hints;
  const body = h(`<div>
    <div class="field"><div class="lbl">${esc(t('export.format'))}</div><div class="choice-grid">
      <label class="choice"><input type="radio" name="exFmt" value="civitai" checked><b>Civitai</b><small class="muted">${esc(t('export.civitai_desc'))}</small></label>
      <label class="choice"><input type="radio" name="exFmt" value="kohya"><b>kohya_ss</b><small class="muted">img/&lt;repeats&gt;_&lt;trigger&gt; &lt;class&gt;/</small></label>
      <label class="choice"><input type="radio" name="exFmt" value="jsonl"><b>HF imagefolder</b><small class="muted">metadata.jsonl</small></label>
    </div></div>
    <div class="row">
      <div class="field grow"><div class="lbl">${esc(t('export.image_format'))}</div><select id="exImg"><option value="original">${esc(t('export.keep_format'))}</option><option value="png">PNG</option><option value="jpg">JPG (95)</option><option value="webp">WEBP (95)</option></select></div>
      <div class="field grow"><div class="lbl">${esc(t('export.max_side'))}</div><input type="number" id="exMax" value="0" min="0" step="64"></div>
    </div>
    <div class="row">
      <div class="field grow"><div class="lbl">${esc(t('export.naming'))}</div><select id="exNaming"><option value="original">${esc(t('export.naming_original'))}</option><option value="sequential">${esc(t('export.naming_seq'))}</option></select></div>
      <div class="field grow" id="exRepWrap"><div class="lbl">${esc(t('export.repeats'))}</div><input type="number" id="exRep" value="10" min="1"></div>
    </div>
    <label class="check"><input type="checkbox" id="exDone"> ${esc(t('export.only_done'))}</label>
    ${pending ? `<div class="alert warn" style="margin-top:10px">${esc(t('export.pending_warn', { n: pending }))}</div>` : ''}
    ${flagged ? `<div class="alert err" style="margin-top:10px">${esc(t('export.flagged_warn', { n: flagged }))}</div>` : ''}
    ${!s.trigger ? `<div class="alert warn" style="margin-top:10px">${esc(t('export.no_trigger'))}</div>` : ''}
    <div class="alert info" style="margin-top:12px">${th('export.hints_html', { shuffle: onOff(hints.shuffle_caption), keep: hints.keep_tokens })}</div>
    ${sample ? `<div class="field" style="margin-top:12px"><div class="lbl">${esc(t('export.sample', { file: sample.original_name }))}</div><div class="preview">${esc(sample.caption)}</div></div>` : ''}
    <div class="alert info" style="margin-top:12px">${th('export.civitai_howto_html', { base: prof.civitai_base })}</div>
    <div id="exResult"></div></div>`);
  const syncRep = () => { $('#exRepWrap', body).style.visibility = $('input[name=exFmt]:checked', body).value === 'kohya' ? 'visible' : 'hidden'; };
  $$('input[name=exFmt]', body).forEach((r) => { r.onchange = syncRep; }); syncRep();
  modal({
    title: t('export.title'), body, size: 'mid',
    actions: [{ label: t('common.close'), onClick: (c) => c() }, { label: t('export.build'), primary: true, onClick: async () => {
      const r = await api(`/projects/${state.project.id}/export`, { method: 'POST', body: {
        format: $('input[name=exFmt]:checked', body).value, image_format: $('#exImg', body).value,
        max_side: parseInt($('#exMax', body).value || '0', 10), repeats: parseInt($('#exRep', body).value || '10', 10),
        naming: $('#exNaming', body).value, only_done: $('#exDone', body).checked,
      } });
      const mb = (r.size_bytes / 1048576).toFixed(1);
      const link = `<a href="${esc(r.download_url)}">${esc(r.file)}</a>`;
      $('#exResult', body).innerHTML = `<div class="alert info" style="margin-top:10px">✅ ${th('export.created_html', { link: raw(link), mb })}${r.excluded.length ? esc(t('export.excluded', { n: r.excluded.length })) : ''}</div>`;
      const a = document.createElement('a'); a.href = r.download_url; a.download = r.file; document.body.append(a); a.click(); a.remove();
    } }],
  });
}

/* ================================================================ Civitai 雲端訓練 */
const CIVITAI_ACTIVE = ['unassigned', 'preparing', 'scheduled', 'processing'];

const civitaiTypeOf = (id) => (state.system.civitai?.types || []).find((x) => x.id === id);
const isVideoUrl = (u) => /\.(mp4|webm|mov)(\?|$)/i.test(u);

/** 依 Civitai 文件的公式估價：步數 × 每步單價 + epochs × 每個 epoch 費用，不低於預設配置價格的 80%。 */
function civitaiEstimate(tp, steps, epochs) {
  if (tp.per_step == null || !steps || !epochs) return null;
  return Math.max(Math.round(steps * tp.per_step + epochs * tp.per_epoch), Math.round(tp.default_price * 0.8));
}

function civitaiStatusName(st) {
  const names = {
    unassigned: t('civitai.st_unassigned'), preparing: t('civitai.st_preparing'), scheduled: t('civitai.st_scheduled'),
    processing: t('civitai.st_processing'), succeeded: t('civitai.st_succeeded'), failed: t('civitai.st_failed'),
    expired: t('civitai.st_expired'), canceled: t('civitai.st_canceled'),
  };
  return names[st] || st || '?';
}

function civitaiModerationName(m) {
  const names = { evaluating: t('civitai.mod_evaluating'), underReview: t('civitai.mod_underReview'),
    approved: t('civitai.mod_approved'), rejected: t('civitai.mod_rejected') };
  return names[m] || m || '–';
}

const civitaiRunStatus = (run) => (run.summary || {}).status || run.status;
// 已送出取消、但 Civitai 還沒停下訓練（取消是非同步的，要幾分鐘才會變成 canceled）
const civitaiCanceling = (run) => CIVITAI_ACTIVE.includes(civitaiRunStatus(run)) && !!(run.summary || {}).cancel_requested_at;
const civitaiRunLabel = (run) => (civitaiCanceling(run) ? t('civitai.st_canceling') : civitaiStatusName(civitaiRunStatus(run)));
const civitaiCancelNote = (run) => t('civitai.cancel_pending', { time: fmtClock(run.summary.cancel_requested_at * 1000) });

function fmtDuration(sec) {
  if (sec < 60) return t('civitai.dur_s', { s: Math.max(1, Math.round(sec)) });
  const m = Math.round(sec / 60);
  if (m < 60) return t('civitai.dur_m', { m });
  return t('civitai.dur_hm', { h: Math.floor(m / 60), m: m % 60 });
}

function fmtClock(ms) {
  const d = new Date(ms);
  const opts = { hour: '2-digit', minute: '2-digit' };
  return Math.abs(ms - Date.now()) > 12 * 3600e3 ? d.toLocaleString(i18n.lang, { month: 'numeric', day: 'numeric', ...opts })
    : d.toLocaleTimeString(i18n.lang, opts);
}

/** 訓練進度：百分比、剩餘時間、epoch / 步數（伺服器依 Civitai 的即時追蹤計算）。 */
function civitaiProgressHTML(run) {
  const pr = (run.summary || {}).progress;
  if (!CIVITAI_ACTIVE.includes(civitaiRunStatus(run)) || !pr) return '';
  const phases = { loading_base_model: t('civitai.ph_loading_base_model'), creating_session: t('civitai.ph_creating_session'),
    copying_previous_epoch: t('civitai.ph_copying_previous_epoch'), running: t('civitai.ph_running'), training: t('civitai.ph_running'),
    uploading: t('civitai.ph_uploading') };
  const parts = [];
  if (pr.stage === 'queued') {
    parts.push(pr.queue_ahead != null ? t('civitai.queue_ahead', { n: pr.queue_ahead }) : t('civitai.queue_waiting'));
    if (pr.start_at) parts.push(t('civitai.start_at', { time: fmtClock(Date.parse(pr.start_at)) }));
  } else {
    if (pr.phase) parts.push(phases[pr.phase] || pr.phase);
    if (pr.epochs_total) parts.push(t('civitai.prog_epochs', { done: pr.epochs_done, total: pr.epochs_total, left: pr.epochs_total - pr.epochs_done }));
    if (pr.step != null && pr.total_steps) parts.push(t('civitai.prog_steps', { step: pr.step, total: pr.total_steps, left: pr.total_steps - pr.step }));
    if (pr.seconds_per_step) parts.push(t('civitai.prog_speed', { sps: pr.seconds_per_step.toFixed(2) }));
    if (pr.elapsed_seconds != null) parts.push(t('civitai.prog_elapsed', { time: fmtDuration(pr.elapsed_seconds) }));
  }
  const pct = pr.percent;
  const remain = pr.remaining_seconds != null
    ? t('civitai.prog_remaining', { time: fmtDuration(pr.remaining_seconds), at: fmtClock(Date.now() + pr.remaining_seconds * 1000) }) : '';
  return `<div class="cv-prog">
    <div class="cv-prog-top"><b>${pct != null ? `${pct < 10 ? pct.toFixed(1) : Math.floor(pct)}%` : esc(t('civitai.prog_unknown'))}</b>
      <span class="muted" title="${esc(t('civitai.prog_remaining_hint'))}">${esc(remain)}</span></div>
    <div class="progress"><i style="width:${pct ?? 0}%"></i></div>
    ${parts.length ? `<div class="muted cv-prog-detail">${parts.map(esc).join(' · ')}</div>` : ''}
    ${pr.error ? `<div class="alert warn">${esc(pr.error)}</div>` : ''}
  </div>`;
}

function civitaiRunHTML(run) {
  const sm = run.summary || { epochs: [] };
  const st = sm.status || run.status;
  const cls = st === 'succeeded' ? 'ok' : ['failed', 'expired', 'canceled'].includes(st) ? 'err' : 'warn';
  const when = new Date(run.created_at * 1000).toLocaleString(i18n.lang);
  // 實際扣款看 Civitai 的交易紀錄（預扣＋結算追加−退款）；訓練中 cost 是 0，舊紀錄沒有交易紀錄就用送出時的費用
  const cost = sm.charged ?? (sm.cost || run.cost);
  const pm = sm.params;
  const paramsLine = [
    pm ? t('civitai.run_params', { steps: pm.steps ?? '–', epochs: pm.epochs ?? '–', lr: pm.lr ?? '–',
      dim: pm.networkDim ?? '–', alpha: pm.networkAlpha ?? '–', optimizer: pm.optimizerType ?? '–' }) : '',
    sm.priority ? t('civitai.run_priority', { p: t(`civitai.prio_${sm.priority}`) }) : '',
  ].filter(Boolean).join(' · ');
  // 已過期的任務在 Civitai 上已不存在，舊的下載連結也失效了
  const epochs = st === 'expired' ? '' : (sm.epochs || []).filter((e) => e.available && e.url).map((e) => `<div class="cv-epoch">
      <a class="btn sm" href="${esc(e.url)}" target="_blank" rel="noopener">⬇ ${esc(t('civitai.epoch', { n: e.epoch }))}</a>
      ${e.samples.map((u) => `<a href="${esc(u)}" target="_blank" rel="noopener">${isVideoUrl(u)
    ? `<video src="${esc(u)}" muted loop autoplay playsinline preload="metadata"></video>`
    : `<img src="${esc(u)}" alt="" loading="lazy">`}</a>`).join('')}
    </div>`).join('');
  return `<div class="cv-run" data-wid="${esc(run.workflow_id)}">
    <div class="cv-run-head"><span class="badge"><span class="dot ${cls}"></span>${esc(civitaiRunLabel(run))}</span>
      <span class="muted">${esc(when)} · ${esc(civitaiTypeOf(run.ecosystem)?.label || run.ecosystem)} · ${esc(t('civitai.images', { n: run.image_count }))}
      ${cost != null ? ` · ${esc(t('civitai.cost', { cost }))}` : ''}${sm.refunded ? ` · ${esc(t('civitai.refunded', { n: sm.refunded }))}` : ''} · ${esc(t('civitai.moderation'))}：${esc(civitaiModerationName(sm.moderation))}</span>
      <span class="spacer"></span>
      ${sm.request ? `<button class="btn sm" data-retrain>${esc(t('civitai.retrain'))}</button>` : ''}
      ${CIVITAI_ACTIVE.includes(st) ? `<button class="btn sm danger" data-cancel>${esc(t(civitaiCanceling(run) ? 'civitai.cancel_again' : 'civitai.cancel'))}</button>` : ''}</div>
    ${civitaiCanceling(run) ? `<div class="alert info" style="margin-top:8px">${esc(civitaiCancelNote(run))}</div>` : ''}
    ${paramsLine ? `<div class="muted cv-params">${esc(paramsLine)}</div>` : ''}
    ${st === 'canceled' && !sm.refunded ? `<div class="muted cv-params">${esc(t('civitai.cancel_no_refund'))}</div>` : ''}
    ${civitaiProgressHTML(run)}
    ${run.error ? `<div class="alert warn">${esc(run.error)}</div>` : ''}
    ${epochs ? `<div class="cv-epochs">${epochs}</div>` : ''}
  </div>`;
}

async function civitaiDialog({ focusRuns = false } = {}) {
  const s = state.project.settings;
  const cv = state.system.civitai || { configured: false, types: [], profiles: {}, optimizers: [], lr_schedulers: [] };
  const profDefault = cv.profiles[s.profile] || { type: 'sdxl', model: null, negative: '' };
  const imgs = state.images;
  const pending = imgs.filter((i) => i.status !== 'done').length;
  const flagged = imgs.filter((i) => i.flag).length;
  const nsfw = imgs.some((i) => ['questionable', 'explicit'].includes(i.rating));
  const hints = state.project.training_hints;
  // data-f：該訓練類型有這個參數才顯示（各類型的參數來自 Civitai 規格，見 GET /api/civitai 的 fields）
  const df = (f) => (f ? ` data-f="${f}"` : '');
  const num = (id, label, attrs = '', f = '') => `<div class="field"${df(f)}><div class="lbl" id="${id}Lbl">${esc(label)}</div>
    <input type="number" id="${id}" ${attrs}></div>`;
  const sel = (id, label, opts, f) => `<div class="field"${df(f)}><div class="lbl">${esc(label)}</div><select id="${id}">${
    opts.map((o) => `<option value="${esc(o)}">${esc(o)}</option>`).join('')}</select></div>`;
  const chk = (id, label, f) => `<label class="check"${df(f)}><input type="checkbox" id="${id}"> ${esc(label)}</label>`;
  const sec = (label) => `<div class="cv-sec">${esc(label)}</div>`;
  const typeOptions = [['image', t('civitai.group_image')], ['video', t('civitai.group_video')]].map(([g, label]) => `<optgroup label="${esc(label)}">${
    cv.types.filter((x) => x.group === g).map((x) => `<option value="${esc(x.id)}">${esc(x.label)}${x.preview ? ` (${esc(t('civitai.preview'))})` : ''}</option>`).join('')}</optgroup>`).join('');
  const body = h(`<div>
    ${cv.configured ? '' : `<div class="alert err" style="margin-bottom:12px">${th('civitai.not_configured_html')}</div>`}
    <p class="muted" style="margin-top:0">${esc(t('civitai.intro'))}</p>
    <div class="row" style="align-items:flex-start">
      <div class="field grow"><div class="lbl">${esc(t('civitai.ecosystem'))}</div><select id="cvType">${typeOptions}</select></div>
      <div class="field" style="flex:2" data-f="base_model"><div class="lbl">${esc(t('civitai.base_model'))}</div>
        <input type="text" id="cvModel"><div class="help">${esc(t('civitai.base_model_help'))}</div></div>
    </div>
    <div id="cvTypeNotes"></div>
    ${sec(t('civitai.sec_length'))}
    <div class="cv-grid">
      ${num('cvSteps', t('civitai.steps'), 'min="1" max="10000" step="100"', 'steps')}
      ${num('cvEpochs', t('civitai.epochs'), 'min="1" max="20"', 'epochs')}
      ${num('cvBatch', '', 'min="1" max="4"', 'batch_size')}
    </div>
    <div id="cvPrice" class="alert info" style="margin:0 0 4px"></div>
    <div id="cvViews" style="margin:6px 0 4px"></div>
    ${sec(t('civitai.sec_lr'))}
    <div class="cv-grid">
      ${num('cvLr', t('civitai.lr'), 'min="0" max="1" step="any"', 'lr')}
      ${sel('cvSched', t('civitai.lr_scheduler'), cv.lr_schedulers, 'lr_scheduler')}
      ${sel('cvOpt', t('civitai.optimizer'), cv.optimizers, 'optimizer')}
      ${num('cvSnr', t('civitai.min_snr'), 'min="0" max="20"', 'min_snr_gamma')}
      ${num('cvTeLr', t('civitai.te_lr'), 'min="0" max="1" step="any"', 'text_encoder_lr')}
    </div>
    ${chk('cvTe', t('civitai.train_te'), 'train_text_encoder')}
    ${sec(t('civitai.sec_network'))}
    <div class="cv-grid">
      ${num('cvDim', t('civitai.dim'), 'min="1" max="256"', 'network_dim')}
      ${num('cvAlpha', t('civitai.alpha'), 'min="1" max="256"', 'network_alpha')}
    </div>
    ${sec(t('civitai.sec_data'))}
    <div class="cv-grid">
      ${num('cvNoise', t('civitai.noise_offset'), 'min="0" max="1" step="any"', 'noise_offset')}
      ${num('cvKeep', t('civitai.keep_tokens'), 'min="0" max="10"', 'keep_tokens')}
    </div>
    <div class="field" data-f="trigger_word"><div class="lbl">${esc(t('civitai.trigger_word'))}</div>
      <input type="text" id="cvTrigger"><div class="help">${esc(t('civitai.trigger_word_help'))}</div></div>
    ${chk('cvShuffle', t('civitai.shuffle'), 'shuffle_tokens')}
    ${chk('cvFlip', t('civitai.flip'), 'flip_augmentation')}
    ${sec(t('civitai.sec_continue'))}
    <div class="field" data-f="continue_from"><div class="lbl">${esc(t('civitai.continue_from'))}</div>
      <input type="text" id="cvContinue" placeholder="urn:air:…:lora:civitai:…"><div class="help">${esc(t('civitai.continue_from_help'))}</div></div>
    ${sec(t('civitai.sec_samples'))}
    <div class="field"><div class="lbl">${esc(t('civitai.samples'))}</div><textarea id="cvSamples" rows="3"></textarea></div>
    <div class="field" data-f="sample_negative"><div class="lbl">${esc(t('civitai.sample_negative'))}</div><textarea id="cvNeg" rows="2"></textarea></div>
    <div class="cv-grid">
      ${num('cvCfg', t('civitai.sample_cfg'), 'min="0" max="30" step="any"', 'sample_cfg')}
      ${num('cvStrength', t('civitai.sample_strength'), 'min="0" max="2" step="0.05"', 'sample_strength')}
    </div>
    <p class="muted" id="cvCfgNote" style="font-size:12px;margin:0 0 6px"></p>
    ${sec(t('civitai.sec_upload'))}
    <div class="cv-grid">${num('cvMax', t('civitai.max_side'), 'min="512" max="4096" step="64" value="2048"')}</div>
    <div class="field"><div class="lbl">${esc(t('civitai.priority'))}</div><select id="cvPriority">${(cv.priorities || []).map((p) =>
    `<option value="${esc(p)}"${p === cv.default_priority ? ' selected' : ''}>${esc(t(`civitai.priority_opt_${p}`))}</option>`).join('')}</select>
      <div class="help">${esc(t('civitai.priority_help'))}</div></div>
    <label class="check"><input type="checkbox" id="cvDone" checked> ${esc(t('civitai.only_done'))}</label>
    <label class="check"><input type="checkbox" id="cvMature" ${nsfw ? 'checked' : ''}> ${esc(t('civitai.allow_mature'))}</label>
    <label class="check"><input type="checkbox" id="cvForceUpload"> ${esc(t('civitai.force_upload'))}</label>
    <p class="muted" style="font-size:12px;margin:0 0 6px 24px">${esc(t('civitai.force_upload_help'))}</p>
    <p class="muted" style="font-size:12px;margin:8px 0 0">${esc(t('civitai.not_in_api'))}</p>
    ${pending ? `<div class="alert warn" style="margin-top:8px">${esc(t('export.pending_warn', { n: pending }))}</div>` : ''}
    ${flagged ? `<div class="alert err" style="margin-top:8px">${esc(t('export.flagged_warn', { n: flagged }))}</div>` : ''}
    ${!s.trigger ? `<div class="alert warn" style="margin-top:8px">${esc(t('export.no_trigger'))}</div>` : ''}
    <div id="cvPrep" style="margin-top:12px"></div>
    <h3 style="margin:18px 0 8px;display:flex;align-items:center;gap:8px">${esc(t('civitai.runs'))}<span class="spacer"></span>
      <button class="btn sm" id="cvRefresh">${esc(t('civitai.refresh'))}</button></h3>
    <div id="cvRuns" class="muted">${esc(t('common.loading'))}</div>
    <p class="muted" style="font-size:12px">${esc(t('civitai.link_note'))}</p>
  </div>`);

  let timer = null;
  let prepId = null;
  const stop = () => { if (timer) clearTimeout(timer); timer = null; };

  const loadRuns = async () => {
    const runs = await api(`/projects/${state.project.id}/civitai/runs`);
    const box = $('#cvRuns', body);
    box.classList.toggle('muted', !runs.length);
    box.innerHTML = runs.length ? runs.map(civitaiRunHTML).join('') : esc(t('civitai.no_runs'));
    $$('[data-cancel]', box).forEach((b) => {
      b.onclick = async () => {
        if (!await confirmBox(t('civitai.cancel_confirm'), { danger: true })) return;
        try {
          await api(`/civitai/runs/${b.closest('.cv-run').dataset.wid}/cancel`, { method: 'POST' });
          toast(t('civitai.cancel_sent'), 'ok', 8000);
        } catch (e) { toast(e.message, 'err'); }
        loadRuns();
      };
    });
    $$('[data-retrain]', box).forEach((b) => {
      const run = runs.find((r) => r.workflow_id === b.closest('.cv-run').dataset.wid);
      b.onclick = () => fillFrom(run.summary.request, new Date(run.created_at * 1000).toLocaleString(i18n.lang));
    });
    return runs;
  };

  const renderPrep = (p) => {
    const box = $('#cvPrep', body);
    const lines = [];
    if (p.status === 'uploading') {
      const pct = p.total ? Math.round((p.done / p.total) * 100) : 0;
      lines.push(`<div class="alert info">${esc(t('civitai.uploading', { done: p.done, total: p.total, reused: p.reused }))}<div class="progress" style="margin-top:6px"><i style="width:${pct}%"></i></div></div>`);
    } else if (p.status === 'estimating') {
      lines.push(`<div class="alert info">${esc(t('civitai.estimating'))}</div>`);
    } else if (p.status === 'error') {
      lines.push(`<div class="alert err">${esc(p.error || '')}</div>`);
    } else if (['ready', 'submitting', 'submitted'].includes(p.status)) {
      lines.push(`<div class="alert ${p.insufficient_buzz ? 'err' : 'info'}"><b>${esc(t('civitai.ready', { n: p.image_count, cost: p.cost ?? '?' }))}</b>
        ${p.cost_full ? `<div>${esc(t('civitai.cost_full', { full: p.cost_full }))}</div>` : ''}
        ${p.insufficient_buzz ? `<div>${esc(t('civitai.insufficient'))}</div>` : ''}</div>`);
    }
    if (p.excluded) lines.push(`<div class="alert warn">${esc(t('civitai.excluded', { n: p.excluded }))}</div>`);
    if (p.truncated) lines.push(`<div class="alert warn">${esc(t('civitai.truncated', { n: p.truncated }))}</div>`);
    if (p.duplicates?.length) {
      lines.push(`<div class="alert warn">${esc(t('civitai.duplicates', { n: p.duplicates.length }))}<ul class="cv-blocked">${
        p.duplicates.slice(0, 20).map((d) => `<li><b>${esc(d.file)}</b> = ${esc(d.same_as)}</li>`).join('')}</ul></div>`);
    }
    if (p.blocked?.length) {
      lines.push(`<div class="alert warn">${esc(t('civitai.blocked', { n: p.blocked.length }))}<ul class="cv-blocked">${
        p.blocked.slice(0, 20).map((b) => `<li><b>${esc(b.file)}</b> — ${esc(b.reason)}</li>`).join('')}</ul></div>`);
    }
    if (p.status === 'ready') {
      lines.push(`<button class="btn primary" id="cvSubmit" ${p.insufficient_buzz ? 'disabled' : ''}>☁ ${esc(t('civitai.submit', { cost: p.cost ?? '?' }))}</button>`);
    }
    box.innerHTML = lines.join('');
    const sb = $('#cvSubmit', box);
    if (sb) {
      sb.onclick = async () => {
        const msg = t('civitai.submit_confirm', { cost: p.cost ?? '?' });
        if (!await confirmBox(p.cost_full ? `${msg} ${t('civitai.cost_full', { full: p.cost_full })}` : msg)) return;
        sb.disabled = true;
        try {
          await api(`/civitai/prepare/${p.id}/submit`, { method: 'POST' });
          toast(t('civitai.submitted'), 'ok');
          box.innerHTML = '';
          prepId = null;
          await loadRuns();
          pollRuns();
        } catch (e) { toast(e.message, 'err', 8000); sb.disabled = false; }
      };
    }
  };

  const pollPrep = async () => {
    if (!prepId) return;
    try {
      const p = await api(`/civitai/prepare/${prepId}`);
      renderPrep(p);
      if (['uploading', 'estimating'].includes(p.status)) timer = setTimeout(pollPrep, 1000);
    } catch (e) { toast(e.message, 'err'); }
  };

  const pollRuns = async () => {
    stop();
    const runs = await loadRuns().catch(() => []);
    if (focusRuns) {  // 從首頁的訓練進度點進來：直接捲到訓練紀錄
      focusRuns = false;
      $('#cvRuns', body).previousElementSibling.scrollIntoView({ block: 'start' });
    }
    const active = runs.some((r) => CIVITAI_ACTIVE.includes((r.summary || {}).status || r.status));
    if (active && !prepId) timer = setTimeout(pollRuns, 15000);
  };

  $('#cvRefresh', body).onclick = () => loadRuns().catch((e) => toast(e.message, 'err'));

  const currentType = () => civitaiTypeOf($('#cvType', body).value);
  // 訓練量：每張圖大約被看到幾次（步數 × batch ÷ 張數）。太少時衣服、配件等只出現在部分圖片的細節學不完整
  const VIEWS_TARGET = 10;
  const VIEWS_LOW = 8;
  const updateViews = () => {
    const n = imgs.filter((i) => !i.flag && (!$('#cvDone', body).checked || i.status === 'done')).length;
    const steps = parseInt($('#cvSteps', body).value, 10);
    const batch = parseInt($('#cvBatch', body).value, 10) || 1;
    const box = $('#cvViews', body);
    if (!n || !steps) { box.innerHTML = ''; return; }
    const views = (steps * batch) / n;
    const suggest = Math.min(10000, Math.ceil((n * VIEWS_TARGET) / batch / 100) * 100);
    const low = views < VIEWS_LOW && suggest > steps;
    box.innerHTML = `<div class="alert ${low ? 'warn' : 'info'}">${esc(t('civitai.views', { steps, batch, n, views: views.toFixed(1) }))}
      ${low ? `<div style="margin-top:4px">${esc(t('civitai.views_low', { suggest, target: VIEWS_TARGET }))}
        <button class="btn sm" id="cvApplySuggest" style="margin-left:6px">${esc(t('civitai.apply_suggest', { steps: suggest }))}</button></div>` : ''}</div>`;
    if (low) $('#cvApplySuggest', box).onclick = () => { $('#cvSteps', body).value = suggest; updatePrice(); };
  };
  const updatePrice = () => {
    updateViews();
    const tp = currentType();
    const est = civitaiEstimate(tp, parseInt($('#cvSteps', body).value, 10), parseInt($('#cvEpochs', body).value, 10));
    $('#cvPrice', body).textContent = est == null ? t('civitai.price_unknown')
      : t('civitai.price_est', { total: est, per_step: tp.per_step, per_epoch: tp.per_epoch });
  };
  const FIELD_NAMES = { base_model: t('civitai.f_base_model'), trigger_word: t('civitai.f_trigger_word'),
    min_snr_gamma: t('civitai.f_min_snr_gamma'), train_text_encoder: t('civitai.f_train_text_encoder') };
  const setv = (id, v) => { $(id, body).value = v ?? ''; };
  const syncDisabled = () => {
    $('#cvTeLr', body).disabled = !$('#cvTe', body).checked;
    $('#cvKeep', body).disabled = !$('#cvShuffle', body).checked;
  };
  // 選了訓練類型就把該類型的數值填進欄位，並只顯示該類型有的參數（送出的就是畫面上的數字）
  const applyType = (id) => {
    const tp = civitaiTypeOf(id);
    const d = tp.defaults;
    $$('[data-f]', body).forEach((el) => { el.hidden = !tp.fields.includes(el.dataset.f); });
    setv('#cvSteps', d.steps);
    setv('#cvEpochs', d.epochs);
    setv('#cvBatch', d.batch_size);
    $('#cvBatch', body).max = tp.max_batch;
    $('#cvBatch', body).disabled = tp.max_batch === 1;
    $('#cvBatchLbl', body).textContent = tp.max_batch === 1 ? t('civitai.batch_fixed') : t('civitai.batch_size', { max: tp.max_batch });
    setv('#cvLr', d.lr);
    setv('#cvSched', d.lr_scheduler);
    setv('#cvOpt', d.optimizer);
    setv('#cvDim', d.network_dim);
    setv('#cvAlpha', d.network_alpha);
    setv('#cvNoise', d.noise_offset);
    $('#cvFlip', body).checked = d.flip_augmentation;
    if (tp.fields.includes('min_snr_gamma')) setv('#cvSnr', d.min_snr_gamma);
    if (tp.fields.includes('train_text_encoder')) { $('#cvTe', body).checked = d.train_text_encoder; setv('#cvTeLr', d.text_encoder_lr); }
    setv('#cvCfg', d.sample_cfg);
    $('#cvCfgNote', body).textContent = d.sample_cfg == null ? t('civitai.sample_cfg_unknown') : '';
    setv('#cvStrength', d.sample_strength);
    setv('#cvModel', (id === profDefault.type ? profDefault.model : tp.model) || '');
    $('#cvModel', body).placeholder = t('civitai.builtin_model', { label: tp.label });
    const notes = [];
    if (!tp.fields.includes('base_model')) notes.push(['info', t('civitai.fixed_model', { label: tp.label })]);
    if (!tp.documented) notes.push(['warn', t('civitai.undocumented')]);
    if (tp.preview) notes.push(['warn', t('civitai.preview_note')]);
    if (tp.group === 'video') notes.push(['info', t('civitai.video_note')]);
    if (tp.ecosystem === 'anima') notes.push(['info', t('civitai.anima_lr_hint')]);
    if (s.trigger && !tp.trigger_word) notes.push(['info', t('civitai.trigger_in_caption', { trigger: s.trigger })]);
    const missing = Object.keys(FIELD_NAMES).filter((f) => !tp.fields.includes(f)).map((f) => FIELD_NAMES[f]);
    $('#cvTypeNotes', body).innerHTML = notes.map(([k, m]) => `<div class="alert ${k}" style="margin-bottom:8px">${esc(m)}</div>`).join('')
      + (missing.length ? `<p class="muted" style="font-size:12px;margin:0 0 6px">${esc(t('civitai.unsupported_fields', { fields: missing.join('、') }))}</p>` : '');
    syncDisabled();
    updatePrice();
  };
  // 跟訓練類型無關、跟專案有關的欄位只在開啟時填一次（切換類型不會蓋掉使用者改過的內容）
  $('#cvShuffle', body).checked = hints.shuffle_caption;
  setv('#cvKeep', hints.keep_tokens);
  setv('#cvTrigger', s.trigger || '');
  setv('#cvNeg', [profDefault.negative, s.block_tag_auto ? s.block_tag : ''].filter(Boolean).join(', '));
  $('#cvTe', body).onchange = syncDisabled;
  $('#cvShuffle', body).onchange = syncDisabled;
  $('#cvType', body).value = profDefault.type;
  $('#cvType', body).onchange = (e) => applyType(e.target.value);
  $('#cvSteps', body).oninput = updatePrice;
  $('#cvEpochs', body).oninput = updatePrice;
  $('#cvBatch', body).oninput = updateViews;
  $('#cvDone', body).onchange = updateViews;
  if (cv.types.length) applyType(profDefault.type);
  // 「用相同參數重新訓練」：把那次訓練的設定填回表單。標籤用目前的；優先度不算訓練參數，維持表單的選擇。
  // 一樣要按「上傳並試算費用」、確認後才會送出
  const fillFrom = (r, when) => {
    const tp = civitaiTypeOf(r.training_type);
    if (!tp) { toast(t('civitai.retrain_unknown_type', { type: r.training_type }), 'err'); return; }
    $('#cvType', body).value = tp.id;
    applyType(tp.id);
    const put = (id, v) => { if (v != null) setv(id, v); };
    const tick = (id, v) => { if (v != null) $(id, body).checked = !!v; };
    put('#cvModel', r.base_model); put('#cvSteps', r.steps); put('#cvEpochs', r.epochs); put('#cvBatch', r.batch_size);
    put('#cvLr', r.lr); put('#cvSched', r.lr_scheduler); put('#cvOpt', r.optimizer);
    put('#cvDim', r.network_dim); put('#cvAlpha', r.network_alpha); put('#cvNoise', r.noise_offset);
    put('#cvSnr', r.min_snr_gamma); put('#cvTeLr', r.text_encoder_lr); put('#cvTrigger', r.trigger_word);
    put('#cvKeep', r.keep_tokens); put('#cvContinue', r.continue_from); put('#cvNeg', r.sample_negative);
    put('#cvCfg', r.sample_cfg); put('#cvStrength', r.sample_strength); put('#cvMax', r.max_side);
    if (r.sample_prompts) setv('#cvSamples', r.sample_prompts.join('\n'));
    tick('#cvFlip', r.flip_augmentation); tick('#cvShuffle', r.shuffle_tokens); tick('#cvTe', r.train_text_encoder);
    tick('#cvMature', r.allow_mature); tick('#cvDone', r.only_done);
    syncDisabled();
    updatePrice();
    $('#cvType', body).scrollIntoView({ block: 'center' });
    toast(t('civitai.retrain_filled', { when, button: t('civitai.prepare') }), 'ok', 8000);
  };
  const val = (id) => $(id, body).value.trim();
  const intOf = (id) => (val(id) === '' ? null : parseInt(val(id), 10));
  const floatOf = (id) => (val(id) === '' ? null : parseFloat(val(id)));
  const checked = (id) => $(id, body).checked;
  const FIELD_VALUES = {
    base_model: () => val('#cvModel'), steps: () => intOf('#cvSteps'), epochs: () => intOf('#cvEpochs'),
    batch_size: () => intOf('#cvBatch'), lr: () => floatOf('#cvLr'), lr_scheduler: () => val('#cvSched'),
    optimizer: () => val('#cvOpt'), network_dim: () => intOf('#cvDim'), network_alpha: () => intOf('#cvAlpha'),
    noise_offset: () => floatOf('#cvNoise'), flip_augmentation: () => checked('#cvFlip'),
    shuffle_tokens: () => checked('#cvShuffle'), keep_tokens: () => intOf('#cvKeep'), trigger_word: () => val('#cvTrigger'),
    min_snr_gamma: () => intOf('#cvSnr'), train_text_encoder: () => checked('#cvTe'), text_encoder_lr: () => floatOf('#cvTeLr'),
    continue_from: () => val('#cvContinue'), sample_negative: () => val('#cvNeg'), sample_cfg: () => floatOf('#cvCfg'),
    sample_strength: () => floatOf('#cvStrength'),
  };
  modal({
    title: t('civitai.title'), body, size: 'mid', onClose: () => { stop(); updateCivitaiButton(); },
    actions: [{ label: t('common.close'), onClick: (c) => c() }, {
      label: t('civitai.prepare'), primary: true, onClick: async () => {
        if (!cv.configured) { toast(t('civitai.not_configured'), 'err', 8000); return; }
        stop();
        const tp = currentType();
        const req = {
          training_type: tp.id, max_side: intOf('#cvMax') || 2048, only_done: checked('#cvDone'), allow_mature: checked('#cvMature'),
          priority: val('#cvPriority') || undefined, force_upload: checked('#cvForceUpload'),
          sample_prompts: val('#cvSamples') ? val('#cvSamples').split('\n').map((x) => x.trim()).filter(Boolean).slice(0, 5) : null,
        };
        // 只送這個類型有的參數
        for (const [f, get] of Object.entries(FIELD_VALUES)) if (tp.fields.includes(f)) req[f] = get();
        const p = await api(`/projects/${state.project.id}/civitai/prepare`, { method: 'POST', body: req });
        prepId = p.id;
        renderPrep(p);
        timer = setTimeout(pollPrep, 800);
      },
    }],
  });
  pollRuns();
}

/* ================================================================ 底模指南 */
function guideHTML(p, compact = false) {
  const g = p.guide;
  const d = p.defaults;
  const lt = state.system.lora_types;
  const trainRows = Object.entries(g.training).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  const copyBlock = (code) => `<div class="copy-wrap"><pre><code>${esc(code)}</code></pre><button class="btn sm copy" data-copy>${esc(t('common.copy'))}</button></div>`;
  return `${compact ? `<h3 style="margin-bottom:6px">${esc(p.name)}</h3>` : ''}
    <p>${md(g.caption_style)}</p>
    <h2>${esc(t('guide.tag_order'))}</h2><div class="order">${g.tag_order.map((x) => `<span>${esc(x)}</span>`).join('<i>→</i>')}</div>
    <h2>${esc(t('guide.example'))}</h2>${copyBlock(g.example)}
    ${p.rating_map ? `<h2>${esc(t('guide.rating_map'))}</h2><div class="order">${Object.entries(p.rating_map).map(([k, v]) => `<span>${esc(ratingName(k))} → <code>${esc(v)}</code></span>`).join('')}</div>` : ''}
    ${g.nsfw ? `<h2>${esc(t('guide.nsfw'))}</h2><p>${md(g.nsfw)}</p><p class="muted">${md(state.system.nsfw_common || '')}</p>` : ''}
    <h2>${esc(t('guide.training'))}</h2><dl class="kv">${trainRows}<dt>${esc(t('guide.civitai_base'))}</dt><dd>${esc(p.civitai_base)}</dd></dl>
    <h2>${esc(t('guide.a1111'))}</h2>
    <div class="lbl muted" style="font-size:12.5px">Positive</div>${copyBlock(g.a1111.positive)}
    <div class="lbl muted" style="font-size:12.5px">Negative</div>${g.a1111.negative ? copyBlock(g.a1111.negative) : `<p class="muted">${esc(g.a1111.negative_note)}</p>`}
    <p class="muted">${esc(g.a1111.settings)}</p>
    ${g.notes?.length ? `<h2>${esc(t('guide.notes'))}</h2><ul>${g.notes.map((n) => `<li>${md(n)}</li>`).join('')}</ul>` : ''}
    ${compact ? '' : `<h2>${esc(t('guide.defaults'))}</h2><dl class="kv">
      <dt>${esc(t('guide.caption_mode'))}</dt><dd>${esc(state.system.caption_modes[d.caption_mode])}</dd>
      <dt>${esc(t('guide.thresholds'))}</dt><dd>${d.general_threshold} / ${d.character_threshold}</dd>
      <dt>${esc(t('guide.max_tags'))}</dt><dd>${d.max_tags}</dd>
      <dt>VLM</dt><dd>${esc(d.use_vlm ? t('guide.vlm_on', { detail: t(`settings.detail_${d.vlm_detail}`) }) : t('common.off'))}</dd>
      <dt>${esc(t('guide.rating_tags'))}</dt><dd>${esc(onOff(d.include_rating))}</dd>
      ${d.prefix_tags ? `<dt>${esc(t('guide.prefix'))}</dt><dd><code>${esc(d.prefix_tags)}</code></dd>` : ''}
    </dl>
    <h2>${esc(t('guide.by_lora_type'))}</h2>${Object.values(lt).map((x) => `<h3 style="margin:12px 0 4px">${esc(x.name)}</h3><ul>${x.tips.map((tip) => `<li>${md(tip)}</li>`).join('')}</ul>`).join('')}`}`;
}

async function renderGuide(key) {
  key = key || 'illustrious';
  const p = await api(`/profiles/${key}`);
  $('#view').innerHTML = `<section class="page">
    <div class="page-head"><h1>${esc(t('guide.title'))}</h1><span class="spacer"></span><a class="btn" href="/api/profiles/${key}/guide.md?lang=${encodeURIComponent(i18n.lang)}" target="_blank">${esc(t('guide.markdown'))}</a></div>
    <div class="guide-layout">
      <nav class="guide-list">${state.profiles.map((x) => `<a href="#/guide/${x.key}" class="${x.key === key ? 'active' : ''}">${esc(x.name)}</a>`).join('')}</nav>
      <article class="doc"><h1 style="font-size:20px">${esc(p.name)}</h1><p class="muted">${esc(p.summary)}</p>${guideHTML(p)}</article>
    </div></section>`;
  bindCopy($('#view'));
}

/* ================================================================ API / MCP */
const MCP_TOOLS = [
  'list_profiles', 'get_tagging_guide', 'quick_tag_image', 'list_projects', 'create_project', 'update_project_settings',
  'add_images_from_urls', 'import_server_folder', 'start_tagging', 'list_captions', 'update_image_caption',
  'bulk_edit_tags', 'get_tag_stats', 'export_dataset',
];

function renderApi() {
  const o = location.origin;
  const auth = state.system.auth;
  const hdr = auth ? ' \\\n  -H "Authorization: Bearer $API_KEY"' : '';
  const snip = (title, code) => `<div class="field"><div class="lbl">${esc(title)}</div><div class="copy-wrap"><pre><code>${esc(code)}</code></pre><button class="btn sm copy" data-copy>${esc(t('common.copy'))}</button></div></div>`;
  const mcpJson = JSON.stringify({ mcpServers: { 'lora-tag-studio': { type: 'http', url: `${o}/mcp`, ...(auth ? { headers: { Authorization: 'Bearer <API_KEY>' } } : {}) } } }, null, 2);
  const mcpRemote = JSON.stringify({ mcpServers: { 'lora-tag-studio': { command: 'npx', args: ['-y', 'mcp-remote', `${o}/mcp`, ...(auth ? ['--header', 'Authorization: Bearer <API_KEY>'] : [])] } } }, null, 2);
  $('#view').innerHTML = `<section class="page">
    <div class="page-head"><h1>${esc(t('api.title'))}</h1><span class="spacer"></span>
      <a class="btn" href="/docs" target="_blank">Swagger UI</a><a class="btn" href="/openapi.json" target="_blank">openapi.json</a><a class="btn" href="/llms.txt" target="_blank">llms.txt</a></div>
    <div class="alert ${auth ? 'info' : 'warn'}" style="margin-bottom:16px">${th(auth ? 'api.auth_on_html' : 'api.auth_off_html')}</div>
    <div class="card-grid">
      <div class="doc"><h2 style="margin-top:0">${esc(t('api.mcp_title'))}</h2>
        <p>${th('api.mcp_desc_html', { url: `${o}/mcp`, n: 18 })}</p>
        ${snip('Claude Code', `claude mcp add --transport http lora-tag-studio ${o}/mcp${auth ? ' \\\n  --header "Authorization: Bearer <API_KEY>"' : ''}`)}
        ${snip(t('api.mcp_http_clients'), mcpJson)}
        ${snip(t('api.mcp_claude_desktop'), mcpRemote)}
        <p class="muted">${th('api.openwebui_html', { url: `${o}/mcp` })}</p>
      </div>
      <div class="doc"><h2 style="margin-top:0">${esc(t('api.openapi_title'))}</h2>
        <p>${esc(t('api.openapi_desc'))}</p>
        ${snip(t('api.openapi_spec'), `${o}/openapi.json`)}
        ${snip(t('api.quick_tag'), `curl -X POST ${o}/api/quick-tag/json${hdr} \\\n  -H "Content-Type: application/json" \\\n  -d '{"image_url":"https://…/a.png","profile":"pony_v6","trigger":"mychar"}'`)}
        <p class="muted">${th('api.lang_html')}</p>
      </div>
    </div>
    <div class="doc" style="margin-top:16px"><h2 style="margin-top:0">${esc(t('api.flow_title'))}</h2>
      ${snip(t('api.step1'), `curl -X POST ${o}/api/projects${hdr} -H "Content-Type: application/json" \\\n  -d '{"name":"my-char","profile":"illustrious","lora_type":"character","trigger":"mychar"}'`)}
      ${snip(t('api.step2'), `curl -X POST ${o}/api/projects/<id>/upload${hdr} \\\n  -F "files=@a.png" -F "files=@dataset.zip"`)}
      ${snip(t('api.step3'), `curl -X POST ${o}/api/projects/<id>/tag${hdr} -H "Content-Type: application/json" -d '{"only_untagged":true}'\ncurl ${o}/api/jobs/<job_id>${hdr}`)}
      ${snip(t('api.step4'), `curl ${o}/api/projects/<id>/captions${hdr}\ncurl -X POST ${o}/api/projects/<id>/bulk${hdr} -H "Content-Type: application/json" \\\n  -d '{"action":"remove","tags":["watermark","*username"]}'`)}
      ${snip(t('api.step5'), `curl -OJ "${o}/api/projects/<id>/export?format=civitai"${auth ? ' -H "Authorization: Bearer $API_KEY"' : ''}`)}
    </div>
    <div class="doc" style="margin-top:16px"><h2 style="margin-top:0">${esc(t('api.tools_title'))}</h2>
      <table><tr><th>${esc(t('api.tool'))}</th><th>${esc(t('api.purpose'))}</th></tr>${MCP_TOOLS.map((n) => `<tr><td class="mono">${n}</td><td>${esc(t(`api.tools.${n}`))}</td></tr>`).join('')}</table>
    </div></section>`;
  bindCopy($('#view'));
}

/* ================================================================ 啟動 */
function initTheme() {
  const saved = store.get('theme', null);
  const theme = saved || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = theme;
  $('#themeBtn').onclick = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next; store.set('theme', next);
  };
}

(async function main() {
  initTheme();
  try {
    await loadI18n();
    await loadSystem();
  } catch (e) {
    $('#view').innerHTML = `<section class="page"><div class="alert err">${esc(t('common.server_unreachable', { error: e.message }))}</div></section>`;
    return;
  }
  window.addEventListener('hashchange', route);
  route();
  // VLM 還沒就緒（例如 vLLM 正在載入模型）時較頻繁地重新檢查，就緒後改為每分鐘
  const refreshSystem = () => {
    const v = state.system && state.system.vlm;
    const waiting = v && v.backend !== 'none' && !v.available;
    setTimeout(() => api('/system').then((s) => { state.system = s; renderSysStatus(); }).catch(() => {}).finally(refreshSystem), waiting ? 15000 : 60000);
  };
  refreshSystem();
}());
