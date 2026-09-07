/* ────────────────────────────────────────────────────────────
   AI Interview Coach — Gemini-powered chatbot
   Context bám sát bộ câu hỏi trong questions.json
   ──────────────────────────────────────────────────────────── */
(() => {
'use strict';

const KEY_API   = 'java_qa_gemini_key';   // định dạng cũ (1 key) — chỉ còn để migrate
const KEY_KEYS  = 'java_qa_gemini_keys';  // tách riêng, KHÔNG nằm trong export JSON
const KEY_MODEL = 'java_qa_gemini_model';
const KEY_CHAT  = 'java_qa_chat_history';
const KEY_MODE  = 'java_qa_ui_mode';      // 'split' (mặc định) | 'float'
const KEY_SUGG  = 'java_qa_ai_suggestions';
const DEFAULT_MODEL = 'gemini-3.5-flash';
const SUGG_TTL = 6 * 60 * 60 * 1000;      // cache gợi ý AI 6 giờ

const API = (m, method) => `https://generativelanguage.googleapis.com/v1beta/models/${m}:${method}`;

const getModel = () => localStorage.getItem(KEY_MODEL) || DEFAULT_MODEL;

/* ── Kho API key ────────────────────────────────────────
   Nhiều key xoay vòng: key nào dính 429 thì bị "nghỉ" (cd = hết nghỉ lúc nào),
   key nào sai/hỏng thì đánh dấu bad và bỏ qua hẳn cho tới khi user Test lại.
   Mỗi phần tử: { k: 'AIza…', cd?: timestamp, bad?: 'lý do' }              */
function loadKeys() {
  let arr = [];
  try { arr = JSON.parse(localStorage.getItem(KEY_KEYS)) || []; } catch {}
  if (!Array.isArray(arr)) arr = [];
  arr = arr.filter(x => x && typeof x.k === 'string' && x.k.trim());
  const legacy = (localStorage.getItem(KEY_API) || '').trim();   // migrate key đơn cũ
  if (legacy && !arr.some(x => x.k === legacy)) arr.unshift({ k: legacy });
  return arr;
}
function saveKeys(arr) {
  try {
    localStorage.setItem(KEY_KEYS, JSON.stringify(arr));
    localStorage.removeItem(KEY_API);   // đã gộp vào danh sách
  } catch {}
}
const isReady  = x => !x.bad && (!x.cd || x.cd <= Date.now());
// Mỗi key chạy model riêng; chưa chọn thì theo model mặc định trong Settings
const keyModel = x => x.m || getModel();
const hasKey   = () => loadKeys().length > 0;

function fmtWait(ms) {
  const s = Math.max(0, Math.ceil(ms / 1000));
  return s < 60 ? s + 's' : Math.floor(s / 60) + 'p' + String(s % 60).padStart(2, '0');
}

function patchKey(k, patch) {
  const arr = loadKeys();
  const e = arr.find(x => x.k === k);
  if (!e) return;
  Object.assign(e, patch);
  if (patch.cd === null) delete e.cd;
  if (patch.bad === null) delete e.bad;
  saveKeys(arr);
  paintKeys();
}

// 429 có kèm RetryInfo.retryDelay và QuotaFailure.quotaId → dùng để định thời gian nghỉ
function quotaWait(err) {
  const d = err?.error?.details || [];
  const retry = d.find(x => String(x['@type']).includes('RetryInfo'))?.retryDelay;
  const m = /^([\d.]+)s$/.exec(retry || '');
  if (m) return Math.max(20e3, Math.ceil(parseFloat(m[1]) * 1000));
  const q = d.find(x => String(x['@type']).includes('QuotaFailure'))?.violations?.[0]?.quotaId || '';
  return /PerDay/i.test(q) ? 15 * 60e3 : 60e3;   // hết quota ngày thì nghỉ lâu hơn
}
const getMode  = () => localStorage.getItem(KEY_MODE) || 'split';

// ── Utils ───────────────────────────────────────────────
const COMBINING = new RegExp('[' + String.fromCharCode(0x300) + '-' + String.fromCharCode(0x36f) + ']', 'g');
// Giữ nguyên độ dài chuỗi để prefix-match ghost text khớp vị trí ký tự
const deaccent = s => s.normalize('NFD').replace(COMBINING, '').replace(/đ/g, 'd').replace(/Đ/g, 'D').toLowerCase();
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const STOP = new Set('la gi cua va cho co the nao khi nhu duoc voi trong den mot cac nhung ra thi de bang khac nhau ve tai sao lam viec giua what how why is the a an of to in for and or'.split(' '));
const tokenize = s => deaccent(s).split(/[^a-z0-9+#.]+/).filter(t => t.length > 1 && !STOP.has(t));

const stripHtml = (() => {
  const div = document.createElement('div');
  return html => { div.innerHTML = html; return div.textContent.replace(/\n{3,}/g, '\n\n').replace(/[ \t]{2,}/g, ' ').trim(); };
})();

// ── Question index (RAG-lite) ───────────────────────────
let INDEX = [];
function buildIndex() {
  if (INDEX.length) return true;
  const data = window.QA_DATA;
  if (!data || !data.length) return false;
  INDEX = data.map(q => ({
    num: q.num, question: q.question, level: q.level, section: q.section,
    subsection: q.subsection || '', tags: q.tags || '', html: q.answerHtml,
    _q: deaccent(q.question), _meta: deaccent(`${q.tags} ${q.section} ${q.subsection || ''}`), _a: null
  }));
  return true;
}
const answerText = e => (e._a ??= stripHtml(e.html));
const byNum = n => INDEX.find(e => e.num === Number(n));

/* Token ngắn phải khớp trọn tiếng, không khớp kiểu substring:
   "so" mà substring thì trúng cả "re-so-urce" → gợi ý lạc đề.
   Regex compile 1 lần cho mỗi token rồi tái dùng cho cả 262 câu. */
const RE_ESC = /[.*+?^${}()|[\]\\]/g;
function prepTerms(query) {
  return tokenize(query).map(t => ({
    t,
    re: t.length > 3 ? null : new RegExp('(^|[^a-z0-9])' + t.replace(RE_ESC, '\\$&') + '([^a-z0-9]|$)')
  }));
}
const hit = (hay, x) => x.re ? x.re.test(hay) : hay.includes(x.t);

function search(query, limit = 6) {
  if (!buildIndex()) return [];
  const terms = prepTerms(query);
  if (!terms.length) return [];
  /* Hạ trọng số từ quá phổ biến. "hoạt động" nằm trong hàng chục tiêu đề nên
     nếu tính ngang "hashmap" thì "HashSet hoạt động…" đội lên bằng
     "HashMap vs HashTable…". Tự đo tần suất, không cần nuôi stopword list. */
  const df = terms.map(() => 0);
  const marks = [];
  for (const e of INDEX) {
    let m = null;
    terms.forEach((t, j) => {
      const inQ = hit(e._q, t);
      const inM = hit(e._meta, t);
      if (inQ || inM) { (m ||= []).push([j, inQ, inM]); if (inQ) df[j]++; }
    });
    marks.push(m);
  }
  const cap = INDEX.length * 0.08;
  let common = df.map(n => n > cap);
  if (common.every(Boolean)) common = df.map(() => false);   // mọi từ đều phổ biến → đừng hạ hết

  const scored = [];
  INDEX.forEach((e, i) => {
    const m = marks[i];
    if (!m) return;
    let s = 0;
    for (const [j, inQ, inM] of m) {
      const w = common[j] ? 0.25 : 1;
      if (inQ) s += 4 * w;
      if (inM) s += 2 * w;
    }
    if (s > 0) scored.push({ e, s });
  });
  // Chỉ quét toàn bộ đáp án khi kết quả theo tiêu đề quá ít
  if (scored.length < limit) {
    const seen = new Set(scored.map(x => x.e));
    const need = Math.max(1, terms.length - 1);
    for (const e of INDEX) {
      if (seen.has(e)) continue;
      const a = deaccent(answerText(e));
      let s = 0;
      for (const t of terms) if (hit(a, t)) s += 1;
      if (s >= need) scored.push({ e, s });
    }
  }
  return scored.sort((a, b) => b.s - a.s).slice(0, limit).map(x => x.e);
}

// ── Prompt ──────────────────────────────────────────────
function outline() {
  if (!buildIndex()) return '';
  const map = new Map();
  for (const e of INDEX) {
    if (!map.has(e.section)) map.set(e.section, new Map());
    const subs = map.get(e.section);
    const k = e.subsection || '-';
    subs.set(k, (subs.get(k) || 0) + 1);
  }
  return [...map].map(([sec, subs]) =>
    `- ${sec}: ${[...subs].map(([s, n]) => `${s} (${n})`).join(', ')}`).join('\n');
}

const SYSTEM = () => `Bạn là "AI Interview Coach" — trợ lý luyện phỏng vấn Java/Spring Boot backend cho thị trường tuyển dụng Việt Nam.

PHẠM VI (bắt buộc):
Chỉ trả lời trong phạm vi bộ câu hỏi phỏng vấn của trang này: Java Core, Spring Boot, Kafka, Redis, Microservice, Database, Cloud, DevOps, Architecture, System Design, ELK Stack, Testing — cùng các chủ đề phụ trợ trực tiếp (CV/JD backend, lộ trình ôn tập, cách trả lời phỏng vấn, kỳ vọng theo level).
Nếu câu hỏi nằm ngoài phạm vi, từ chối ngắn gọn trong 1 câu rồi gợi ý 2 câu hỏi thuộc phạm vi.

CẤU TRÚC BỘ CÂU HỎI (${INDEX.length} câu):
${outline()}

CÁCH TRẢ LỜI:
1. Ưu tiên tuyệt đối nội dung trong "NGỮ CẢNH" được cung cấp. Khi dùng, trích số câu dạng #12.
2. Nếu NGỮ CẢNH không đủ, được phép bổ sung kiến thức của bạn nhưng phải ghi rõ "(ngoài bộ câu hỏi)".
3. Bám sát yêu cầu tuyển dụng hiện tại: Java 17/21 (virtual threads, records, pattern matching), Spring Boot 3.x + Jakarta EE, GraalVM native, observability (OpenTelemetry, Micrometer), event-driven với Kafka, Kubernetes, và tích hợp AI/LLM vào backend. Khi liên quan, nêu rõ nhà tuyển dụng thường đào sâu điểm nào và bẫy thường gặp.
4. Trả lời bằng tiếng Việt; thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
5. NGẮN GỌN: mặc định dưới 200 từ, ưu tiên gạch đầu dòng. Code chỉ khi thực sự cần, tối đa khoảng 15 dòng.
6. Kết thúc bằng đúng 1 dòng: "👉 Follow-up: <câu hỏi nhà tuyển dụng sẽ hỏi tiếp>".`;

// Câu đã dùng làm ngữ cảnh cho lượt vừa rồi → không gợi ý hỏi lại chính nó
let lastCtxNums = new Set();

function buildContext(query, pinned) {
  const hits = search(query, 6);
  if (pinned && !hits.includes(pinned)) hits.unshift(pinned);
  lastCtxNums = new Set(hits.slice(0, 6).map(e => e.num));
  if (!hits.length) return '';
  return 'NGỮ CẢNH (trích từ bộ câu hỏi):\n\n' + hits.slice(0, 6).map(e =>
    `[#${e.num} | ${e.section}${e.subsection ? ' > ' + e.subsection : ''} | ${e.level} | tags: ${e.tags}]\nQ: ${e.question}\nA: ${answerText(e).slice(0, 1200)}`
  ).join('\n\n---\n\n');
}

// ── Mini markdown ───────────────────────────────────────
const PH = i => '%%CBBLOCK' + i + '%%';
function md(src) {
  const blocks = [];
  let t = esc(src).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    PH(blocks.push(`<pre class="cb-code"><code>${code.replace(/\n$/, '')}</code></pre>`) - 1));
  t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>')
       .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
       .replace(/(^|\s)\*([^*\n]+)\*/g, '$1<em>$2</em>')
       .replace(/#(\d{1,3})\b/g, '<a class="cb-ref" href="#q$1" data-q="$1">#$1</a>');
  const out = [];
  let list = null;
  for (const line of t.split('\n')) {
    const ul = line.match(/^\s*[-*•]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    const h  = line.match(/^#{1,4}\s+(.*)/);
    const tag = ul ? 'ul' : ol ? 'ol' : null;
    if (tag) {
      if (list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; }
      out.push(`<li>${(ul || ol)[1]}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    const ph = line.trim().match(/^%%CBBLOCK(\d+)%%$/);   // code block đứng riêng → không bọc <p>
    if (ph) out.push(blocks[ph[1]]);
    else if (h) out.push(`<h4>${h[1]}</h4>`);
    else if (line.trim()) out.push(`<p>${line}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join('').replace(/%%CBBLOCK(\d+)%%/g, (_, i) => blocks[i]);
}

// ── State ───────────────────────────────────────────────
let messages = [];
let busy = false, controller = null;
let suggestions = [], sugIdx = 0;   // pool đang dùng cho ghost text
let aiSuggestions = [];             // gợi ý do AI sinh theo ngữ cảnh
let pinned = null;                  // câu hỏi đang "Hỏi AI về câu này"
let chips = [];

// Fallback khi AI chưa load được
const FALLBACK_SUGG = [
  'Lộ trình ôn phỏng vấn Middle Java trong 2 tuần?',
  'Virtual Threads (Java 21) thay ThreadPool được không?',
  'Kafka đảm bảo exactly-once bằng cách nào?',
  'Cách chống N+1 query trong JPA?',
  'Thiết kế hệ thống rút gọn URL chịu 10k RPS?',
  'Nhà tuyển dụng Senior Java hiện nay đào sâu những gì?'
];
const FALLBACK_CHIPS = [
  'Giải thích đơn giản hơn',
  'Cho ví dụ code thực tế',
  'Nhà tuyển dụng sẽ đào sâu thế nào?',
  'Câu hỏi bẫy thường gặp?',
  'So sánh với giải pháp thay thế'
];

const loadHistory = () => { try { messages = JSON.parse(localStorage.getItem(KEY_CHAT)) || []; } catch { messages = []; } };
const saveHistory = () => { try { localStorage.setItem(KEY_CHAT, JSON.stringify(messages.slice(-30))); } catch {} };

// ── Gemini transport ────────────────────────────────────
// Mỗi thế hệ model nhận cấu hình "thinking" khác nhau; nếu API từ chối thì
// thử lại đúng 1 lần với body đã bỏ thinkingConfig.
function thinkingFor(model) {
  if (model.startsWith('gemini-3')) return { thinkingLevel: 'low' };
  if (model.includes('flash')) return { thinkingBudget: 0 };   // 2.5 flash / flash-lite
  return null;                                                 // 2.5 pro: không tắt được
}

// Gọi bằng ĐÚNG 1 key. Model từ chối thinkingConfig thì thử lại 1 lần đã bỏ field.
async function rawCall(k, model, method, payload, signal) {
  const url = API(model, method) + (method.startsWith('stream') ? '?alt=sse' : '');
  const send = p => fetch(url, {
    method: 'POST', signal,
    headers: { 'content-type': 'application/json', 'x-goog-api-key': k },
    body: JSON.stringify(p)
  });
  let res = await send(payload);
  if (res.status === 400 && payload.generationConfig?.thinkingConfig) {
    const { thinkingConfig, ...rest } = payload.generationConfig;
    res = await send({ ...payload, generationConfig: rest });
  }
  return res;
}

const KEY_IS_BAD = /api key not valid|api_key_invalid|api key expired|api key must be set/i;

// Thông báo đã soạn sẵn cho người dùng — explain() phải để nguyên, đừng dịch lại
function friendly(msg) { const e = new Error(msg); e.friendly = true; return e; }

/* Xoay vòng key: hết quota (429) hoặc key hỏng thì tự nhảy sang key kế tiếp.
   Lỗi không liên quan tới key (payload sai, 5xx…) thì ném ra luôn,
   đổi key cũng vô ích mà còn đốt thêm request. */
async function callGemini(method, makeBody, signal) {
  const all = loadKeys();
  if (!all.length) throw friendly('Chưa có API key nào — mở Settings để thêm.');
  const usable = all.filter(isReady);
  if (!usable.length) {
    const cds = all.filter(x => x.cd).map(x => x.cd);
    const bad = all.filter(x => x.bad).length;
    throw friendly(cds.length
      ? `Cả ${all.length} key đều hết quota — key sớm nhất mở lại sau ${fmtWait(Math.min(...cds) - Date.now())}`
      : `Cả ${bad} key đều lỗi — mở Settings bấm Test để kiểm tra lại.`);
  }

  let lastErr = null;
  for (const entry of usable) {
    // body dựng lại cho từng key vì thinkingConfig/token phụ thuộc model của key đó
    const model = keyModel(entry);
    const res = await rawCall(entry.k, model, method, makeBody(model), signal);
    if (res.ok) {
      if (entry.cd || entry.bad) patchKey(entry.k, { cd: null, bad: null });
      activeKey = entry.k;
      activeModel = model;
      if (el.cbModel) el.cbModel.textContent = modelLabel();
      paintStatusLine();
      return res;
    }
    const err = await res.json().catch(() => null);
    const msg = err?.error?.message || `HTTP ${res.status} ${res.statusText}`;
    lastErr = new Error(msg);

    if (res.status === 429) {
      patchKey(entry.k, { cd: Date.now() + quotaWait(err) });
      continue;                                   // → key kế tiếp
    }
    if (res.status === 403 || res.status === 404 ||
        (res.status === 400 && KEY_IS_BAD.test(msg))) {
      patchKey(entry.k, { bad: explain(msg, model) });
      continue;                                   // → key kế tiếp
    }
    throw lastErr;                                // lỗi không phải do key
  }
  throw lastErr || new Error('Không key nào gọi được');
}

// Gemini phát text tăng dần qua SSE; gói thành generator cho send() dễ dùng
async function* streamAnswer(req, signal) {
  const res = await callGemini('streamGenerateContent', model => {
    const think = thinkingFor(model);
    return {
      systemInstruction: { parts: [{ text: req.system }] },
      contents: [
        ...req.history.map(m => ({ role: m.role, parts: [{ text: m.text }] })),
        { role: 'user', parts: [{ text: req.prompt }] }
      ],
      generationConfig: {
        temperature: 0.6,
        maxOutputTokens: model.includes('pro') ? 4096 : 2048,
        ...(think ? { thinkingConfig: think } : {})
      }
    };
  }, signal);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    let out = '';
    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;
      try {
        const chunk = JSON.parse(payload);
        for (const p of chunk?.candidates?.[0]?.content?.parts || []) {
          if (!p.thought) out += p.text || '';
        }
      } catch {}
    }
    if (out) yield out;
  }
}

// ── Gợi ý sinh bởi AI ───────────────────────────────────
const SUGG_SCHEMA = { type: 'ARRAY', items: { type: 'STRING' }, minItems: 4, maxItems: 5 };

function parseSuggList(raw) {
  let s = String(raw).trim().replace(/^```(?:json)?\s*|\s*```$/g, '').trim();
  const a = s.indexOf('['), b = s.lastIndexOf(']');
  if (a !== -1 && b > a) s = s.slice(a, b + 1);
  const arr = JSON.parse(s);
  return Array.isArray(arr) ? arr.map(x => String(x).trim()).filter(Boolean).slice(0, 5) : [];
}

async function askForSuggestions(instruction) {
  const res = await callGemini('generateContent', model => {
    const think = thinkingFor(model);
    return {
      contents: [{ role: 'user', parts: [{ text: instruction }] }],
      generationConfig: {
        temperature: 0.9,
        maxOutputTokens: 512,
        responseMimeType: 'application/json',
        responseSchema: SUGG_SCHEMA,
        ...(think ? { thinkingConfig: think } : {})
      }
    };
  });
  const data = await res.json();
  return parseSuggList(data?.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || '[]');
}

function applySuggestions(list) {
  const seen = new Set();
  const clean = list.map(s => String(s).trim()).filter(s => {
    const k = deaccent(s);
    if (!s || seen.has(k)) return false;
    seen.add(k);
    return true;
  }).slice(0, 5);
  if (!clean.length) return;
  aiSuggestions = clean;
  sugIdx = 0;                       // pool đổi thì quay về gợi ý đầu
  if (el.cbText && !el.cbText.value.trim()) onInput();
}

/* Gợi ý tức thì sau khi model trả lời — không đợi thêm 1 vòng API.
   Nguồn 1: dòng "👉 Follow-up:" model đã tự viết ở cuối câu trả lời.
   Nguồn 2: câu trong bộ đề khớp với nội dung đáp án, TRỪ những câu vừa
            được dùng làm ngữ cảnh (hỏi lại đúng cái vừa trả lời thì vô nghĩa). */
const FOLLOWUP_RE = /👉\s*Follow-?\s*up\s*[:：]?\s*(.+)/i;

function seedFollowUps(question, answer) {
  const out = [];
  const m = answer.match(FOLLOWUP_RE);
  if (m) {
    const s = m[1].trim().replace(/^["'*\s]+|["'*\s]+$/g, '');
    if (s) out.push(s);
  }
  /* Dò bộ đề bằng câu hỏi + dòng follow-up, KHÔNG bằng cả đoạn đáp án:
     đáp án dài toàn từ phổ thông nên search sẽ khớp lung tung.
     Dòng follow-up ngắn và chỉ đúng chủ đề kế tiếp nên là truy vấn tốt nhất. */
  const probe = [out[0], question].filter(Boolean).join(' ');
  for (const e of search(probe, 10)) {
    if (out.length >= 5) break;
    if (lastCtxNums.has(e.num)) continue;
    out.push(e.question);
  }
  return out;
}

// Gợi ý "nguội": chưa có hội thoại → gợi ý mở đầu, cache 6h cho đỡ tốn quota
async function coldSuggestions() {
  try {
    const c = JSON.parse(localStorage.getItem(KEY_SUGG));
    if (c && Date.now() - c.at < SUGG_TTL && c.items?.length) { applySuggestions(c.items); return; }
  } catch {}
  const sample = INDEX.filter((_, i) => i % 17 === 0).map(e => e.question).slice(0, 12);
  const items = await askForSuggestions(`Bạn tạo gợi ý câu hỏi mở đầu cho một trang luyện phỏng vấn Java/Spring Boot backend (${INDEX.length} câu hỏi).

Các chủ đề có sẵn:
${outline()}

Ví dụ câu hỏi trong bộ đề:
${sample.map(s => '- ' + s).join('\n')}

Sinh 5 câu hỏi mà người dùng nhiều khả năng muốn hỏi trợ lý đầu tiên.
Yêu cầu: tiếng Việt, mỗi câu tối đa 70 ký tự, kết thúc bằng dấu hỏi, bám sát chủ đề trên và yêu cầu tuyển dụng backend hiện tại, không trùng ý nhau.
Trả về JSON array các string.`);
  if (items.length) {
    applySuggestions(items);
    try { localStorage.setItem(KEY_SUGG, JSON.stringify({ at: Date.now(), items })); } catch {}
  }
}

// Gợi ý "nóng": đào sâu vào ĐÚNG câu trả lời vừa đọc
async function followUpSuggestions(question, answer) {
  const items = await askForSuggestions(`Người học vừa hỏi: "${question}"

Trợ lý đã trả lời:
${answer.slice(0, 2000)}

Sinh 5 câu hỏi TIẾP THEO đào sâu vào chính câu trả lời trên: trade-off, cách triển khai thực tế, trường hợp hỏng/giới hạn, cách đo đạc, hoặc hướng nhà tuyển dụng sẽ vặn tiếp.
Yêu cầu: tiếng Việt, mỗi câu tối đa 70 ký tự, kết thúc bằng dấu hỏi, KHÔNG lặp lại câu người học vừa hỏi, không hỏi chung chung, bám phạm vi Java/Spring backend.
Trả về JSON array các string.`);
  if (items.length) applySuggestions(items);
}

async function refreshSuggestions() {
  if (!hasKey() || !buildIndex()) return;
  const last = messages[messages.length - 1];
  const prevUser = [...messages].reverse().find(m => m.role === 'user');
  try {
    if (last?.role === 'model' && prevUser) await followUpSuggestions(prevUser.text, last.text);
    else await coldSuggestions();
  } catch { /* im lặng — seed + FALLBACK_SUGG vẫn còn đó */ }
}

/* Chip cho câu đang ghim. Trước khi hỏi: dựa trên đáp án có sẵn trong bộ đề.
   Sau khi model trả lời: chuyển thành follow-up của chính câu trả lời đó. */
async function refreshChips(entry, lastAnswer) {
  if (!lastAnswer) { chips = FALLBACK_CHIPS.slice(); paintChips(); }
  if (!hasKey()) return;
  const prompt = lastAnswer
    ? `Câu hỏi phỏng vấn đang ôn: "${entry.question}".

Trợ lý vừa trả lời:
${lastAnswer.slice(0, 1400)}

Sinh 5 lời nhắc ngắn để người học bấm hỏi tiếp, đào sâu vào chính câu trả lời trên.
Yêu cầu: tiếng Việt, mỗi lời nhắc tối đa 34 ký tự, khác nhau rõ rệt, không lặp nội dung đã trả lời.
Trả về JSON array các string.`
    : `Câu hỏi phỏng vấn: "${entry.question}" (chủ đề ${entry.section}, level ${entry.level}).
Đáp án tóm tắt: ${answerText(entry).slice(0, 700)}

Sinh 5 lời nhắc ngắn mà người học sẽ bấm để hỏi thêm về đúng câu này.
Yêu cầu: tiếng Việt, mỗi lời nhắc tối đa 34 ký tự, dạng mệnh lệnh hoặc câu hỏi ngắn, khác nhau rõ rệt.
Trả về JSON array các string.`;
  try {
    const items = await askForSuggestions(prompt);
    if (items.length && pinned === entry) { chips = items; paintChips(); }
  } catch {}
}

// ── DOM ─────────────────────────────────────────────────
const el = {};
function mount() {
  const root = document.createElement('div');
  root.innerHTML = `
<button class="cb-fab" id="cbFab" aria-label="AI Interview Coach" data-tooltip="AI Coach (Ctrl+/)">
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
</button>
<aside class="cb-panel" id="cbPanel" role="complementary" aria-label="AI Interview Coach">
  <div class="cb-head">
    <span class="cb-dot"></span>
    <span class="cb-title">AI Interview Coach</span>
    <span class="cb-model" id="cbModel"></span>
    <button class="cb-icon" id="cbClear" data-tooltip="Xoá hội thoại">&#8635;</button>
    <button class="cb-icon cb-only-float" id="cbClose" data-tooltip="Đóng (Esc)">&times;</button>
  </div>
  <div class="cb-body" id="cbBody"></div>
  <div class="cb-pin" id="cbPin" hidden>
    <span class="cb-pin-num" id="cbPinNum"></span>
    <span class="cb-pin-text" id="cbPinText"></span>
    <button class="cb-pin-x" id="cbPinX" aria-label="Bỏ ghim">&times;</button>
  </div>
  <div class="cb-chips" id="cbChips" hidden></div>
  <div class="cb-input">
    <div class="cb-ta-wrap">
      <div class="cb-ghost" id="cbGhost" aria-hidden="true"></div>
      <textarea id="cbText" rows="1" spellcheck="false" aria-label="Nhập câu hỏi"></textarea>
    </div>
    <button class="cb-send" id="cbSend" aria-label="Gửi">
      <svg class="cb-ico-send" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
      <svg class="cb-ico-stop" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
    </button>
  </div>
</aside>`;
  document.body.appendChild(root);
  for (const id of ['cbFab', 'cbPanel', 'cbBody', 'cbText', 'cbGhost', 'cbSend', 'cbClear', 'cbClose',
                    'cbModel', 'cbPin', 'cbPinNum', 'cbPinText', 'cbPinX', 'cbChips'])
    el[id] = document.getElementById(id);

  el.cbFab.onclick = () => toggle();
  el.cbClose.onclick = () => toggle(false);
  el.cbClear.onclick = () => { messages = []; saveHistory(); unpin(); render(); refreshSuggestions(); };
  el.cbSend.onclick = () => busy ? abort() : send();
  el.cbPinX.onclick = unpin;
  el.cbText.addEventListener('input', onInput);
  el.cbText.addEventListener('keydown', onKeydown);
  el.cbText.addEventListener('scroll', () => { el.cbGhost.scrollTop = el.cbText.scrollTop; });
  el.cbBody.addEventListener('click', onRefClick);
  el.cbChips.addEventListener('click', e => {
    const b = e.target.closest('.cb-chip');
    if (b) { el.cbText.value = b.textContent; onInput(); send(); }
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && getMode() === 'float' && el.cbPanel.classList.contains('open')) { toggle(false); e.stopPropagation(); }
    else if (e.key === '/' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); focusChat(); }
  }, true);
}

function onRefClick(e) {
  const ref = e.target.closest('.cb-ref');
  if (!ref) return;
  e.preventDefault();
  revealCard(ref.dataset.q);
}

function revealCard(num) {
  const card = document.getElementById('q' + num);
  if (!card) return;
  if (!card.classList.contains('open')) card.querySelector('.qa-question').click();
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.add('cb-flash');
  setTimeout(() => card.classList.remove('cb-flash'), 1400);
}

// ── Mode: split (mặc định) | float ──────────────────────
function applyMode(mode) {
  document.documentElement.dataset.mode = mode;   // khớp với style pre-paint trong <head>
  document.body.classList.toggle('mode-split', mode === 'split');
  document.body.classList.toggle('mode-float', mode !== 'split');
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', mode === 'split');
    b.setAttribute('data-tooltip', mode === 'split' ? 'Chuyển sang chat nổi' : 'Chuyển sang chat cạnh trang');
  });
  if (mode === 'split' && el.cbPanel) { el.cbPanel.classList.remove('open'); el.cbFab.classList.remove('active'); }
}

window.cbToggleMode = () => {
  const next = getMode() === 'split' ? 'float' : 'split';
  localStorage.setItem(KEY_MODE, next);
  applyMode(next);
  if (next === 'split') { render(); onInput(); }
};

const isVisible = () => getMode() === 'split' || el.cbPanel.classList.contains('open');

function toggle(force) {
  if (getMode() === 'split') return;   // split thì panel luôn hiện
  const open = force ?? !el.cbPanel.classList.contains('open');
  el.cbPanel.classList.toggle('open', open);
  el.cbFab.classList.toggle('active', open);
  if (open) { render(); onInput(); el.cbText.focus(); }
}

function focusChat() {
  if (getMode() === 'float') toggle(true);
  el.cbText.focus();
}

// ── Ghim 1 câu hỏi ("Hỏi AI về câu hỏi này") ────────────
window.cbAsk = (num, event) => {
  if (event) event.stopPropagation();
  const entry = byNum(num);
  if (!entry) return;
  pinned = entry;
  el.cbPin.hidden = false;
  el.cbPinNum.textContent = '#' + entry.num;
  el.cbPinText.textContent = entry.question;
  focusChat();
  onInput();
  refreshChips(entry);
};

function unpin() {
  pinned = null;
  chips = [];
  el.cbPin.hidden = true;
  paintChips();
  onInput();
}

function paintChips() {
  const show = pinned && chips.length;
  el.cbChips.hidden = !show;
  el.cbChips.innerHTML = show
    ? chips.map(c => `<button class="cb-chip">${esc(c)}</button>`).join('')
    : '';
}

// ── Ghost suggestion trong input ────────────────────────
function pool() {
  const ai = aiSuggestions.length ? aiSuggestions : FALLBACK_SUGG;
  if (!pinned) return ai;
  // Chưa hỏi gì: ghost mời các chip của câu đang ghim.
  // Đã có câu trả lời: ghost ưu tiên follow-up (chip vốn đã hiện thành nút rồi).
  return messages.length ? [...ai, ...chips] : [...chips, ...ai];
}

function onInput() {
  const t = el.cbText;
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 132) + 'px';
  const raw = t.value;
  const v = raw.trim();

  if (!v) {
    suggestions = pool();
  } else {
    // deaccent giữ nguyên độ dài → cắt đúng vị trí ký tự để ghép ghost
    const key = deaccent(raw);
    const cand = [...pool(), ...(buildIndex() ? INDEX.map(e => e.question) : [])];
    suggestions = cand.filter(s => s.length > raw.length && deaccent(s).startsWith(key));
  }
  if (sugIdx >= suggestions.length) sugIdx = 0;
  paintGhost();
}

function ghostRest() {
  const s = suggestions[sugIdx];
  if (!s) return null;
  const raw = el.cbText.value;
  return raw ? s.slice(raw.length) : s;
}

function paintGhost() {
  const rest = ghostRest();
  const raw = el.cbText.value;
  el.cbGhost.innerHTML = rest
    ? `<span class="cb-ghost-typed">${esc(raw)}</span><span class="cb-ghost-rest">${esc(rest)}</span>` +
      `<span class="cb-ghost-tab">Tab</span>`
    : '';
  el.cbGhost.scrollTop = el.cbText.scrollTop;
}

function accept() {
  const s = suggestions[sugIdx];
  if (!s || ghostRest() == null) return;
  el.cbText.value = s;   // lấy nguyên câu gợi ý (đúng dấu) thay vì nối đuôi phần đã gõ
  suggestions = [];
  onInput();
  el.cbText.focus();
}

function onKeydown(e) {
  if (e.key === 'Tab' && ghostRest() != null) { e.preventDefault(); accept(); return; }
  if (suggestions.length > 1 && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
    e.preventDefault();
    sugIdx = (sugIdx + (e.key === 'ArrowDown' ? 1 : suggestions.length - 1)) % suggestions.length;
    paintGhost();
    return;
  }
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

// ── Render ──────────────────────────────────────────────
function render() {
  if (!messages.length) {
    const total = INDEX.length || window.QA_DATA?.length || 0;
    el.cbBody.innerHTML = hasKey()
      ? `<div class="cb-empty">
           <div class="cb-empty-icon">&#129302;</div>
           <p>Hỏi bất kỳ điều gì về <strong>${total} câu hỏi phỏng vấn</strong> trong trang.</p>
           <p class="cb-hint">Gợi ý hiện mờ ngay trong ô nhập — <kbd class="cb-kbd">Tab</kbd> chèn · <kbd class="cb-kbd">&uarr;&darr;</kbd> đổi gợi ý · <kbd class="cb-kbd">Enter</kbd> gửi</p>
           <p class="cb-hint">Bấm <span class="cb-inline-btn">&#129302;</span> trên mỗi câu hỏi để hỏi riêng câu đó.</p>
         </div>`
      : `<div class="cb-empty">
           <div class="cb-empty-icon">&#128273;</div>
           <p>Chưa có <strong>Gemini API key</strong>.</p>
           <button class="cb-link" id="cbGoSettings">Mở Settings để nhập key &rarr;</button>
           <p class="cb-hint">Lấy key miễn phí tại <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">aistudio.google.com/apikey</a></p>
         </div>`;
    const go = document.getElementById('cbGoSettings');
    if (go) go.onclick = () => { if (getMode() === 'float') toggle(false); window.toggleSettings(new Event('click')); };
    return;
  }
  el.cbBody.innerHTML = messages.map(m =>
    `<div class="cb-msg ${m.role}">${m.role === 'user' ? esc(m.text) : md(m.text)}</div>`).join('');
  el.cbBody.scrollTop = el.cbBody.scrollHeight;
}

function bubble(role) {
  const d = document.createElement('div');
  d.className = 'cb-msg ' + role;
  el.cbBody.appendChild(d);
  return d;
}

// ── Send ────────────────────────────────────────────────
function abort() { controller?.abort(); }

async function send() {
  if (busy) return;
  const text = el.cbText.value.trim();
  if (!text) return;
  if (!hasKey()) { render(); return; }

  const askedAbout = pinned;
  el.cbText.value = '';
  el.cbText.style.height = 'auto';
  suggestions = [];
  paintGhost();

  if (!messages.length) el.cbBody.innerHTML = '';
  const shown = askedAbout ? `[#${askedAbout.num}] ${text}` : text;
  messages.push({ role: 'user', text: shown });
  bubble('user').textContent = shown;

  const out = bubble('model');
  out.innerHTML = '<span class="cb-typing"><i></i><i></i><i></i></span>';
  el.cbBody.scrollTop = el.cbBody.scrollHeight;

  busy = true;
  el.cbSend.classList.add('stop');
  controller = new AbortController();

  const model = getModel();
  const ctx = buildContext(text + ' ' + (askedAbout?.question || ''), askedAbout);
  const ask = askedAbout ? `Người dùng đang hỏi VỀ CÂU #${askedAbout.num} ("${askedAbout.question}").\nCÂU HỎI: ${text}` : 'CÂU HỎI: ' + text;
  const req = {
    system: SYSTEM(),
    history: messages.slice(-9, -1),
    prompt: (ctx ? ctx + '\n\n---\n\n' : '') + ask
  };

  let acc = '';
  try {
    for await (const delta of streamAnswer(req, controller.signal)) {
      acc += delta;
      out.innerHTML = md(acc);
      el.cbBody.scrollTop = el.cbBody.scrollHeight;
    }
    if (!acc.trim()) throw new Error('Model không trả về nội dung (có thể bị chặn bởi safety filter).');
    messages.push({ role: 'model', text: acc });
    saveHistory();
    // Gợi ý bám câu trả lời vừa xong: seed hiện ngay, AI thay thế khi về tới
    applySuggestions(seedFollowUps(text, acc));
    refreshSuggestions();
    if (askedAbout) refreshChips(askedAbout, acc);
  } catch (e) {
    if (e.name === 'AbortError') {
      if (acc) { messages.push({ role: 'model', text: acc + '\n\n*(đã dừng)*' }); saveHistory(); }
      else out.innerHTML = '<span class="cb-err">Đã dừng.</span>';
    } else {
      out.innerHTML = `<span class="cb-err">&#10060; ${esc(e.friendly ? e.message : explain(e.message, model))}</span>`;
    }
  } finally {
    busy = false;
    controller = null;
    el.cbSend.classList.remove('stop');
    el.cbBody.scrollTop = el.cbBody.scrollHeight;
  }
}

// ── Danh sách key trong Settings ────────────────────────
let activeKey = '';          // key vừa gọi thành công gần nhất
let activeModel = '';        // model của key đó
let keyTicker = null;        // đếm ngược thời gian nghỉ

const maskKey = k => k.length > 12 ? k.slice(0, 6) + '…' + k.slice(-4) : k.slice(0, 4) + '…';

/* Dựng option cho dropdown từng dòng bằng cách nhân bản select mặc định trong
   index.html — danh sách model chỉ khai báo một chỗ, thêm/bớt không phải sửa 2 nơi.
   Bỏ tiền tố thứ tự ("6 · ") cho vừa bề ngang. */
function modelOptions(selected) {
  const src = document.getElementById('geminiModel');
  if (!src) return '';
  return [...src.querySelectorAll('option')].map(o =>
    `<option value="${esc(o.value)}"${o.value === selected ? ' selected' : ''}>` +
    `${esc(o.textContent.replace(/^\d+\s*·\s*/, ''))}</option>`).join('');
}

function paintKeys() {
  const box = document.getElementById('geminiKeys');
  if (!box) return;
  const keys = loadKeys();
  const firstReady = keys.find(isReady);
  box.innerHTML = keys.length
    ? keys.map((x, i) => {
        const wait = x.cd && x.cd > Date.now() ? x.cd - Date.now() : 0;
        const cls = x.bad ? 'bad' : wait ? 'wait' : (x === firstReady ? 'live' : 'idle');
        const note = x.bad ? x.bad
          : wait ? '⏳ còn ' + fmtWait(wait)
          : x.k === activeKey ? 'đang dùng'
          : x === firstReady ? 'ưu tiên' : 'dự phòng';
        return `<div class="gk-row ${cls}">
  <span class="gk-dot"></span>
  <code class="gk-id">${esc(maskKey(x.k))}</code>
  <select class="gk-model" data-i="${i}" title="Model dùng cho key này">${modelOptions(keyModel(x))}</select>
  <span class="gk-note" title="${esc(note)}">${esc(note)}</span>
  <button class="gk-btn" data-act="test" data-i="${i}">Test</button>
  <button class="gk-btn gk-x" data-act="del" data-i="${i}" aria-label="Xoá key">&times;</button>
</div>`;
      }).join('')
    : '<div class="gk-empty">Chưa có key nào — dán key vào ô trên rồi bấm Thêm</div>';
  paintStatusLine();

  // chỉ chạy đồng hồ khi có key đang nghỉ và Settings đang mở
  const needTick = keys.some(x => x.cd && x.cd > Date.now());
  const open = document.getElementById('settingsPopover')?.classList.contains('open');
  clearInterval(keyTicker);
  keyTicker = needTick && open ? setInterval(paintKeys, 1000) : null;
}

// Ưu tiên model của key vừa gọi thành công, chưa gọi lần nào thì lấy mặc định
const modelLabel = () => (activeModel || getModel()).replace('gemini-', '').replace('-preview', '');

function paintStatusLine() {
  const keys = loadKeys();
  if (!keys.length) { setStatus('', '○ Chưa có key'); return; }
  const ready = keys.filter(isReady).length;
  setStatus(ready ? 'ok' : 'err', `${ready ? '●' : '✗'} ${ready}/${keys.length} key sẵn sàng`);
}

// ── Settings hooks (gọi từ index.html) ──────────────────
window.cbSettings = {
  init() {
    const m = document.getElementById('geminiModel');
    if (m) m.value = getModel();
    paintKeys();
  },
  addKey(raw) {
    const input = document.getElementById('geminiKey');
    const text = raw ?? (input ? input.value : '');
    const arr = loadKeys();
    let added = 0;
    // cho dán nhiều key một lượt, cách nhau bằng xuống dòng / dấu phẩy / khoảng trắng
    for (const k of String(text).split(/[\s,;]+/).map(s => s.trim()).filter(Boolean)) {
      if (arr.some(x => x.k === k)) continue;
      arr.push({ k });
      added++;
    }
    if (!added) { if (input) input.value = ''; return; }
    saveKeys(arr);
    if (input) input.value = '';
    paintKeys();
    if (isVisible()) render();
    refreshSuggestions();
  },
  removeKey(i) {
    const arr = loadKeys();
    if (i < 0 || i >= arr.length) return;
    if (arr[i].k === activeKey) activeKey = '';
    arr.splice(i, 1);
    saveKeys(arr);
    paintKeys();
    if (isVisible()) render();
  },
  saveModel(v) {
    localStorage.setItem(KEY_MODEL, v);
    // Chỉ reset cờ của key đang ăn theo mặc định; key đã chọn model riêng giữ nguyên
    saveKeys(loadKeys().map(x => x.m ? x : { k: x.k }));
    activeModel = '';
    if (el.cbModel) el.cbModel.textContent = modelLabel();
    paintKeys();
  },
  setKeyModel(i, model) {
    const arr = loadKeys();
    if (!arr[i]) return;
    // model khác → quota/quyền cũ không còn đúng, xoá cờ để thử lại
    arr[i] = { k: arr[i].k, ...(model ? { m: model } : {}) };
    saveKeys(arr);
    if (arr[i].k === activeKey) { activeModel = ''; if (el.cbModel) el.cbModel.textContent = modelLabel(); }
    paintKeys();
  },
  paintStatus() { paintStatusLine(); },

  // Test 1 key (bỏ qua vòng xoay để biết chính xác key đó sống hay chết)
  async testOne(i) {
    const arr = loadKeys();
    const entry = arr[i];
    if (!entry) return;
    const model = keyModel(entry);
    const row = document.querySelectorAll('#geminiKeys .gk-row')[i];
    const note = row?.querySelector('.gk-note');
    if (row) row.className = 'gk-row busy';
    if (note) note.textContent = 'đang test…';
    const t0 = Date.now();
    try {
      const think = thinkingFor(model);
      const res = await rawCall(entry.k, model, 'generateContent', {
        contents: [{ role: 'user', parts: [{ text: 'ping' }] }],
        generationConfig: { maxOutputTokens: 16, ...(think ? { thinkingConfig: think } : {}) }
      });
      if (res.ok) {
        patchKey(entry.k, { cd: null, bad: null });
        setStatus('ok', `✓ ${maskKey(entry.k)} · ${model.replace('gemini-', '')} · ${Date.now() - t0}ms`);
        return true;
      }
      const err = await res.json().catch(() => null);
      const msg = err?.error?.message || `HTTP ${res.status}`;
      if (res.status === 429) patchKey(entry.k, { cd: Date.now() + quotaWait(err), bad: null });
      else patchKey(entry.k, { bad: explain(msg, model) });
      setStatus('err', `✗ ${maskKey(entry.k)}: ${explain(msg, model)}`);
    } catch (e) {
      patchKey(entry.k, { bad: explain(e.message, model) });
      setStatus('err', `✗ ${maskKey(entry.k)}: ${explain(e.message, model)}`);
    }
    return false;
  },

  async testKey() {
    const btn = document.getElementById('geminiTest');
    const keys = loadKeys();
    if (!keys.length) { setStatus('err', '✗ Chưa có key nào'); return; }
    if (btn) { btn.disabled = true; btn.textContent = 'Đang test…'; }
    let ok = 0;
    for (let i = 0; i < keys.length; i++) if (await this.testOne(i)) ok++;
    setStatus(ok ? 'ok' : 'err', `${ok ? '✓' : '✗'} ${ok}/${keys.length} key dùng được`);
    if (btn) { btn.disabled = false; btn.textContent = 'Test tất cả'; }
  }
};

function setStatus(kind, text) {
  const s = document.getElementById('geminiStatus');
  if (!s) return;
  s.textContent = text;
  s.className = 'gemini-status' + (kind ? ' ' + kind : '');
  s.title = text;
}

// Đổi lỗi thô của Gemini API sang câu người dùng hành động được
function explain(msg, model) {
  const m = (msg || '').toLowerCase();
  const short = model.replace('gemini-', '');
  if (m.includes('api key not valid') || m.includes('api_key_invalid') || m.includes('api key expired'))
    return 'API key không hợp lệ hoặc đã hết hạn';
  if (m.includes('permission') || m.includes('403'))
    return `Key không có quyền dùng ${short}`;
  if (m.includes('not found') || m.includes('404'))
    return `Không có model ${short} cho key này — chọn model khác`;
  if (m.includes('quota') || m.includes('resource_exhausted') || m.includes('429'))
    return 'Hết quota — đợi hoặc đổi sang model nhẹ hơn';
  if (m.includes('failed to fetch') || m.includes('networkerror'))
    return 'Không gọi được mạng (chặn CORS / offline?)';
  return (msg || 'Lỗi không rõ').slice(0, 90);
}

// ── Boot ────────────────────────────────────────────────
function wireSettings() {
  const box = document.getElementById('geminiKeys');
  if (box) {
    box.addEventListener('click', e => {
      const b = e.target.closest('.gk-btn');
      if (!b) return;
      const i = Number(b.dataset.i);
      if (b.dataset.act === 'del') window.cbSettings.removeKey(i);
      else window.cbSettings.testOne(i);
    });
    box.addEventListener('change', e => {
      const sel = e.target.closest('.gk-model');
      if (sel) window.cbSettings.setKeyModel(Number(sel.dataset.i), sel.value);
    });
  }
  const input = document.getElementById('geminiKey');
  if (input) input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); window.cbSettings.addKey(); }
  });
}

function boot() {
  mount();
  wireSettings();
  loadHistory();
  buildIndex();
  applyMode(getMode());
  el.cbModel.textContent = modelLabel();
  window.cbSettings.init();
  render();
  onInput();
  refreshSuggestions();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
window.addEventListener('qa-ready', () => {
  if (buildIndex() && isVisible()) { render(); onInput(); }
});
})();
