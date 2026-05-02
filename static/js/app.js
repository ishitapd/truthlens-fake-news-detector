/* ─── app.js — TruthLens frontend logic ─── */

const DEMO_REAL = `The World Health Organization announced new guidelines following rising tensions in the global healthcare system. According to official reports, the Director General stated that expanded healthcare access would be implemented by 2025. Scientists at Johns Hopkins have published findings on vaccine efficacy in the Nature journal. Market analysts say Goldman Sachs shares rose 2.5% after strong Q3 earnings. The Senate passed legislation to regulate pharmaceutical companies following last month's summit.`;

const DEMO_FAKE = `SHOCKING: Bill Gates EXPOSED for secretly microchipping citizens — the media won't tell you! BREAKING: Scientists BANNED from revealing the truth about 5G towers. Share before it's deleted! This simple herb CURES cancer in just 7 days — Big Pharma doesn't want you to know! George Soros CONFIRMS the New World Order plan in leaked video that mainstream media is ignoring. URGENT: Forward this to everyone — the Deep State is planning population control before the election.`;

// ── Word counter ──────────────────────────────────────────
const textarea = document.getElementById('article-input');
const counter  = document.getElementById('word-counter');

textarea.addEventListener('input', () => {
  const words = textarea.value.trim().split(/\s+/).filter(Boolean).length;
  counter.textContent = `${words} word${words !== 1 ? 's' : ''}`;
});

// ── Demo loaders ──────────────────────────────────────────
function loadDemo(type) {
  textarea.value = type === 'real' ? DEMO_REAL : DEMO_FAKE;
  textarea.dispatchEvent(new Event('input'));
  textarea.focus();
}

// ── Clear ─────────────────────────────────────────────────
function clearAll() {
  textarea.value = '';
  counter.textContent = '0 words';
  document.getElementById('result-panel').style.display = 'none';
  textarea.focus();
}

// ── Status check ──────────────────────────────────────────
async function checkStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    const dot  = document.querySelector('.status-dot');
    const text = document.getElementById('status-text');

    if (data.vectorizer_loaded && data.base_models_loaded.length > 0) {
      dot.classList.add('online');
      dot.classList.remove('error');
      text.textContent = `${data.base_models_loaded.length} models ready`;
      document.getElementById('stat-models').textContent = data.base_models_loaded.length;
    } else {
      dot.classList.add('error');
      text.textContent = 'Models not loaded';
    }

    // Load results if available
    if (data.results_available) {
      loadResults();
    }
  } catch (e) {
    const dot = document.querySelector('.status-dot');
    dot.classList.add('error');
    document.getElementById('status-text').textContent = 'Server error';
  }
}

async function loadResults() {
  try {
    const res  = await fetch('/api/results');
    const data = await res.json();
    if (!data.length) return;

    // Update best accuracy stat
    const accs = data.map(r => parseFloat(r.accuracy)).filter(Boolean);
    if (accs.length) {
      document.getElementById('stat-acc').textContent = Math.max(...accs).toFixed(2);
    }

    // Show table
    document.getElementById('no-results-msg').style.display   = 'none';
    document.getElementById('perf-table-wrap').style.display  = 'block';

    // Populate table
    const tbody = document.getElementById('perf-tbody');
    if (!tbody.children.length) {
      tbody.innerHTML = '';
      data.forEach((r, i) => {
        const last = i === data.length - 1;
        const acc  = parseFloat(r.accuracy) || 0;
        const row  = document.createElement('tr');
        if (last) row.className = 'best-row';
        row.innerHTML = `
          <td class="model-name-cell">${last ? '<span class="champion-badge">★</span>' : ''}${r.model}</td>
          <td>
            <div class="bar-cell">
              <div class="bar-fill" style="width:${Math.round(acc * 100)}%"></div>
              <span>${r.accuracy}</span>
            </div>
          </td>
          <td>${r.precision}</td>
          <td>${r.recall}</td>
          <td class="f1-cell">${r.f1}</td>
          <td>${r.auc_roc && r.auc_roc !== 'None' ? r.auc_roc : '—'}</td>
        `;
        tbody.appendChild(row);
      });
    }

    // Populate bar chart
    const chart = document.getElementById('bar-chart');
    if (!chart.children.length) {
      chart.innerHTML = '';
      data.forEach((r, i) => {
        const last = i === data.length - 1;
        const f1   = parseFloat(r.f1) || 0;
        const div  = document.createElement('div');
        div.className = 'chart-row';
        div.innerHTML = `
          <span class="chart-label">${r.model}</span>
          <div class="chart-bar-track">
            <div class="chart-bar-fill ${last ? 'bar-ensemble' : ''}" style="width:0%"
                 data-w="${Math.round(f1 * 100)}%"></div>
          </div>
          <span class="chart-val">${r.f1}</span>
        `;
        chart.appendChild(div);
      });

      // Animate bars after a tick
      requestAnimationFrame(() => {
        document.querySelectorAll('.chart-bar-fill[data-w]').forEach(el => {
          el.style.width = el.dataset.w;
        });
      });
    }
  } catch (e) {
    console.error('Failed to load results:', e);
  }
}

// ── Main analyze ──────────────────────────────────────────
async function analyzeArticle() {
  const text = textarea.value.trim();
  if (!text) { textarea.focus(); return; }

  const btn     = document.getElementById('analyze-btn');
  const btnText = document.getElementById('btn-text');

  btn.disabled = true;
  btnText.innerHTML = '<span class="spinner"></span> Analyzing…';

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    renderResult(data);
  } catch (e) {
    alert('Server unreachable. Is Flask running?');
  } finally {
    btn.disabled = false;
    btnText.innerHTML = 'Analyze Article';
  }
}

// ── Render result ─────────────────────────────────────────
function renderResult(data) {
  const panel  = document.getElementById('result-panel');
  const banner = document.getElementById('verdict-banner');
  const isFake = data.final_label === 1;

  // Verdict banner
  banner.className = `verdict-banner ${isFake ? 'fake' : 'real'}`;
  document.getElementById('verdict-icon').textContent  = isFake ? '🚨' : '✅';
  document.getElementById('verdict-label').textContent = isFake ? 'FAKE NEWS' : 'REAL NEWS';
  document.getElementById('verdict-sub').textContent   =
    `Ensemble confidence: ${data.final_confidence}%  ·  ${data.word_count} words analyzed`;

  // Confidence ring
  const conf      = data.final_confidence / 100;
  const circumf   = 201; // 2π × 32
  const offset    = circumf - conf * circumf;
  const ringFill  = document.getElementById('ring-fill');
  ringFill.style.strokeDashoffset = offset;
  ringFill.style.stroke           = isFake ? '#f87171' : '#34d399';
  document.getElementById('ring-text').textContent = `${data.final_confidence}%`;
  document.getElementById('ring-text').style.color = isFake ? '#f87171' : '#34d399';

  // Stats
  const ensText = data.ensemble_label !== null
    ? `${data.ensemble_label === 1 ? 'FAKE' : 'REAL'} (${data.ensemble_confidence}%)`
    : 'N/A';
  document.getElementById('res-ensemble').textContent = ensText;
  document.getElementById('res-ensemble').style.color = data.ensemble_label === 1 ? '#f87171' : '#34d399';

  const majText = `${data.majority_vote === 1 ? 'FAKE' : 'REAL'} (${data.vote_confidence}%)`;
  document.getElementById('res-majority').textContent = majText;
  document.getElementById('res-majority').style.color = data.majority_vote === 1 ? '#f87171' : '#34d399';

  document.getElementById('res-words').textContent  = data.word_count;
  document.getElementById('res-tokens').textContent = data.cleaned_length;

  // Model grid
  const grid = document.getElementById('model-grid');
  grid.innerHTML = '';
  for (const [mName, mData] of Object.entries(data.model_predictions)) {
    const mFake = mData.label === 1;
    const card  = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <div class="model-card-name">${mName}</div>
      <div class="model-card-verdict">
        <span class="model-card-label ${mFake ? 'fake' : 'real'}">${mFake ? 'FAKE' : 'REAL'}</span>
        <span class="model-card-conf">${mData.confidence}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-bar-fill ${mFake ? 'fake' : 'real'}" style="width:${mData.confidence}%"></div>
      </div>
    `;
    grid.appendChild(card);
  }

  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Enter key ─────────────────────────────────────────────
textarea.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') analyzeArticle();
});

// ── Nav active ────────────────────────────────────────────
const sections = ['classifier', 'performance', 'pipeline'];
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const id = e.target.id;
      document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
      const nav = document.getElementById(`nav-${id === 'classifier' ? 'classify' : id === 'performance' ? 'perf' : 'pipe'}`);
      if (nav) nav.classList.add('active');
    }
  });
}, { threshold: 0.4 });

sections.forEach(id => {
  const el = document.getElementById(id);
  if (el) observer.observe(el);
});

// ── Init ──────────────────────────────────────────────────
checkStatus();
