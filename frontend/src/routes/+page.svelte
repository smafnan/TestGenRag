<script lang="ts">
  import { onMount } from 'svelte';

  const API: string =
    (import.meta.env.VITE_API_BASE as string | undefined) ??
    (import.meta.env.DEV ? 'http://localhost:8000' : '');

  const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

  async function safeError(res: Response): Promise<string> {
    const raw = await res.text().catch(() => '');
    try {
      const body = JSON.parse(raw);
      return body.detail ?? body.message ?? `Request failed (${res.status})`;
    } catch {
      return raw.slice(0, 200) || `Request failed (${res.status})`;
    }
  }

  interface TestStep { action: string; expected_result: string; }
  interface TestCase {
    title: string; priority: string; preconditions: string;
    steps: TestStep[]; source_citations: string[]; signed?: boolean;
  }
  interface Doc { name: string; chunks: number; s3: string | null; }
  interface Chunk { source: string; page: number | string; page_type: string; snippet: string; }
  interface Trace {
    hyde: string; queries: string[]; chunks: Chunk[];
    rerank_method: string; extractor: string; model: string; elapsed_ms: number;
  }
  interface Finding { observation: string; evidence: string; citation: string; }
  interface AnalysisReport {
    summary: string; findings: Finding[]; assessment: string;
    recommendation: string; caveats: string;
  }
  interface Relevance { relevant: boolean; document_topic: string; reason: string; }

  let apiKey = $state('');
  let keyVisible = $state(false);
  let selectedModel = $state('meta/llama-3.1-8b-instruct');
  let customModel = $state('');

  let file = $state<File | null>(null);
  let docs: Doc[] = $state([]);
  let requirement = $state('');
  let testCases: TestCase[] = $state([]);
  let loading = $state(false);
  let uploading = $state(false);
  let grounded = $state<boolean | null>(null);
  let attempts = $state(0);
  let cached = $state(false);
  let error = $state('');
  let trace = $state<Trace | null>(null);

  type StageState = 'pending' | 'running' | 'done';
  let stageState = $state<Record<string, StageState>>({
    hyde: 'pending', retrieve: 'pending', draft: 'pending', judge: 'pending',
  });
  let openStage = $state<string | null>(null);

  let analyzing = $state(false);
  let analysisError = $state('');
  let relevance = $state<Relevance | null>(null);
  let report = $state<AnalysisReport | null>(null);
  let analysisGrounded = $state<boolean | null>(null);
  let analysisTrace = $state<Trace | null>(null);
  let analysisApproved = $state(false);

  const STAGES = [
    { key: 'hyde',     label: 'HyDE',     desc: 'Rewrite as spec'  },
    { key: 'retrieve', label: 'Retrieve', desc: 'Search + re-rank' },
    { key: 'draft',    label: 'Draft',    desc: 'Write cases'      },
    { key: 'judge',    label: 'Judge',    desc: 'Verify grounding' },
  ];

  const NIM_MODELS = [
    { value: 'meta/llama-3.1-8b-instruct',             label: 'Llama 3.1 8B - fast, recommended'  },
    { value: 'meta/llama-3.1-70b-instruct',            label: 'Llama 3.1 70B - stronger'           },
    { value: 'meta/llama-3.3-70b-instruct',            label: 'Llama 3.3 70B - newest Meta'        },
    { value: 'nvidia/llama-3.1-nemotron-70b-instruct', label: 'Nemotron 70B - NVIDIA tuned'        },
    { value: 'mistralai/mistral-7b-instruct-v0.3',     label: 'Mistral 7B - lightweight'           },
    { value: 'mistralai/mixtral-8x7b-instruct-v0.1',   label: 'Mixtral 8x7B - mixture of experts'  },
    { value: 'microsoft/phi-3-mini-128k-instruct',     label: 'Phi-3 Mini - small but capable'     },
    { value: 'custom',                                 label: 'Custom - enter any model ID'        },
  ];

  $effect(() => { if (selectedModel !== 'custom') customModel = ''; });

  let signedCount = $derived(testCases.filter((t) => t.signed).length);
  let sourceList = $derived(
    trace && trace.chunks.length
      ? [...new Set(trace.chunks.map((c) => `${c.source} p.${c.page}`))].join(', ')
      : '-'
  );

  // SVG pipeline electron animation
  const NODE_X = [60, 360, 540, 840];
  let electronX = $state(60);
  let electronVisible = $state(false);
  let electronAnimId: number | null = null;
  let prevRunningIdx = -1;

  const doneCount = $derived(Object.values(stageState).filter((v) => v === 'done').length);
  const fillOffset = $derived(1200 - (1200 * doneCount) / 4);

  function animateElectron(fromX: number, toX: number, duration: number) {
    if (electronAnimId) cancelAnimationFrame(electronAnimId);
    electronVisible = true;
    let start: number | null = null;
    function step(ts: number) {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const ease = p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p;
      electronX = fromX + (toX - fromX) * ease;
      if (p < 1) { electronAnimId = requestAnimationFrame(step); }
      else { electronX = toX; electronAnimId = null; }
    }
    electronAnimId = requestAnimationFrame(step);
  }

  $effect(() => {
    const runningIdx = STAGES.findIndex((s) => stageState[s.key] === 'running');
    if (runningIdx !== -1 && runningIdx !== prevRunningIdx) {
      animateElectron(NODE_X[Math.max(runningIdx - 1, 0)], NODE_X[runningIdx], 600);
      prevRunningIdx = runningIdx;
    }
    if (runningIdx === -1 && Object.values(stageState).every((v) => v === 'pending')) {
      prevRunningIdx = -1;
      electronVisible = false;
      electronX = 60;
    }
  });

  async function refreshDocs(): Promise<void> {
    try {
      const res = await fetch(`${API}/documents`);
      docs = (await res.json()) as Doc[];
    } catch { /* backend not ready */ }
  }

  async function upload(): Promise<void> {
    if (!file) return;
    uploading = true; error = '';
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`${API}/ingest`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await safeError(res));
      await refreshDocs();
      file = null;
    } catch (e) {
      error = `Upload failed: ${(e as Error).message}`;
    } finally {
      uploading = false;
    }
  }

  function normalise(tc: Partial<TestCase>): TestCase {
    return {
      title: tc.title ?? 'Untitled test case',
      priority: tc.priority ?? 'Medium',
      preconditions: tc.preconditions ?? '-',
      steps: Array.isArray(tc.steps) ? tc.steps : [],
      source_citations: Array.isArray(tc.source_citations) ? tc.source_citations : [],
      signed: false,
    };
  }

  function resetStages() {
    stageState = { hyde: 'pending', retrieve: 'pending', draft: 'pending', judge: 'pending' };
    openStage = null;
  }
  async function runStageAnimation() {
    const order = ['hyde', 'retrieve', 'draft', 'judge'];
    for (let i = 0; i < order.length; i++) {
      if (!loading) return;
      stageState[order[i]] = 'running';
      stageState = { ...stageState };
      await sleep(750);
      if (loading && i < order.length - 1) {
        stageState[order[i]] = 'done';
        stageState = { ...stageState };
      }
    }
  }
  function finishStages() {
    stageState = { hyde: 'done', retrieve: 'done', draft: 'done', judge: 'done' };
    electronVisible = false;
  }
  function failStages() {
    for (const k of Object.keys(stageState)) if (stageState[k] === 'running') stageState[k] = 'pending';
    stageState = { ...stageState };
    electronVisible = false;
  }
  function toggleStage(k: string) {
    if (stageState[k] !== 'done') return;
    openStage = openStage === k ? null : k;
  }

  async function generate(): Promise<void> {
    if (!requirement.trim()) return;
    if (!apiKey.trim()) { error = 'Please enter your NVIDIA API key above before generating.'; return; }
    loading = true; error = ''; testCases = []; grounded = null; trace = null;
    resetStages();
    runStageAnimation();
    try {
      const res = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Api-Key': apiKey.trim() },
        body: JSON.stringify({ requirement, model: selectedModel === 'custom' ? customModel.trim() : selectedModel }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      const data = await res.json();
      trace = (data.trace ?? null) as Trace | null;
      grounded = data.grounded ?? null;
      attempts = data.attempts ?? 0;
      cached = Boolean(data.cached);
      testCases = (Array.isArray(data.test_cases) ? data.test_cases : []).map(normalise);
      finishStages();
      openStage = 'judge';
    } catch (e) {
      error = (e as Error).message;
      failStages();
    } finally {
      loading = false;
    }
  }

  async function sign(i: number): Promise<void> {
    const tc = testCases[i];
    try {
      const res = await fetch(`${API}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requirement,
          test_case: {
            title: tc.title, priority: tc.priority, preconditions: tc.preconditions,
            steps: tc.steps, source_citations: tc.source_citations,
          },
          signed_by: 'reviewer',
        }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      testCases[i].signed = true;
      testCases = [...testCases];
    } catch (e) {
      error = `Approve failed: ${(e as Error).message}`;
    }
  }

  async function analyze(): Promise<void> {
    if (!requirement.trim()) return;
    if (!apiKey.trim()) { analysisError = 'Please enter your NVIDIA API key above first.'; return; }
    analyzing = true; analysisError = '';
    relevance = null; report = null; analysisGrounded = null; analysisTrace = null; analysisApproved = false;
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Api-Key': apiKey.trim() },
        body: JSON.stringify({ requirement, model: selectedModel === 'custom' ? customModel.trim() : selectedModel }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      const d = await res.json();
      relevance = { relevant: d.relevant, document_topic: d.document_topic, reason: d.reason };
      analysisTrace = (d.trace ?? null) as Trace | null;
      if (d.relevant && d.report) { report = d.report as AnalysisReport; analysisGrounded = d.grounded ?? null; }
    } catch (e) {
      analysisError = (e as Error).message;
    } finally {
      analyzing = false;
    }
  }

  function approveAnalysis() { analysisApproved = true; }

  function downloadFile(name: string, content: string, type: string) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  }

  function exportCases(): TestCase[] {
    const signed = testCases.filter((t) => t.signed);
    return signed.length ? signed : testCases;
  }

  function exportAnalysisMarkdown() {
    if (!report) return;
    let md = `# Analysis Report\n\n**Request:** ${requirement}\n\n`;
    md += `**Model:** ${analysisTrace?.model ?? '-'}  \n**Grounded:** ${analysisGrounded ? 'Yes' : 'No'}\n\n---\n\n`;
    md += `## Summary\n\n${report.summary}\n\n`;
    if (report.findings.length) {
      md += `## Findings\n\n`;
      report.findings.forEach((f, i) => {
        md += `${i + 1}. **${f.observation}**\n`;
        if (f.evidence) md += `   - Evidence: ${f.evidence}\n`;
        if (f.citation) md += `   - Source: ${f.citation}\n`;
      });
      md += `\n`;
    }
    md += `## Assessment\n\n${report.assessment}\n\n`;
    if (report.recommendation) md += `## Recommendation\n\n${report.recommendation}\n\n`;
    if (report.caveats) md += `## Caveats\n\n${report.caveats}\n`;
    downloadFile('analysis-report.md', md, 'text/markdown');
  }

  function exportMarkdown() {
    const cases = exportCases();
    let md = `# Test Plan\n\n**Requirement:** ${requirement}\n\n`;
    md += `**Model:** ${trace?.model ?? '-'}  \n**Grounded:** ${grounded ? 'Yes' : 'No'} (after ${attempts} attempt${attempts !== 1 ? 's' : ''})  \n`;
    md += `**Sources used:** ${sourceList}\n\n---\n\n`;
    cases.forEach((tc, i) => {
      md += `## ${i + 1}. ${tc.title}  _(Priority: ${tc.priority})_\n\n**Preconditions:** ${tc.preconditions}\n\n`;
      md += `| # | Action | Expected result |\n|---|--------|------------------|\n`;
      tc.steps.forEach((s, j) => { md += `| ${j + 1} | ${s.action} | ${s.expected_result} |\n`; });
      md += `\n**Traceability:** ${tc.source_citations.join(', ')}\n\n`;
    });
    downloadFile('test-plan.md', md, 'text/markdown');
  }

  function exportCSV() {
    const cases = exportCases();
    const rows: string[][] = [['Test Case','Priority','Preconditions','Step','Action','Expected Result','Citations']];
    cases.forEach((tc) => tc.steps.forEach((s, j) => rows.push([
      tc.title, tc.priority, tc.preconditions, String(j + 1),
      s.action, s.expected_result, tc.source_citations.join('; '),
    ])));
    downloadFile('test-plan.csv',
      rows.map((r) => r.map((cell) => `"${(cell ?? '').replace(/"/g, '""')}"`).join(',')).join('\n'),
      'text/csv');
  }

  onMount(() => {
    refreshDocs();

    // Custom cursor
    const cdot = document.getElementById('cdot');
    const cring = document.getElementById('cring');
    let mx = 0, my = 0, rx = 0, ry = 0;
    if (window.matchMedia('(pointer:fine)').matches) {
      document.addEventListener('mousemove', (e) => {
        mx = e.clientX; my = e.clientY;
        if (cdot) { cdot.style.left = mx + 'px'; cdot.style.top = my + 'px'; }
        document.documentElement.style.setProperty('--mx', mx + 'px');
        document.documentElement.style.setProperty('--my', my + 'px');
      });
      (function loop() {
        rx += (mx - rx) * 0.1; ry += (my - ry) * 0.1;
        if (cring) { cring.style.left = rx + 'px'; cring.style.top = ry + 'px'; }
        requestAnimationFrame(loop);
      })();
      document.addEventListener('mouseover', (e) => {
        if ((e.target as Element).closest('button,a,label,select,input,textarea'))
          cring?.classList.add('hover');
      });
      document.addEventListener('mouseout', () => cring?.classList.remove('hover'));
      document.addEventListener('mousedown', () => { cring?.classList.remove('hover'); cring?.classList.add('click'); });
      document.addEventListener('mouseup', () => cring?.classList.remove('click'));
    } else {
      if (cdot) cdot.style.display = 'none';
      if (cring) cring.style.display = 'none';
    }

    // Button ripple
    document.addEventListener('click', (e) => {
      const b = (e.target as Element).closest('.btn') as HTMLButtonElement | null;
      if (!b || b.disabled) return;
      const r = b.getBoundingClientRect();
      const rip = document.createElement('span');
      rip.className = 'rip';
      const s = Math.max(r.width, r.height);
      rip.style.cssText = `width:${s}px;height:${s}px;left:${e.clientX - r.left - s / 2}px;top:${e.clientY - r.top - s / 2}px`;
      b.appendChild(rip);
      rip.addEventListener('animationend', () => rip.remove());
    });

    // Header scramble
    const stTag = document.querySelector('.st-tag-txt') as HTMLElement | null;
    if (stTag) {
      const final = 'online';
      const ch = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
      let f = 0; const tot = Math.ceil(1000 / 38);
      const iv = setInterval(() => {
        stTag.textContent = [...final].map((c, i) => {
          if (c === ' ') return ' ';
          if (f >= tot * (i / final.length + 0.5)) return c;
          return ch[Math.floor(Math.random() * ch.length)];
        }).join('');
        if (++f >= tot) { stTag.textContent = final; clearInterval(iv); }
      }, 38);
    }
  });
</script>

<svelte:head><title>TestGenRAG</title></svelte:head>

<div id="cdot"></div>
<div id="cring"></div>
<div class="mouse-glow"></div>
<div class="grain"></div>

<div class="app">

  <!-- HEADER -->
  <header>
    <div class="wrap hd">
      <a class="logo" href="#">TestGen<span class="logo-sep">/</span><span class="logo-rag">RAG</span></a>
      <div class="hd-right">
        <div class="pipe-dots-h">
          {#each STAGES as s}
            <div class="pdh {stageState[s.key] === 'running' ? 'on' : stageState[s.key] === 'done' ? 'done' : ''}"></div>
          {/each}
        </div>
        <span class="st-tag"><span class="st-dot"></span><span class="st-tag-txt">online</span></span>
      </div>
    </div>
  </header>

  <!-- CONFIG BAR -->
  <div class="cfg wrap" id="cfg">
    <div class="cfg-row">
      <div class="cfg-col grow">
        <span class="cfg-lbl">NVIDIA API key</span>
        <div class="fwrap">
          <input class="f mono pr" type={keyVisible ? 'text' : 'password'}
                 placeholder="nvapi-..." autocomplete="off" bind:value={apiKey} spellcheck="false" />
          <div class="fa">
            <button class="btn btn-ghost" onclick={() => (keyVisible = !keyVisible)}>
              <span>{keyVisible ? 'Hide' : 'Show'}</span>
            </button>
          </div>
        </div>
      </div>
      <div class="cfg-col grow">
        <span class="cfg-lbl">Model</span>
        <select class="f" bind:value={selectedModel}>
          {#each NIM_MODELS as m}<option value={m.value}>{m.label}</option>{/each}
        </select>
      </div>
      {#if selectedModel === 'custom'}
        <div class="cfg-col grow">
          <span class="cfg-lbl">Custom model ID</span>
          <input class="f mono" type="text" placeholder="e.g. deepseek-ai/deepseek-v3"
                 bind:value={customModel} spellcheck="false" />
        </div>
      {/if}
      {#if apiKey.trim()}<span class="ok-tag">key entered</span>{/if}
    </div>
  </div>

  <main>

    <!-- STEP 01: INGEST -->
    <section class="sec wrap" id="sec-ingest">
      <span class="sec-bg">01</span>
      <div class="sec-inner">
        <div class="sec-eyebrow"><span class="sec-num">Step 01</span><span class="sec-rule"></span></div>
        <h2 class="sec-title">Ingest</h2>
        <p class="sec-hint">Upload a PDF. It is extracted, chunked (~1,000-char with overlap), embedded with MiniLM-L6-v2, and indexed in FAISS locally.</p>
        <label class="upload">
          <input type="file" accept=".pdf"
                 onchange={(e) => (file = (e.currentTarget as HTMLInputElement).files?.[0] ?? null)} />
          <span class="upl-ico">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 17V5M12 5L7 10M12 5L17 10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M3 17v2a2 2 0 002 2h14a2 2 0 002-2v-2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
            </svg>
          </span>
          <div class="upl-txt">
            <div class="upl-primary">{file ? file.name : 'Choose or drop a PDF'}</div>
            <div class="upl-sub">Supports multi-column layouts, tables, embedded figures</div>
          </div>
          <button class="btn btn-ghost" disabled={!file || uploading}
                  onclick={(e) => { e.preventDefault(); upload(); }}>
            <span>{uploading ? 'Indexing...' : 'Upload & index'}</span>
          </button>
        </label>
        {#if docs.length > 0}
          <div class="doc-list">
            {#each docs as doc}
              <div class="doc-item">
                <span class="doc-ico">
                  <svg width="13" height="14" viewBox="0 0 13 14" fill="none">
                    <rect x="1" y=".5" width="11" height="13" rx="1.2" stroke="currentColor" stroke-width="1.1"/>
                    <path d="M3.5 5h6M3.5 7.5h6M3.5 10h4" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>
                  </svg>
                </span>
                <span class="doc-name">{doc.name}</span>
                <span class="doc-ch">{doc.chunks} chunks{doc.s3 ? ' · S3' : ''}</span>
              </div>
            {/each}
          </div>
        {:else}
          <p class="empty">No documents indexed yet.</p>
        {/if}
      </div>
    </section>

    <!-- STEP 02: REQUIREMENT -->
    <section class="sec wrap" id="sec-req">
      <span class="sec-bg">02</span>
      <div class="sec-inner">
        <div class="sec-eyebrow"><span class="sec-num">Step 02</span><span class="sec-rule"></span></div>
        <h2 class="sec-title">Requirement</h2>
        <p class="sec-hint">The agent runs HyDE, then query selection, FAISS retrieval with re-ranking, draft, and LLM-as-judge with an automatic retry loop if grounding fails.</p>
        <textarea class="f" rows="4"
          placeholder="e.g. The system must trigger an audible alarm when a monitored value drops below the configured threshold."
          bind:value={requirement}></textarea>
        <div class="actions">
          <button class="btn btn-accent btn-xl" onclick={generate}
                  disabled={loading || analyzing || !requirement.trim()}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <polygon points="1.5,1 11,6 1.5,11" fill="currentColor"/>
            </svg>
            <span>{loading ? 'Running agent...' : 'Generate test cases'}</span>
          </button>
          <button class="btn btn-solid btn-xl" onclick={analyze}
                  disabled={analyzing || loading || !requirement.trim()}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="4.5" stroke="currentColor" stroke-width="1.2"/>
              <path d="M6 4v3M6 8.5v.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            </svg>
            <span>{analyzing ? 'Analyzing...' : 'Analyze & answer'}</span>
          </button>
          {#if grounded !== null}
            <span class="vrd">
              <span class="vrd-dot"></span>
              <span>{grounded ? 'Grounded' : 'Unverified'} · {attempts} attempt{attempts !== 1 ? 's' : ''}{cached ? ' · cached' : ''}</span>
            </span>
          {/if}
        </div>
        <p class="req-note">Generate writes structured test cases. Analyze reasons over the document to answer your request directly.</p>
      </div>
    </section>

    <!-- ERRORS -->
    {#if error}<div class="wrap"><div class="err">{error}</div></div>{/if}
    {#if analysisError}<div class="wrap"><div class="err">{analysisError}</div></div>{/if}

    <!-- PIPELINE -->
    {#if loading || trace}
      <section class="pipe-sec" id="pipe-sec">
        <div class="wrap">
          <div class="pipe-hd">
            <span class="pipe-hd-title">
              Agent pipeline
              {#if loading}<span class="live-pill">LIVE</span>{/if}
            </span>
            {#if trace}<span class="pipe-timing">{(trace.elapsed_ms / 1000).toFixed(1)} s · {trace.model}</span>{/if}
          </div>
          <div class="pipe-svg-wrap">
            <svg class="pipe-svg" viewBox="0 0 900 60" preserveAspectRatio="xMidYMid meet">
              <line class="p-base" x1="60" y1="30" x2="840" y2="30"/>
              <line class="p-fill" x1="60" y1="30" x2="840" y2="30"
                    style="stroke-dashoffset: {fillOffset}"/>
              {#if electronVisible}
                <circle class="p-electron" cx={electronX} cy="30" r="4.5"/>
              {/if}
            </svg>
          </div>
          <div class="pipe-nodes">
            {#each STAGES as s}
              <button class="pn {stageState[s.key]} {openStage === s.key ? 'open' : ''}"
                      onclick={() => toggleStage(s.key)}>
                <div class="pn-disc"><div class="pn-dot"></div></div>
                <div class="pn-lbl">{s.label}</div>
                <div class="pn-dsc">{s.desc}</div>
              </button>
            {/each}
          </div>
          {#if trace}<p class="pipe-hint">Click a completed stage to inspect what the agent did.</p>{/if}

          {#if openStage && trace}
            <div class="pipe-det">
              {#if openStage === 'hyde'}
                <p class="dlbl">Hypothetical spec used for embedding search</p>
                <p class="mono-blk">{trace.hyde}</p>
              {:else if openStage === 'retrieve'}
                <p class="dlbl">Search queries ({trace.queries.length}) · re-rank: {trace.rerank_method}</p>
                <div class="chips">{#each trace.queries as q}<code class="chip">{q.length > 80 ? q.slice(0, 80) + '...' : q}</code>{/each}</div>
                <p class="dlbl" style="margin-top:.8rem">Retrieved passages ({trace.chunks.length}) · extractor: {trace.extractor}</p>
                {#each trace.chunks as c}
                  <div class="chunk">
                    <div class="chunk-src">{c.source} · p.{c.page} · {c.page_type}</div>
                    <div class="chunk-txt">{c.snippet}...</div>
                  </div>
                {/each}
              {:else if openStage === 'draft'}
                <p class="dlbl">Drafted {testCases.length} test case{testCases.length !== 1 ? 's' : ''}</p>
                <ul style="padding-left:1.1rem;display:flex;flex-direction:column;gap:.25rem">
                  {#each testCases as tc}<li style="font-size:.81rem;color:var(--soft)">{tc.title}</li>{/each}
                </ul>
              {:else if openStage === 'judge'}
                <p class="dlbl">Grounding verdict (LLM-as-judge)</p>
                <div class="mono-blk">{grounded ? 'Every claim traced to a cited source.' : 'Could not fully verify grounding.'} · {attempts} attempt{attempts !== 1 ? 's' : ''}</div>
              {/if}
            </div>
          {/if}
        </div>
      </section>
    {/if}

    <!-- ANALYSIS -->
    {#if analyzing || relevance}
      <section class="ana-sec wrap" id="ana-sec">
        <div class="sec-eyebrow" style="margin-bottom:.65rem"><span class="sec-num">Analysis</span><span class="sec-rule"></span></div>
        <h2 class="sec-title" style="margin-bottom:1rem">
          Document analysis
          {#if analysisTrace}<span class="pipe-timing" style="font-size:.78rem;font-family:var(--mono);margin-left:.75rem">{(analysisTrace.elapsed_ms / 1000).toFixed(1)} s · {analysisTrace.model}</span>{/if}
        </h2>
        {#if analyzing}
          <div class="spin-row">
            <svg class="spn" width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 1a6 6 0 016 6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
            </svg>
            Checking relevance, then analyzing...
          </div>
        {:else if relevance && !relevance.relevant}
          <div class="guardrail">
            <div style="font-weight:600;color:var(--fog);margin-bottom:.4rem">Document does not match your request</div>
            <p style="font-size:.82rem;color:var(--soft)">Detected type: <b style="color:var(--fog)">{relevance.document_topic}</b></p>
            <p style="font-size:.82rem;color:var(--soft);margin-top:.3rem">{relevance.reason}</p>
          </div>
        {:else if relevance && report}
          <div class="match-row">
            <span class="match-dot"></span>
            Document matches · <b style="color:var(--fog);margin-left:.3rem">{relevance.document_topic}</b>
            {#if analysisGrounded !== null}
              <span class="vrd" style="margin-left:.5rem"><span class="vrd-dot"></span><span>{analysisGrounded ? 'Grounded' : 'Unverified'}</span></span>
            {/if}
          </div>
          <div class="rpt">
            <div class="rpt-row"><span class="lbl">Summary</span><p>{report.summary}</p></div>
            {#if report.findings.length}
              <div class="rpt-row"><span class="lbl">Findings</span>
                <ul class="fnd-ul">
                  {#each report.findings as f}
                    <li class="fnd">
                      <p class="f-obs">{f.observation}</p>
                      {#if f.evidence}<p class="f-ev">{f.evidence}</p>{/if}
                      {#if f.citation}<code class="c">{f.citation}</code>{/if}
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}
            <div class="rpt-row"><span class="lbl">Assessment</span><p>{report.assessment}</p></div>
            {#if report.recommendation}<div class="rpt-row"><span class="lbl">Recommendation</span><p>{report.recommendation}</p></div>{/if}
            {#if report.caveats}<div class="rpt-row"><span class="lbl">Caveats</span><p>{report.caveats}</p></div>{/if}
          </div>
          <div class="rpt-foot">
            {#if analysisApproved}
              <span class="sgnd-lbl">Reviewed & approved</span>
              <button class="btn btn-ghost" onclick={exportAnalysisMarkdown}><span>Download report (.md)</span></button>
            {:else}
              <button class="btn btn-sign" onclick={approveAnalysis}><span>Approve analysis</span></button>
              <span class="sgn-hint">Review findings and caveats before approving</span>
            {/if}
          </div>
        {/if}
      </section>
    {/if}

    <!-- TEST CASES -->
    {#if testCases.length > 0}
      <section class="res-sec wrap" id="res-sec">
        <div class="res-hd">
          <h2 class="res-title">Test cases</h2>
          <span class="tc-cnt">{testCases.length}</span>
          <span class="tc-prog">{signedCount}/{testCases.length} approved</span>
        </div>
        <div class="tc-list">
          {#each testCases as tc, i}
            <article class="tc {tc.signed ? 'signed' : ''}">
              <div class="tc-hd">
                <h3 class="tc-title">{tc.title}</h3>
                <span class="prio">{tc.priority}</span>
              </div>
              <div class="tc-fld"><span class="lbl">Preconditions</span><p>{tc.preconditions}</p></div>
              {#if tc.steps.length > 0}
                <div class="tc-fld"><span class="lbl">Steps</span>
                  <ol>{#each tc.steps as step}
                    <li><span class="act">{step.action}</span><span class="arr">&#8594;</span><span class="exp">{step.expected_result}</span></li>
                  {/each}</ol>
                </div>
              {/if}
              <div class="tc-fld"><span class="lbl">Traceability</span>
                <div class="cite-row">{#each tc.source_citations as cite}<code class="c">{cite}</code>{/each}</div>
              </div>
              <div class="tc-foot">
                {#if tc.signed}
                  <span class="sgnd-lbl">&#10003; Reviewed &amp; e-signed</span>
                {:else}
                  <button class="btn btn-sign" onclick={() => sign(i)}><span>Approve &amp; e-sign</span></button>
                  <span class="sgn-hint">Review all steps before approving</span>
                {/if}
              </div>
            </article>
          {/each}
        </div>
      </section>
    {/if}

    <!-- EXPORT -->
    {#if signedCount > 0}
      <section class="exp-sec wrap" id="exp-sec">
        <div class="sec-eyebrow" style="margin-bottom:.65rem"><span class="sec-num">Export</span><span class="sec-rule"></span></div>
        <h2 class="sec-title" style="margin-bottom:1rem">Test plan ready</h2>
        <div class="sm-grid">
          <div class="sm-i"><span class="sm-k">Requirement</span><span class="sm-v">{requirement.slice(0, 80)}{requirement.length > 80 ? '...' : ''}</span></div>
          <div class="sm-i"><span class="sm-k">Approved</span><span class="sm-v">{signedCount} of {testCases.length}</span></div>
          <div class="sm-i"><span class="sm-k">Model</span><span class="sm-v">{trace?.model ?? '-'}</span></div>
          <div class="sm-i"><span class="sm-k">Grounded</span><span class="sm-v">{grounded ? 'Yes' : 'No'} · {attempts} attempt{attempts !== 1 ? 's' : ''}</span></div>
          <div class="sm-i full"><span class="sm-k">Sources</span><span class="sm-v">{sourceList}</span></div>
        </div>
        <div class="exp-row">
          <button class="btn btn-accent" onclick={exportMarkdown}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M6 1v8M3.5 6.5L6 9l2.5-2.5M1 10h10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Download test plan (.md)</span>
          </button>
          <button class="btn btn-ghost" onclick={exportCSV}><span>Export for test tools (.csv)</span></button>
        </div>
        <p class="req-note">All approved cases with steps, expected results, and source citations, ready to attach to a ticket or import into a test manager.</p>
      </section>
    {/if}

  </main>

  <footer><div class="wrap">TestGenRAG &mdash; AI-drafted, citation-backed test cases &middot; RAG + LangGraph + HyDE + re-rank + LLM-as-judge</div></footer>
</div>

<style>
  /* ── TOKENS ── */
  :global(:root) {
    --bg:    #030303;
    --ink:   #070707;
    --deep:  #0d0d0d;
    --mid:   #141414;
    --dim:   #1e1e1e;
    --line:  #202020;
    --sub:   #2c2c2c;
    --mute:  #444;
    --soft:  #666;
    --pale:  #999;
    --fog:   #bbb;
    --light: #e4e4e4;
    --white: #f0f0f0;
    --a:     #d4ff00;
    --a-dim: rgba(212,255,0,.08);
    --a-glow: 0 0 24px rgba(212,255,0,.35);
    --display: 'Syne', system-ui;
    --body:    'Space Grotesk', system-ui;
    --mono:    'JetBrains Mono', monospace;
    --mx: 50%; --my: 50%;
  }
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(html) { scroll-behavior: smooth; }
  :global(body) {
    font-family: var(--body); background: var(--bg); color: var(--light);
    line-height: 1.55; overflow-x: hidden; -webkit-font-smoothing: antialiased; min-height: 100vh;
  }
  :global(button, input, textarea, select) { font-family: inherit; }
  :global(a) { color: inherit; text-decoration: none; }

  /* ── CURSOR ── */
  :global(#cdot) {
    position: fixed; width: 7px; height: 7px; border-radius: 50%; background: var(--white);
    pointer-events: none; z-index: 9999; transform: translate(-50%, -50%);
    mix-blend-mode: difference; transition: width .12s, height .12s; will-change: left, top;
  }
  :global(#cring) {
    position: fixed; width: 34px; height: 34px; border-radius: 50%;
    border: 1px solid rgba(240,240,240,.22); pointer-events: none; z-index: 9998;
    transform: translate(-50%, -50%); will-change: left, top;
    transition: width .32s cubic-bezier(.2,0,.2,1), height .32s cubic-bezier(.2,0,.2,1), border-color .2s;
  }
  :global(#cring.hover) { width: 52px; height: 52px; border-color: rgba(240,240,240,.5); }
  :global(#cring.click) { width: 18px; height: 18px; border-color: rgba(240,240,240,.8); }
  :global(.mouse-glow) {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: radial-gradient(700px circle at var(--mx) var(--my), rgba(255,255,255,.03), transparent 70%);
  }
  :global(.grain) {
    position: fixed; inset: 0; pointer-events: none; z-index: 9990; opacity: .025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
  }
  :global(.rip) {
    position: absolute; border-radius: 50%; pointer-events: none;
    background: rgba(255,255,255,.1); transform: scale(0); z-index: 10;
    animation: ripOut .55s ease-out forwards;
  }
  @keyframes ripOut { to { transform: scale(5); opacity: 0; } }

  /* ── LAYOUT ── */
  .app { position: relative; z-index: 1; display: flex; flex-direction: column; min-height: 100vh; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 0 2rem; width: 100%; }

  /* ── HEADER ── */
  header {
    position: sticky; top: 0; z-index: 100; height: 54px;
    border-bottom: 1px solid var(--dim);
    background: rgba(3,3,3,.92); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  }
  .hd { display: flex; align-items: center; justify-content: space-between; height: 100%; }
  .logo {
    font-family: var(--display); font-size: 1.05rem; font-weight: 800;
    letter-spacing: -.03em; color: var(--white); display: flex; align-items: center; gap: .3rem;
  }
  .logo-sep { color: var(--sub); font-weight: 300; font-size: .8rem; letter-spacing: .05em; margin: 0 .15rem; }
  .logo-rag { color: var(--soft); font-weight: 700; }
  .hd-right { display: flex; align-items: center; gap: 1.5rem; }
  .pipe-dots-h { display: flex; align-items: center; gap: .3rem; }
  .pdh { width: 5px; height: 5px; border-radius: 50%; background: var(--dim); transition: background .3s, box-shadow .3s; }
  .pdh.on { background: var(--a); box-shadow: 0 0 8px var(--a); }
  .pdh.done { background: var(--sub); }
  .st-tag {
    font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
    text-transform: uppercase; color: var(--soft); display: flex; align-items: center; gap: .3rem;
  }
  .st-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--sub); animation: breathe 3s ease-in-out infinite; }
  @keyframes breathe { 0%,100% { opacity: .3; transform: scale(.8); } 50% { opacity: 1; transform: scale(1.15); } }

  /* ── CONFIG ── */
  .cfg {
    padding: 1.25rem; border: 1px solid var(--dim); border-radius: 16px;
    background: #0d0d0c; overflow: hidden; position: relative; isolation: isolate;
    margin: 1.25rem auto;
  }
  .cfg::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.03) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .cfg:hover::after { left: 120%; top: -80%; }
  .cfg-row { display: flex; gap: .65rem; align-items: flex-end; flex-wrap: wrap; }
  .cfg-col { display: flex; flex-direction: column; gap: .35rem; }
  .cfg-col.grow { flex: 1; min-width: 180px; }
  .cfg-lbl { font-family: var(--mono); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--pale); }
  .ok-tag {
    font-family: var(--mono); font-size: .76rem; color: var(--soft);
    display: flex; align-items: center; gap: .25rem; white-space: nowrap; align-self: center;
  }
  .ok-tag::before { content: ''; width: 4px; height: 4px; border-radius: 50%; background: var(--sub); display: inline-block; }

  /* ── FIELDS ── */
  .f {
    width: 100%; padding: .6rem .85rem; border: 1px solid var(--dim); border-radius: 5px;
    background: var(--ink); color: var(--light); font-size: .84rem; outline: none;
    transition: border-color .2s, box-shadow .2s;
  }
  .f:focus { border-color: var(--sub); box-shadow: 0 0 0 3px rgba(255,255,255,.04); }
  .f.mono { font-family: var(--mono); font-size: .78rem; }
  .f.pr { padding-right: 5rem; }
  textarea.f { resize: vertical; line-height: 1.6; caret-color: var(--white); min-height: 90px; }
  select.f {
    cursor: pointer; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23444'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right .8rem center; padding-right: 2.4rem;
  }
  .fwrap { position: relative; }
  .fwrap .fa { position: absolute; right: .5rem; top: 50%; transform: translateY(-50%); }

  /* ── BUTTONS ── */
  .btn {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .58rem 1.15rem; border-radius: 5px; font-size: .84rem; font-weight: 500;
    border: 1px solid var(--sub); color: var(--light); background: transparent;
    cursor: pointer; position: relative; overflow: hidden; white-space: nowrap;
    transition: border-color .22s, color .22s; user-select: none;
  }
  .btn > * { position: relative; z-index: 1; pointer-events: none; }
  .btn::before {
    content: ''; position: absolute; inset: 0; background: var(--white);
    transform: translateX(-101%); transform-origin: left;
    transition: transform .3s cubic-bezier(.4,0,.2,1); z-index: 0;
  }
  .btn:not(:disabled):hover::before { transform: translateX(0); }
  .btn:not(:disabled):hover { color: var(--bg); border-color: var(--white); }
  .btn:not(:disabled):active { transform: scale(.97); }
  .btn:disabled { opacity: .3; cursor: not-allowed; }
  .btn-solid { background: var(--white); color: var(--bg); border-color: var(--white); }
  .btn-solid::before { background: var(--mid); }
  .btn-solid:not(:disabled):hover { color: var(--light); border-color: var(--sub); }
  .btn-accent { background: var(--a); color: var(--bg); border-color: var(--a); font-weight: 600; box-shadow: 0 0 20px rgba(212,255,0,.18); }
  .btn-accent::before { background: var(--mid); }
  .btn-accent:not(:disabled):hover { color: var(--light); border-color: var(--sub); box-shadow: none; }
  .btn-ghost { color: var(--fog); border-color: var(--sub); font-size: .84rem; padding: .5rem 1rem; }
  .btn-ghost::before { background: var(--deep); }
  .btn-ghost:not(:disabled):hover { color: var(--light); border-color: var(--sub); }
  .btn-sign { color: var(--soft); border-color: var(--dim); }
  .btn-xl { padding: .72rem 1.5rem; font-size: .94rem; font-weight: 600; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spn { animation: spin .7s linear infinite; display: inline-block; }

  /* ── MAIN ── */
  main { display: flex; flex-direction: column; gap: 1rem; padding: 0 0 4rem; }

  /* ── SECTION CARDS ── */
  .sec {
    padding: clamp(2rem,4vw,3rem); border: 1px solid var(--dim); border-radius: 16px;
    background: var(--ink); overflow: hidden; position: relative; isolation: isolate;
  }
  .sec::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.03) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .sec:hover::after { left: 120%; top: -80%; }
  #sec-ingest { background: #0b0c0f; border-color: #14161c; }
  #sec-req    { background: #0f0e0a; border-color: #1c1a12; }
  .sec-bg {
    position: absolute; right: -.02em; top: -.1em;
    font-family: var(--display); font-size: clamp(9rem,16vw,14rem);
    font-weight: 800; color: rgba(255,255,255,.04); line-height: 1;
    pointer-events: none; user-select: none; letter-spacing: -.05em; z-index: 0;
  }
  .sec-inner { position: relative; z-index: 1; }
  .sec-eyebrow { display: flex; align-items: center; gap: .5rem; margin-bottom: .65rem; }
  .sec-num { font-family: var(--mono); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--soft); }
  .sec-rule { flex: 1; height: 1px; background: var(--dim); max-width: 2rem; }
  .sec-title {
    font-family: var(--display); font-size: clamp(1.6rem,3.5vw,2.2rem);
    font-weight: 800; color: var(--white); letter-spacing: -.04em; margin-bottom: .65rem; line-height: 1.1;
  }
  .sec-hint { font-size: .93rem; color: var(--fog); margin-bottom: 1.4rem; line-height: 1.7; max-width: 640px; }
  .req-note { font-size: .71rem; color: var(--mute); margin-top: .65rem; }

  /* ── UPLOAD ── */
  .upload {
    border: 1px dashed var(--sub); border-radius: 8px; padding: 2rem 1.75rem;
    display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap;
    cursor: pointer; margin-bottom: 1rem; position: relative; overflow: hidden;
    transition: border-color .2s, background .2s;
  }
  .upload:hover { border-color: var(--pale); }
  .upload input { display: none; }
  .upload::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.05) 50%, transparent 75%);
    transition: left .75s ease, top .75s ease; pointer-events: none;
  }
  .upload:hover::after { left: 130%; top: -80%; }
  .upl-ico { color: var(--mute); flex-shrink: 0; transition: color .2s; }
  .upload:hover .upl-ico { color: var(--pale); }
  .upl-txt { flex: 1; min-width: 0; }
  .upl-primary { font-size: .9rem; font-weight: 500; color: var(--light); }
  .upl-sub { font-size: .84rem; color: var(--soft); margin-top: .2rem; }
  .doc-list { display: flex; flex-direction: column; gap: .3rem; }
  .doc-item {
    display: flex; align-items: center; gap: .6rem;
    padding: .52rem .8rem; border: 1px solid var(--dim); border-radius: 5px;
    background: var(--ink); font-size: .82rem; transition: border-color .2s;
  }
  .doc-item:hover { border-color: var(--sub); }
  .doc-ico { color: var(--mute); flex-shrink: 0; }
  .doc-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-ch { font-family: var(--mono); font-size: .75rem; color: var(--soft); white-space: nowrap; }
  .empty { font-size: .8rem; color: var(--mute); font-style: italic; }

  /* ── ACTIONS ── */
  .actions { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; margin-top: 1rem; }
  .vrd {
    display: flex; align-items: center; gap: .3rem;
    font-family: var(--mono); font-size: .78rem; color: var(--soft);
    border: 1px solid var(--sub); border-radius: 20px; padding: .3rem .8rem;
  }
  .vrd-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--sub); }

  /* ── PIPELINE ── */
  .pipe-sec {
    padding: clamp(2rem,4vw,3rem); border: 1px solid #10131e; border-radius: 16px;
    background: #080a10; position: relative; overflow: hidden; isolation: isolate;
  }
  .pipe-sec::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.025) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .pipe-sec:hover::after { left: 120%; top: -80%; }
  .pipe-hd {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: .5rem; margin-bottom: 2.5rem;
  }
  .pipe-hd-title {
    font-family: var(--display); font-size: 1.15rem; font-weight: 800;
    color: var(--white); letter-spacing: -.03em; display: flex; align-items: center; gap: .5rem;
  }
  .live-pill {
    font-family: var(--mono); font-size: .68rem; letter-spacing: .09em; text-transform: uppercase;
    color: var(--a); border: 1px solid rgba(212,255,0,.3); border-radius: 3px; padding: .15rem .5rem;
  }
  .pipe-timing { font-family: var(--mono); font-size: .78rem; color: var(--soft); }
  .pipe-svg-wrap { position: relative; margin-bottom: 1.25rem; }
  .pipe-svg { width: 100%; height: 60px; overflow: visible; }
  .p-base { stroke: var(--dim); stroke-width: 1; fill: none; }
  .p-fill {
    stroke: var(--sub); stroke-width: 1.5; fill: none;
    stroke-dasharray: 1200; stroke-dashoffset: 1200;
    transition: stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1);
  }
  .p-electron {
    fill: var(--white);
    filter: drop-shadow(0 0 4px rgba(255,255,255,.9)) drop-shadow(0 0 8px rgba(255,255,255,.5));
  }
  .pipe-nodes { display: flex; justify-content: space-between; padding: 0 1px; }
  .pn {
    display: flex; flex-direction: column; align-items: center; gap: .55rem;
    flex: 0 0 auto; background: transparent; border: none; cursor: default;
  }
  .pn-disc {
    width: 32px; height: 32px; border-radius: 50%;
    border: 1.5px solid var(--dim); background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    transition: border-color .25s, background .25s;
  }
  .pn-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dim); transition: all .2s; }
  .pn.pending .pn-disc { border-color: var(--line); }
  .pn.pending .pn-dot { background: var(--line); }
  .pn.running .pn-disc { border-color: var(--a); box-shadow: 0 0 16px var(--a-dim); }
  .pn.running .pn-dot { background: var(--a); box-shadow: 0 0 8px var(--a); animation: glow 1s ease-in-out infinite; }
  @keyframes glow { 0%,100% { box-shadow: 0 0 6px rgba(212,255,0,.4); transform: scale(.85); } 50% { box-shadow: 0 0 16px rgba(212,255,0,.9); transform: scale(1.2); } }
  .pn.done .pn-disc { border-color: var(--sub); background: var(--deep); cursor: pointer; }
  .pn.done .pn-dot { background: var(--soft); }
  .pn.done:hover .pn-disc { border-color: var(--pale); }
  .pn.open .pn-disc { border-color: var(--fog); }
  .pn-lbl { font-size: .84rem; font-weight: 600; color: var(--soft); transition: color .2s; text-align: center; }
  .pn-dsc { font-size: .72rem; color: var(--mute); text-align: center; }
  .pn.running .pn-lbl { color: var(--a); }
  .pn.done .pn-lbl { color: var(--fog); }
  .pipe-hint { font-size: .82rem; color: var(--soft); margin-top: .6rem; }
  .pipe-det { background: var(--ink); border: 1px solid var(--dim); border-radius: 7px; padding: 1.1rem 1.25rem; margin-top: 1.2rem; }
  .dlbl { font-family: var(--mono); font-size: .69rem; text-transform: uppercase; letter-spacing: .08em; color: var(--soft); margin: .7rem 0 .35rem; }
  .dlbl:first-child { margin-top: 0; }
  .mono-blk { font-family: var(--mono); font-size: .75rem; color: var(--soft); background: var(--bg); border: 1px solid var(--dim); border-radius: 5px; padding: .75rem .9rem; line-height: 1.7; }
  .chips { display: flex; flex-wrap: wrap; gap: .3rem; }
  .chip { font-family: var(--mono); font-size: .65rem; color: var(--soft); background: var(--ink); border: 1px solid var(--dim); border-radius: 3px; padding: .15rem .45rem; }
  .chunk { background: var(--bg); border: 1px solid var(--dim); border-radius: 5px; padding: .5rem .7rem; margin-bottom: .35rem; }
  .chunk-src { font-family: var(--mono); font-size: .73rem; color: var(--soft); margin-bottom: .2rem; }
  .chunk-txt { font-size: .78rem; color: var(--soft); }

  /* ── ANALYSIS ── */
  .ana-sec {
    padding: clamp(2rem,4vw,3rem); border: 1px solid #181419; border-radius: 16px;
    background: #0d0b10; overflow: hidden; position: relative; isolation: isolate;
  }
  .ana-sec::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.025) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .ana-sec:hover::after { left: 120%; top: -80%; }
  .spin-row { display: flex; align-items: center; gap: .55rem; font-size: .84rem; color: var(--soft); padding: .25rem 0; }
  .match-row { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; font-size: .81rem; color: var(--soft); margin: .75rem 0 1rem; }
  .match-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--sub); flex-shrink: 0; }
  .guardrail { background: var(--ink); border: 1px solid var(--dim); border-radius: 6px; padding: .9rem 1rem; }
  .rpt { background: var(--ink); border: 1px solid var(--dim); border-radius: 8px; padding: 1.1rem 1.25rem; display: flex; flex-direction: column; gap: .8rem; margin-bottom: .85rem; }
  .rpt-row p { font-size: .9rem; color: var(--pale); line-height: 1.7; }
  .lbl { display: block; font-family: var(--mono); font-size: .69rem; text-transform: uppercase; letter-spacing: .07em; color: var(--soft); margin-bottom: .35rem; }
  .fnd-ul { list-style: none; display: flex; flex-direction: column; gap: .45rem; }
  .fnd { background: var(--bg); border: 1px solid var(--dim); border-radius: 5px; padding: .55rem .7rem; }
  .f-obs { font-size: .82rem; font-weight: 500; color: var(--fog); margin-bottom: .14rem; }
  .f-ev { font-size: .75rem; color: var(--mute); font-style: italic; margin-bottom: .25rem; }
  .c { font-family: var(--mono); font-size: .61rem; color: var(--mute); background: var(--ink); border: 1px solid var(--dim); border-radius: 3px; padding: .1rem .38rem; }
  .rpt-foot { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }

  /* ── TEST CASES ── */
  .res-sec {
    padding: clamp(2rem,4vw,3rem); border: 1px solid #131913; border-radius: 16px;
    background: #0b0e0b; overflow: hidden; position: relative; isolation: isolate;
  }
  .res-sec::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.025) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .res-sec:hover::after { left: 120%; top: -80%; }
  .res-hd { display: flex; align-items: center; gap: .55rem; margin-bottom: 1.75rem; }
  .res-title { font-family: var(--display); font-size: clamp(1.4rem,3vw,2rem); font-weight: 800; color: var(--white); letter-spacing: -.04em; }
  .tc-cnt { font-family: var(--mono); font-size: .6rem; background: var(--mid); border: 1px solid var(--sub); border-radius: 10px; padding: .1rem .5rem; color: var(--mute); }
  .tc-prog { font-size: .75rem; color: var(--mute); margin-left: auto; }
  .tc-list { display: flex; flex-direction: column; gap: .65rem; }
  .tc {
    background: var(--ink); border: 1px solid var(--dim); border-radius: 8px;
    padding: 1.35rem 1.5rem; position: relative; overflow: hidden; transition: border-color .22s;
  }
  .tc::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
    background: var(--sub); transition: background .3s; border-radius: 8px 0 0 8px; z-index: 2;
  }
  .tc::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.055) 50%, transparent 75%);
    transition: left .7s ease, top .7s ease; pointer-events: none; z-index: 1;
  }
  .tc:hover::after { left: 130%; top: -80%; }
  .tc:hover { border-color: var(--sub); }
  .tc:hover::before { background: var(--mute); }
  .tc.signed::before { background: var(--a); }
  .tc-hd { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: .9rem; }
  .tc-title { font-size: .96rem; font-weight: 600; color: var(--light); line-height: 1.35; letter-spacing: -.01em; }
  .prio { font-family: var(--mono); font-size: .68rem; letter-spacing: .07em; text-transform: uppercase; color: var(--pale); border: 1px solid var(--sub); border-radius: 3px; padding: .18rem .55rem; white-space: nowrap; flex-shrink: 0; }
  .tc-fld { margin-bottom: .75rem; }
  .tc-fld p { font-size: .89rem; color: var(--pale); }
  .tc-fld ol { padding-left: 1.1rem; display: flex; flex-direction: column; gap: .4rem; }
  .tc-fld li { font-size: .89rem; }
  .act { color: var(--soft); } .arr { color: var(--mute); margin: 0 .3rem; } .exp { color: var(--fog); font-style: italic; }
  .cite-row { display: flex; flex-wrap: wrap; gap: .25rem; margin-top: .35rem; }
  .tc-foot { display: flex; align-items: center; gap: .65rem; border-top: 1px solid var(--dim); padding-top: .8rem; }
  .sgnd-lbl { font-size: .86rem; font-weight: 500; color: var(--fog); }
  .sgn-hint { font-size: .8rem; color: var(--soft); }

  /* ── EXPORT ── */
  .exp-sec {
    padding: clamp(2rem,4vw,3rem); border: 1px solid #1c1814; border-radius: 16px;
    background: #100d0a; overflow: hidden; position: relative; isolation: isolate;
  }
  .exp-sec::after {
    content: ''; position: absolute; top: -80%; left: -60%; width: 55%; height: 260%;
    background: linear-gradient(105deg, transparent 25%, rgba(255,255,255,.025) 50%, transparent 75%);
    transition: left .9s ease, top .9s ease; pointer-events: none; z-index: 10;
  }
  .exp-sec:hover::after { left: 120%; top: -80%; }
  .sm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .55rem .85rem; margin: 1rem 0 1.2rem; }
  .sm-i { display: flex; flex-direction: column; gap: .12rem; }
  .sm-i.full { grid-column: 1/-1; }
  .sm-k { font-family: var(--mono); font-size: .69rem; text-transform: uppercase; letter-spacing: .06em; color: var(--soft); }
  .sm-v { font-size: .82rem; color: var(--light); font-weight: 500; word-break: break-word; }
  .exp-row { display: flex; gap: .65rem; flex-wrap: wrap; }

  /* ── ERROR / FOOTER ── */
  .err { background: var(--ink); border: 1px solid var(--sub); border-radius: 6px; padding: .7rem 1rem; font-size: .82rem; color: var(--soft); margin: .5rem 0; }
  footer { border-top: 1px solid var(--dim); padding: 2rem 0; font-family: var(--mono); font-size: .55rem; letter-spacing: .1em; text-transform: uppercase; color: var(--mute); text-align: center; }

  /* ── RESPONSIVE ── */
  @media (max-width: 600px) {
    .hd-right .pipe-dots-h { display: none; }
    .cfg-row { flex-direction: column; }
    .sm-grid { grid-template-columns: 1fr; }
    .sec-bg { font-size: 6rem; }
    .actions { flex-direction: column; align-items: stretch; }
    .pn-dsc { display: none; }
    .sec, .pipe-sec, .ana-sec, .res-sec, .exp-sec, .cfg { border-radius: 10px; }
  }
</style>
