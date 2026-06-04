<script lang="ts">
  import { onMount } from 'svelte';

  // Same-origin in production (UI served by FastAPI); localhost in dev.
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

  // Analysis / execution
  let analyzing = $state(false);
  let analysisError = $state('');
  let relevance = $state<Relevance | null>(null);
  let report = $state<AnalysisReport | null>(null);
  let analysisGrounded = $state<boolean | null>(null);
  let analysisTrace = $state<Trace | null>(null);
  let analysisApproved = $state(false);

  const STAGES = [
    { key: 'hyde', label: 'HyDE', desc: 'Rewrite as spec' },
    { key: 'retrieve', label: 'Retrieve', desc: 'Search + re-rank' },
    { key: 'draft', label: 'Draft', desc: 'Write cases' },
    { key: 'judge', label: 'Judge', desc: 'Verify grounding' },
  ];

  const NIM_MODELS = [
    { value: 'meta/llama-3.1-8b-instruct',             label: 'Llama 3.1 8B  — fast, recommended' },
    { value: 'meta/llama-3.1-70b-instruct',            label: 'Llama 3.1 70B — stronger' },
    { value: 'meta/llama-3.3-70b-instruct',            label: 'Llama 3.3 70B — newest Meta' },
    { value: 'nvidia/llama-3.1-nemotron-70b-instruct', label: 'Nemotron 70B — NVIDIA tuned' },
    { value: 'mistralai/mistral-7b-instruct-v0.3',     label: 'Mistral 7B — lightweight' },
    { value: 'mistralai/mixtral-8x7b-instruct-v0.1',   label: 'Mixtral 8x7B — mixture of experts' },
    { value: 'microsoft/phi-3-mini-128k-instruct',     label: 'Phi-3 Mini — small but capable' },
    { value: 'custom',                                 label: 'Custom — enter any model ID' },
  ];

  const PRIORITY_COLOR: Record<string, string> = {
    High: '#f87171', Medium: '#fbbf24', Low: '#34d399',
  };

  $effect(() => { if (selectedModel !== 'custom') customModel = ''; });

  let signedCount = $derived(testCases.filter((t) => t.signed).length);
  let sourceList = $derived(
    trace && trace.chunks.length
      ? [...new Set(trace.chunks.map((c) => `${c.source} p.${c.page}`))].join(', ')
      : '—'
  );

  async function refreshDocs(): Promise<void> {
    try {
      const res = await fetch(`${API}/documents`);
      docs = (await res.json()) as Doc[];
    } catch { /* backend not ready yet */ }
  }
  onMount(() => { refreshDocs(); });

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
      preconditions: tc.preconditions ?? '—',
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
  }
  function failStages() {
    for (const k of Object.keys(stageState)) if (stageState[k] === 'running') stageState[k] = 'pending';
    stageState = { ...stageState };
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
        body: JSON.stringify({
          requirement,
          model: selectedModel === 'custom' ? customModel.trim() : selectedModel,
        }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      const data = await res.json();
      trace = (data.trace ?? null) as Trace | null;
      grounded = data.grounded ?? null;
      attempts = data.attempts ?? 0;
      cached = Boolean(data.cached);
      const cases: Partial<TestCase>[] = Array.isArray(data.test_cases) ? data.test_cases : [];
      testCases = cases.map(normalise);
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
        body: JSON.stringify({
          requirement,
          model: selectedModel === 'custom' ? customModel.trim() : selectedModel,
        }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      const d = await res.json();
      relevance = { relevant: d.relevant, document_topic: d.document_topic, reason: d.reason };
      analysisTrace = (d.trace ?? null) as Trace | null;
      if (d.relevant && d.report) {
        report = d.report as AnalysisReport;
        analysisGrounded = d.grounded ?? null;
      }
    } catch (e) {
      analysisError = (e as Error).message;
    } finally {
      analyzing = false;
    }
  }

  function approveAnalysis() { analysisApproved = true; }

  function exportAnalysisMarkdown() {
    if (!report) return;
    let md = `# Analysis Report\n\n`;
    md += `**Request:** ${requirement}\n\n`;
    md += `**Generated:** ${new Date().toLocaleString()}  \n`;
    md += `**Model:** ${analysisTrace?.model ?? '—'}  \n`;
    md += `**Document type:** ${relevance?.document_topic ?? '—'}  \n`;
    md += `**Grounded:** ${analysisGrounded ? 'Yes' : 'No'}\n\n---\n\n`;
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
  function exportMarkdown() {
    const cases = exportCases();
    let md = `# Test Plan\n\n`;
    md += `**Requirement:** ${requirement}\n\n`;
    md += `**Generated:** ${new Date().toLocaleString()}  \n`;
    md += `**Model:** ${trace?.model ?? '—'}  \n`;
    md += `**Grounded:** ${grounded ? 'Yes' : 'No'} (after ${attempts} attempt${attempts !== 1 ? 's' : ''})  \n`;
    md += `**Sources used:** ${sourceList}\n\n---\n\n`;
    cases.forEach((tc, i) => {
      md += `## ${i + 1}. ${tc.title}  _(Priority: ${tc.priority})_\n\n`;
      md += `**Preconditions:** ${tc.preconditions}\n\n`;
      md += `| # | Action | Expected result |\n|---|--------|------------------|\n`;
      tc.steps.forEach((s, j) => { md += `| ${j + 1} | ${s.action} | ${s.expected_result} |\n`; });
      md += `\n**Traceability:** ${tc.source_citations.join(', ')}\n\n`;
    });
    downloadFile('test-plan.md', md, 'text/markdown');
  }
  function exportCSV() {
    const cases = exportCases();
    const rows: string[][] = [[
      'Test Case', 'Priority', 'Preconditions', 'Step', 'Action', 'Expected Result', 'Citations',
    ]];
    cases.forEach((tc) => tc.steps.forEach((s, j) => rows.push([
      tc.title, tc.priority, tc.preconditions, String(j + 1),
      s.action, s.expected_result, tc.source_citations.join('; '),
    ])));
    const csv = rows
      .map((r) => r.map((cell) => `"${(cell ?? '').replace(/"/g, '""')}"`).join(','))
      .join('\n');
    downloadFile('test-plan.csv', csv, 'text/csv');
  }
</script>

<svelte:head><title>TestGenRAG</title></svelte:head>

<div class="app">
  <div class="bg-grid"></div>

  <header>
    <div class="inner head-inner">
      <div class="logo">
        <span class="logo-mark">⚗</span>
        <span class="logo-text">TestGen<strong>RAG</strong></span>
        <span class="status"><span class="status-dot"></span>online</span>
      </div>
      <p class="tagline">RAG · LangGraph · HyDE · re-rank · LLM-as-judge · e-sign</p>
    </div>
  </header>

  <main class="inner">
    <!-- API key + model -->
    <section class="card key-card">
      <div class="key-header">
        <div>
          <h2 class="card-title">🔑 Your NVIDIA API key</h2>
          <p class="hint">
            Sent directly to the model for your request only, never stored on this server.
            <a class="link" href="https://build.nvidia.com" target="_blank" rel="noopener">
              Get a free key →
            </a>
          </p>
        </div>
        <span class="badge-free">Free · No credit card</span>
      </div>
      <div class="key-row">
        <input class="field mono" type={keyVisible ? 'text' : 'password'} placeholder="nvapi-..."
               bind:value={apiKey} spellcheck="false" autocomplete="off" />
        <button class="btn btn-ghost" onclick={() => (keyVisible = !keyVisible)}>
          {keyVisible ? 'Hide' : 'Show'}
        </button>
        {#if apiKey.trim()}<span class="ok">✓ Key entered</span>{/if}
      </div>
      <div class="model-row">
        <span class="model-label">Model</span>
        <select class="field" bind:value={selectedModel}>
          {#each NIM_MODELS as m}<option value={m.value}>{m.label}</option>{/each}
        </select>
        {#if selectedModel === 'custom'}
          <input class="field mono" type="text" placeholder="e.g. deepseek-ai/deepseek-v3"
                 bind:value={customModel} spellcheck="false" />
        {/if}
      </div>
      <p class="subnote">Free, reliable models on NVIDIA NIM. If a custom model errors, it isn't on your account — pick a Llama or Mistral one.</p>
    </section>

    <!-- Step 1: Ingest -->
    <section class="card">
      <h2 class="card-title"><span class="step">1</span> Ingest documentation</h2>
      <p class="hint">Each PDF is extracted, chunked, embedded, and indexed in FAISS.</p>
      <div class="upload-row">
        <label class="file-picker">
          <input type="file" accept=".pdf"
                 onchange={(e) => (file = (e.currentTarget as HTMLInputElement).files?.[0] ?? null)} />
          <span class="file-name">{file ? `📄 ${file.name}` : 'Choose a PDF…'}</span>
        </label>
        <button class="btn btn-primary" onclick={upload} disabled={!file || uploading}>
          {uploading ? '⏳ Indexing…' : '⬆ Upload & index'}
        </button>
      </div>
      {#if docs.length > 0}
        <ul class="doc-list">
          {#each docs as doc}
            <li><span>📄</span><span class="doc-name">{doc.name}</span>
              <span class="doc-meta">{doc.chunks} chunks{doc.s3 ? ' · ☁ S3' : ''}</span></li>
          {/each}
        </ul>
      {:else}
        <p class="empty">No documents indexed yet.</p>
      {/if}
    </section>

    <!-- Step 2: Generate -->
    <section class="card">
      <h2 class="card-title"><span class="step">2</span> Describe a requirement</h2>
      <p class="hint">The agent runs HyDE → query selection → FAISS + re-rank → draft → LLM-as-judge, with a retry loop.</p>
      <textarea class="field req" rows="3"
        placeholder="e.g. The system must trigger an audible alarm when a monitored value drops below the configured threshold."
        bind:value={requirement}></textarea>
      <div class="gen-row">
        <button class="btn btn-accent btn-lg" onclick={generate} disabled={loading || analyzing || !requirement.trim()}>
          {#if loading}<span class="spinner"></span> Running agent…{:else}▶ Generate test cases{/if}
        </button>
        <button class="btn btn-primary btn-lg" onclick={analyze} disabled={analyzing || loading || !requirement.trim()}>
          {#if analyzing}<span class="spinner spinner-dark"></span> Analyzing…{:else}🔬 Analyze &amp; answer{/if}
        </button>
        {#if grounded !== null}
          <span class="verdict {grounded ? 'v-pass' : 'v-warn'}">
            {grounded ? '✔ Grounded' : '⚠ Unverified'} · {attempts} attempt{attempts !== 1 ? 's' : ''}{cached ? ' · cached' : ''}
          </span>
        {/if}
      </div>
      <p class="subnote">Generate writes test cases. Analyze reasons over the document to answer your request, and warns if the document doesn't match.</p>
    </section>

    {#if analysisError}<div class="error">⚠ {analysisError}</div>{/if}

    <!-- Analysis / execution -->
    {#if analyzing || relevance}
      <section class="card analysis">
        <h2 class="card-title">🔬 Document analysis
          {#if analysisTrace}<span class="pipe-meta">{(analysisTrace.elapsed_ms / 1000).toFixed(1)}s · {analysisTrace.model}</span>{/if}
        </h2>
        {#if analyzing}
          <div class="analyzing"><span class="spinner spinner-dark"></span> Checking relevance, then analyzing the document…</div>
        {:else if relevance && !relevance.relevant}
          <div class="guardrail">
            <div class="guardrail-head">⚠ This document doesn't match your request</div>
            <p class="guardrail-topic">Detected document type: <b>{relevance.document_topic}</b></p>
            <p class="guardrail-reason">{relevance.reason}</p>
            <p class="guardrail-hint">Upload a document that contains the right information, or rephrase your request — no analysis was produced because it would not be grounded in this document.</p>
          </div>
        {:else if relevance && report}
          <div class="match-badge">
            <span>✓ Document matches your request · <b>{relevance.document_topic}</b></span>
            {#if analysisGrounded !== null}
              <span class="verdict {analysisGrounded ? 'v-pass' : 'v-warn'}">{analysisGrounded ? '✔ Grounded' : '⚠ Unverified'}</span>
            {/if}
          </div>
          <div class="report {analysisApproved ? 'report-approved' : ''}">
            <div class="rep-block"><span class="lbl">Summary</span><p>{report.summary}</p></div>
            {#if report.findings.length}
              <div class="rep-block"><span class="lbl">Findings</span>
                <ul class="findings">
                  {#each report.findings as f}
                    <li>
                      <p class="f-obs">{f.observation}</p>
                      {#if f.evidence}<p class="f-ev">{f.evidence}</p>{/if}
                      {#if f.citation}<code class="cite">{f.citation}</code>{/if}
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}
            <div class="rep-block"><span class="lbl">Assessment</span><p>{report.assessment}</p></div>
            {#if report.recommendation}<div class="rep-block"><span class="lbl">Recommendation</span><p>{report.recommendation}</p></div>{/if}
            {#if report.caveats}<div class="rep-block caveat"><span class="lbl">Caveats</span><p>{report.caveats}</p></div>{/if}
          </div>
          <div class="report-foot">
            {#if analysisApproved}
              <span class="signed-label">✔ Reviewed &amp; approved</span>
              <button class="btn btn-accent" onclick={exportAnalysisMarkdown}>⬇ Download report (.md)</button>
            {:else}
              <button class="btn btn-sign" onclick={approveAnalysis}>✍ Approve analysis</button>
              <span class="sign-hint">Review the findings and caveats before approving</span>
            {/if}
          </div>
        {/if}
      </section>
    {/if}

    {#if error}<div class="error">⚠ {error}</div>{/if}

    <!-- Live pipeline -->
    {#if loading || trace}
      <section class="card pipeline">
        <div class="pipe-head">
          <h2 class="card-title">Agent pipeline <span class="live-tag">live</span></h2>
          {#if trace}<span class="pipe-meta">{(trace.elapsed_ms / 1000).toFixed(1)}s · {trace.model}</span>{/if}
        </div>
        <div class="nodes">
          {#each STAGES as s, i}
            <button class="node {stageState[s.key]} {openStage === s.key ? 'open' : ''}"
                    onclick={() => toggleStage(s.key)} disabled={stageState[s.key] === 'pending'}>
              <span class="node-dot"></span>
              <span class="node-label">{s.label}</span>
              <span class="node-desc">{s.desc}</span>
            </button>
            {#if i < STAGES.length - 1}
              <span class="connector {stageState[s.key] === 'done' ? 'lit' : ''}"></span>
            {/if}
          {/each}
        </div>
        {#if trace}<p class="pipe-hint">Tap a completed step to see what the agent actually did.</p>{/if}

        {#if openStage && trace}
          <div class="detail">
            {#if openStage === 'hyde'}
              <p class="detail-label">Hypothetical spec the agent wrote, then searched with</p>
              <p class="mono-block">{trace.hyde}</p>
            {:else if openStage === 'retrieve'}
              <p class="detail-label">Search queries ({trace.queries.length}) · re-rank: {trace.rerank_method}</p>
              <div class="chips">
                {#each trace.queries as q}<code class="chip">{q.length > 80 ? q.slice(0, 80) + '…' : q}</code>{/each}
              </div>
              <p class="detail-label">Retrieved passages ({trace.chunks.length}) · extractor: {trace.extractor}</p>
              {#each trace.chunks as c}
                <div class="chunk">
                  <div class="chunk-src">{c.source} · p.{c.page} · {c.page_type}</div>
                  <div class="chunk-text">{c.snippet}…</div>
                </div>
              {/each}
            {:else if openStage === 'draft'}
              <p class="detail-label">Drafted {testCases.length} test case{testCases.length !== 1 ? 's' : ''}</p>
              <ul class="draft-list">{#each testCases as tc}<li>{tc.title}</li>{/each}</ul>
            {:else if openStage === 'judge'}
              <p class="detail-label">Grounding verdict (LLM-as-judge)</p>
              <p class="verdict-detail {grounded ? 'ok' : 'warn'}">
                {grounded ? '✓ Every claim traced to a cited source' : '⚠ Could not fully verify grounding'}
                · {attempts} attempt{attempts !== 1 ? 's' : ''}
              </p>
            {/if}
          </div>
        {/if}
      </section>
    {/if}

    <!-- Results -->
    {#if testCases.length > 0}
      <section class="results">
        <div class="results-header">
          <h2 class="card-title">Draft test cases <span class="count">{testCases.length}</span></h2>
          <span class="progress">{signedCount}/{testCases.length} approved
            {#if signedCount === testCases.length}<span class="all-signed">· done ✔</span>{/if}</span>
        </div>
        {#each testCases as tc, i}
          <article class="tc {tc.signed ? 'tc-signed' : ''}">
            <div class="tc-head">
              <h3>{tc.title}</h3>
              <span class="priority" style="--p:{PRIORITY_COLOR[tc.priority] ?? '#94a3b8'}">{tc.priority}</span>
            </div>
            <div class="tc-field"><span class="lbl">Preconditions</span><p>{tc.preconditions}</p></div>
            {#if tc.steps.length > 0}
              <div class="tc-field"><span class="lbl">Steps</span>
                <ol>{#each tc.steps as step}
                  <li><span class="action">{step.action}</span><span class="arrow">→</span><span class="expected">{step.expected_result}</span></li>
                {/each}</ol>
              </div>
            {/if}
            <div class="tc-field citations"><span class="lbl">Traceability</span>
              <div class="cite-list">{#each tc.source_citations as cite}<code class="cite">{cite}</code>{/each}</div>
            </div>
            <div class="tc-foot">
              {#if tc.signed}
                <span class="signed-label">✔ Reviewed & e-signed · saved</span>
              {:else}
                <button class="btn btn-sign" onclick={() => sign(i)}>✍ Approve & e-sign</button>
                <span class="sign-hint">Review all steps before approving</span>
              {/if}
            </div>
          </article>
        {/each}
      </section>
    {/if}

    <!-- Export / output -->
    {#if signedCount > 0}
      <section class="card export">
        <h2 class="card-title">✓ Approved — your test plan is ready</h2>
        <div class="summary">
          <div><span>Requirement</span><b>{requirement}</b></div>
          <div><span>Approved</span><b>{signedCount} of {testCases.length}</b></div>
          <div><span>Model</span><b>{trace?.model ?? '—'}</b></div>
          <div><span>Grounded</span><b>{grounded ? 'Yes' : 'No'} ({attempts} attempt{attempts !== 1 ? 's' : ''})</b></div>
          <div class="full"><span>Sources</span><b>{sourceList}</b></div>
        </div>
        <div class="export-actions">
          <button class="btn btn-accent" onclick={exportMarkdown}>⬇ Download test plan (.md)</button>
          <button class="btn btn-primary" onclick={exportCSV}>⬇ Export for test tools (.csv)</button>
        </div>
        <p class="subnote">The plan includes every approved case with steps, expected results, and source citations — ready to attach to a ticket or import into a test manager.</p>
      </section>
    {/if}
  </main>

  <footer>TestGenRAG — AI-drafted, citation-backed test cases for regulated software, with human review and e-sign.</footer>
</div>

<style>
  *,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    background: #080b15; color: #e2e8f0; line-height: 1.5;
  }
  .app { position: relative; min-height: 100vh; display: flex; flex-direction: column;
    background: radial-gradient(1200px 600px at 70% -10%, #16203b 0%, #0a0f1f 45%, #080b15 100%); }
  .bg-grid { position: fixed; inset: 0; pointer-events: none; opacity: .35;
    background:
      linear-gradient(rgba(34,211,238,.04) 1px, transparent 1px) 0 0 / 100% 38px,
      linear-gradient(90deg, rgba(34,211,238,.04) 1px, transparent 1px) 0 0 / 38px 100%; }

  .inner { max-width: 880px; margin: 0 auto; width: 100%; }
  header { position: relative; z-index: 1; padding: 1.1rem 1.5rem;
    border-bottom: 1px solid #1c2746; background: rgba(8,11,21,.7); backdrop-filter: blur(8px); }
  .head-inner { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }
  .logo { display: flex; align-items: center; gap: .55rem; font-size: 1.3rem; font-weight: 300; }
  .logo-mark { font-size: 1.5rem; filter: drop-shadow(0 0 8px rgba(34,211,238,.6)); }
  .logo-text strong { font-weight: 800;
    background: linear-gradient(90deg, #22d3ee, #a78bfa); -webkit-background-clip: text;
    background-clip: text; -webkit-text-fill-color: transparent; }
  .status { display: inline-flex; align-items: center; gap: .3rem; font-size: .65rem; text-transform: uppercase;
    letter-spacing: .08em; color: #34d399; margin-left: .4rem; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399;
    animation: dotPulse 1.6s infinite; }
  .tagline { font-size: .72rem; color: #5e6b8a; letter-spacing: .03em; }

  main.inner { position: relative; z-index: 1; flex: 1; padding: 1.5rem 1rem; display: flex; flex-direction: column; gap: 1.1rem; }

  .card { background: rgba(18,24,41,.85); border: 1px solid #232b44; border-radius: 16px; padding: 1.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 32px -16px rgba(0,0,0,.6); }
  .card-title { display: flex; align-items: center; gap: .55rem; font-size: .98rem; font-weight: 650; color: #f1f5f9; }
  .step { width: 22px; height: 22px; border-radius: 50%; font-size: .7rem; font-weight: 800; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; color: #051018;
    background: linear-gradient(135deg, #22d3ee, #38bdf8); box-shadow: 0 0 12px rgba(34,211,238,.4); }
  .hint { font-size: .82rem; color: #8b95ad; margin: .35rem 0 1rem; }
  .subnote { font-size: .73rem; color: #5e6b8a; margin-top: .6rem; }
  .link { color: #38bdf8; text-decoration: none; } .link:hover { text-decoration: underline; }

  .field { width: 100%; padding: .55rem .8rem; border: 1px solid #2a3350; border-radius: 9px;
    background: #0c1120; color: #e2e8f0; font-family: inherit; font-size: .86rem; }
  .field:focus { outline: none; border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34,211,238,.15); }
  .mono { font-family: 'SFMono-Regular', Consolas, monospace; font-size: .82rem; }
  select.field { cursor: pointer; }
  .req { resize: vertical; line-height: 1.5; }

  .key-card { border-color: #283a63; background: linear-gradient(180deg, rgba(20,28,50,.9), rgba(16,22,40,.85)); }
  .key-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: .9rem; }
  .badge-free { font-size: .68rem; font-weight: 700; color: #34d399; border: 1px solid #1f5f48;
    background: rgba(52,211,153,.1); border-radius: 10px; padding: .2rem .6rem; white-space: nowrap; }
  .key-row, .model-row { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
  .model-row { margin-top: .7rem; }
  .model-label { font-size: .82rem; font-weight: 600; color: #aeb8d0; }
  .key-row .field, .model-row select { flex: 1; min-width: 220px; }
  .ok { font-size: .8rem; font-weight: 600; color: #34d399; }

  .btn { display: inline-flex; align-items: center; gap: .4rem; padding: .55rem 1.1rem; border: none;
    border-radius: 9px; font-size: .86rem; font-weight: 600; cursor: pointer; white-space: nowrap;
    transition: transform .1s, box-shadow .15s, background .15s, opacity .15s; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn:not(:disabled):active { transform: translateY(1px); }
  .btn-primary { background: #1f2a44; color: #dbe4f5; border: 1px solid #2f3c5e; }
  .btn-primary:not(:disabled):hover { background: #283655; }
  .btn-accent { color: #04141a; background: linear-gradient(135deg, #22d3ee, #38bdf8);
    box-shadow: 0 0 18px -2px rgba(34,211,238,.5); }
  .btn-accent:not(:disabled):hover { box-shadow: 0 0 26px 0 rgba(34,211,238,.65); }
  .btn-lg { padding: .65rem 1.4rem; font-size: .92rem; }
  .btn-ghost { background: transparent; color: #aeb8d0; border: 1px solid #2a3350; font-size: .8rem; padding: .5rem .8rem; }
  .btn-ghost:hover { background: #131a2c; }
  .btn-sign { background: rgba(52,211,153,.12); color: #6ee7b7; border: 1px solid #1f6f52; }
  .btn-sign:hover { background: rgba(52,211,153,.2); }

  .upload-row { display: flex; gap: .75rem; flex-wrap: wrap; align-items: center; margin-bottom: 1rem; }
  .file-picker { position: relative; cursor: pointer; }
  .file-picker input[type='file'] { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .file-name { display: block; padding: .5rem 1rem; background: #0c1120; border: 1.5px dashed #2f3c5e;
    border-radius: 8px; font-size: .84rem; color: #aeb8d0; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-picker:hover .file-name { border-color: #22d3ee; }
  .doc-list { list-style: none; display: flex; flex-direction: column; gap: .35rem; }
  .doc-list li { display: flex; align-items: center; gap: .5rem; padding: .45rem .75rem; background: #0c1120;
    border: 1px solid #1e2740; border-radius: 8px; font-size: .84rem; }
  .doc-name { font-weight: 500; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-meta { font-size: .74rem; color: #6ee7b7; white-space: nowrap; }
  .empty { font-size: .82rem; color: #5e6b8a; font-style: italic; }

  .gen-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-top: .75rem; }
  .verdict { font-size: .82rem; font-weight: 600; padding: .3rem .75rem; border-radius: 20px; }
  .v-pass { background: rgba(52,211,153,.12); color: #6ee7b7; }
  .v-warn { background: rgba(251,191,36,.12); color: #fcd34d; }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { width: 13px; height: 13px; border: 2px solid rgba(4,20,26,.4); border-top-color: #04141a;
    border-radius: 50%; animation: spin .65s linear infinite; }
  @keyframes dotPulse { 0%,100% { opacity: .45; } 50% { opacity: 1; } }
  @keyframes glow { 0%,100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); } 50% { box-shadow: 0 0 18px 1px rgba(34,211,238,.4); } }

  .error { background: rgba(248,113,113,.1); color: #fca5a5; padding: .75rem 1rem; border-radius: 10px;
    font-size: .88rem; border: 1px solid #5b2230; }

  /* Pipeline */
  .pipeline { border-color: #243457; }
  .pipe-head { display: flex; align-items: center; justify-content: space-between; gap: .5rem; flex-wrap: wrap; }
  .live-tag { font-size: .6rem; text-transform: uppercase; letter-spacing: .1em; color: #22d3ee;
    border: 1px solid #1f5b6b; border-radius: 6px; padding: .1rem .4rem; }
  .pipe-meta { font-size: .76rem; color: #6ee7b7; font-family: monospace; }
  .nodes { display: flex; align-items: stretch; gap: 0; margin: 1rem 0 .3rem; overflow-x: auto; padding-bottom: .3rem; }
  .node { flex: 1; min-width: 110px; background: #0c1120; border: 1.5px solid #243150; border-radius: 12px;
    padding: .7rem .6rem; display: flex; flex-direction: column; align-items: flex-start; gap: .15rem;
    cursor: default; color: #6e7b99; transition: all .25s; text-align: left; }
  .node-dot { width: 9px; height: 9px; border-radius: 50%; background: #394a6b; }
  .node-label { font-size: .82rem; font-weight: 700; color: #8b95ad; }
  .node-desc { font-size: .68rem; color: #5e6b8a; }
  .node.running { border-color: #22d3ee; animation: glow 1.4s infinite; }
  .node.running .node-dot { background: #22d3ee; box-shadow: 0 0 8px #22d3ee; animation: dotPulse 1s infinite; }
  .node.running .node-label { color: #67e8f9; }
  .node.done { border-color: #1f6f52; cursor: pointer; }
  .node.done .node-dot { background: #34d399; box-shadow: 0 0 6px #34d399; }
  .node.done .node-label { color: #d7e3f4; }
  .node.done:hover { border-color: #34d399; background: #0e1626; }
  .node.open { border-color: #38bdf8; background: #0e1830; }
  .connector { align-self: center; width: 26px; height: 2px; background: #243150; flex-shrink: 0; }
  .connector.lit { background: linear-gradient(90deg, #34d399, #22d3ee); }
  .pipe-hint { font-size: .73rem; color: #5e6b8a; margin-top: .25rem; }

  .detail { margin-top: 1rem; border-top: 1px solid #1e2740; padding-top: 1rem; }
  .detail-label { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
    color: #7c89a8; margin: .6rem 0 .4rem; }
  .detail-label:first-child { margin-top: 0; }
  .mono-block { font-family: 'SFMono-Regular', Consolas, monospace; font-size: .8rem; color: #cdd8ec;
    background: #0c1120; border: 1px solid #1e2740; border-radius: 8px; padding: .7rem .85rem; line-height: 1.55; }
  .chips { display: flex; flex-wrap: wrap; gap: .35rem; }
  .chip { font-family: monospace; font-size: .73rem; color: #93c5fd; background: rgba(56,189,248,.08);
    border: 1px solid #244a6b; border-radius: 6px; padding: .2rem .5rem; }
  .chunk { background: #0c1120; border: 1px solid #1e2740; border-radius: 8px; padding: .55rem .7rem; margin-bottom: .45rem; }
  .chunk-src { font-family: monospace; font-size: .72rem; color: #6ee7b7; margin-bottom: .25rem; }
  .chunk-text { font-size: .8rem; color: #aeb8d0; }
  .draft-list { padding-left: 1.1rem; display: flex; flex-direction: column; gap: .25rem; }
  .draft-list li { font-size: .84rem; color: #cdd8ec; }
  .verdict-detail { font-size: .86rem; font-weight: 600; padding: .55rem .8rem; border-radius: 8px; }
  .verdict-detail.ok { background: rgba(52,211,153,.1); color: #6ee7b7; border: 1px solid #1f6f52; }
  .verdict-detail.warn { background: rgba(251,191,36,.1); color: #fcd34d; border: 1px solid #6b5320; }

  /* Results */
  .results { display: flex; flex-direction: column; gap: 1rem; }
  .results-header { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
  .count { background: linear-gradient(135deg,#22d3ee,#38bdf8); color: #04141a; border-radius: 10px;
    padding: .1rem .5rem; font-size: .72rem; font-weight: 800; }
  .progress { font-size: .8rem; color: #6e7b99; margin-left: auto; }
  .all-signed { color: #6ee7b7; font-weight: 600; }
  .tc { background: rgba(18,24,41,.85); border: 1.5px solid #232b44; border-radius: 14px; padding: 1.25rem; transition: border-color .3s, box-shadow .3s; }
  .tc-signed { border-color: #1f6f52; box-shadow: 0 0 20px -6px rgba(52,211,153,.4); }
  .tc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }
  .tc-head h3 { font-size: .93rem; font-weight: 650; color: #f1f5f9; line-height: 1.3; }
  .priority { font-size: .68rem; font-weight: 800; color: #04141a; padding: .22rem .6rem; border-radius: 10px;
    flex-shrink: 0; letter-spacing: .03em; background: var(--p); }
  .tc-field { margin-bottom: .85rem; }
  .lbl { display: block; font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #6e7b99; margin-bottom: .3rem; }
  .tc-field p { font-size: .86rem; color: #c2cce0; }
  .tc-field ol { padding-left: 1.25rem; display: flex; flex-direction: column; gap: .35rem; }
  .tc-field ol li { font-size: .85rem; }
  .action { color: #c2cce0; } .arrow { color: #46557a; margin: 0 .3rem; } .expected { color: #7dd3fc; font-style: italic; }
  .citations { border-top: 1px solid #1e2740; padding-top: .75rem; }
  .cite-list { display: flex; flex-wrap: wrap; gap: .3rem; }
  .cite { background: rgba(52,211,153,.1); color: #6ee7b7; border: 1px solid #1f5f48; border-radius: 4px;
    padding: .15rem .45rem; font-size: .72rem; font-family: monospace; }
  .tc-foot { display: flex; align-items: center; gap: .75rem; border-top: 1px solid #1e2740; padding-top: .85rem; }
  .signed-label { font-size: .88rem; font-weight: 600; color: #6ee7b7; }
  .sign-hint { font-size: .74rem; color: #46557a; }

  /* Export */
  .export { border-color: #1f6f52; background: linear-gradient(180deg, rgba(16,38,32,.6), rgba(16,22,40,.85)); }
  .summary { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem .9rem; margin: 1rem 0 1.2rem; }
  .summary > div { display: flex; flex-direction: column; gap: .15rem; }
  .summary .full { grid-column: 1 / -1; }
  .summary span { font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; color: #6e7b99; }
  .summary b { font-size: .85rem; color: #dbe7d8; font-weight: 600; word-break: break-word; }
  .export-actions { display: flex; gap: .7rem; flex-wrap: wrap; }

  footer { position: relative; z-index: 1; text-align: center; padding: 1.5rem; font-size: .74rem; color: #46557a; }

  /* Analysis */
  .analysis { border-color: #283a63; }
  .spinner-dark { border: 2px solid rgba(226,232,240,.25); border-top-color: #e2e8f0; }
  .analyzing { display: flex; align-items: center; gap: .6rem; font-size: .88rem; color: #8b95ad; padding: .6rem 0 .2rem; }
  .guardrail { background: rgba(251,191,36,.07); border: 1px solid #6b5320; border-radius: 12px; padding: 1.1rem; margin-top: .8rem; }
  .guardrail-head { font-size: .95rem; font-weight: 700; color: #fcd34d; margin-bottom: .6rem; }
  .guardrail-topic { font-size: .86rem; color: #e2e8f0; margin-bottom: .4rem; }
  .guardrail-topic b { color: #fbbf24; }
  .guardrail-reason { font-size: .85rem; color: #c2cce0; margin-bottom: .5rem; }
  .guardrail-hint { font-size: .8rem; color: #8b95ad; }
  .match-badge { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; font-size: .86rem;
    color: #6ee7b7; font-weight: 600; margin: .9rem 0 1rem; }
  .match-badge b { color: #a7f3d0; }
  .report { background: #0c1120; border: 1px solid #1e2740; border-radius: 12px; padding: 1.1rem;
    display: flex; flex-direction: column; gap: 1rem; transition: border-color .3s, box-shadow .3s; }
  .report-approved { border-color: #1f6f52; box-shadow: 0 0 20px -6px rgba(52,211,153,.4); }
  .rep-block p { font-size: .87rem; color: #c2cce0; }
  .findings { list-style: none; display: flex; flex-direction: column; gap: .6rem; }
  .findings li { background: #0e1626; border: 1px solid #1e2740; border-radius: 8px; padding: .6rem .75rem; }
  .f-obs { font-size: .86rem; color: #dbe4f5; font-weight: 500; margin-bottom: .25rem; }
  .f-ev { font-size: .8rem; color: #8b95ad; font-style: italic; margin-bottom: .35rem; }
  .caveat p { color: #fcd34d; }
  .report-foot { display: flex; align-items: center; gap: .75rem; margin-top: 1rem; flex-wrap: wrap; }

  @media (max-width: 560px) {
    .summary { grid-template-columns: 1fr; }
    .connector { width: 14px; }
  }
</style>
