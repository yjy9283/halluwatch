const SVG_NS = "http://www.w3.org/2000/svg";
const PALETTE = ["#1E3A5F", "#3F5A3D", "#A6791F", "#5B4A7A", "#7A2E2E", "#2E5F5A", "#6B5B3A"];

const $ = (id) => document.getElementById(id);

// --- 슬라이더 라벨 실시간 갱신 ---
$("nSamples").addEventListener("input", (e) => ($("nSamplesVal").textContent = e.target.value));
$("temperature").addEventListener("input", (e) => ($("tempVal").textContent = parseFloat(e.target.value).toFixed(1)));
$("threshold").addEventListener("input", (e) => ($("threshVal").textContent = parseFloat(e.target.value).toFixed(2)));

// --- 스코프 배경 그리드(graticule) 그리기 ---
function drawGrid() {
  const svg = $("scopeGrid");
  svg.innerHTML = "";
  for (let x = 0; x <= 800; x += 40) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x); line.setAttribute("y1", 0);
    line.setAttribute("x2", x); line.setAttribute("y2", 200);
    line.setAttribute("stroke", "#D8D0BC"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
  }
  for (let y = 0; y <= 200; y += 25) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("y1", y);
    line.setAttribute("x2", 800); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#D8D0BC"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
  }
  const mid = document.createElementNS(SVG_NS, "line");
  mid.setAttribute("x1", 0); mid.setAttribute("y1", 100);
  mid.setAttribute("x2", 800); mid.setAttribute("y2", 100);
  mid.setAttribute("stroke", "#7A7262"); mid.setAttribute("stroke-width", "1"); mid.setAttribute("opacity", "0.6");
  svg.appendChild(mid);
}
drawGrid();

// --- 대기 상태: 잔잔한 노이즈 파형 애니메이션 ---
let idleAnim = null;
function startIdleWave() {
  const svg = $("scopeTraces");
  let t = 0;
  function tick() {
    t += 0.05;
    let d = "M0,100";
    for (let x = 0; x <= 800; x += 8) {
      const y = 100 + Math.sin(x / 40 + t) * 4;
      d += ` L${x},${y}`;
    }
    svg.innerHTML = `<path d="${d}" stroke="#B8AE8F" stroke-width="1.2" fill="none" opacity="0.7"/>`;
    idleAnim = requestAnimationFrame(tick);
  }
  tick();
}
function stopIdleWave() {
  if (idleAnim) cancelAnimationFrame(idleAnim);
}
startIdleWave();

// --- 결과 파형: 클러스터별 위상/주파수로 "일치 vs 불일치"를 시각화 ---
// 같은 클러스터(같은 의미) = 거의 같은 위상·주파수라서 서로 겹쳐 굵고 밝은 하나의 선처럼 보임
// 다른 클러스터(다른 의미) = 위상/주파수가 어긋나 지지직거리는 간섭무늬로 보임
function renderResultWave(samples, clusterLabels) {
  const svg = $("scopeTraces");
  svg.innerHTML = "";
  const uniqueClusters = [...new Set(clusterLabels)];

  samples.forEach((sample, i) => {
    const clusterIdx = uniqueClusters.indexOf(clusterLabels[i]);
    const color = PALETTE[clusterIdx % PALETTE.length];
    const freq = 30 + clusterIdx * 9;
    const phase = clusterIdx * 1.3 + (Math.random() - 0.5) * 0.15; // 같은 클러스터끼리는 살짝만 흔들림
    let d = "";
    for (let x = 0; x <= 800; x += 4) {
      const y = 100 + Math.sin(x / freq + phase) * 55;
      d += (x === 0 ? "M" : "L") + `${x},${y.toFixed(1)}`;
    }
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("stroke", color);
    path.setAttribute("opacity", "0.75");
    svg.appendChild(path);
  });
}

// --- 클러스터 색상 매핑 (트랜스크립트와 파형 색을 일치시키기 위함) ---
function clusterColor(clusterId, allClusterIds) {
  const idx = allClusterIds.indexOf(clusterId);
  return PALETTE[idx % PALETTE.length];
}

function renderTranscript(samples, clusterLabels) {
  const container = $("transcript");
  container.innerHTML = "";
  const uniqueClusters = [...new Set(clusterLabels)].sort((a, b) => a - b);

  uniqueClusters.forEach((cid) => {
    const color = clusterColor(cid, uniqueClusters);
    const block = document.createElement("div");
    block.className = "cluster-block";
    block.style.borderLeftColor = color;

    const count = clusterLabels.filter((l) => l === cid).length;
    const head = document.createElement("div");
    head.className = "cluster-block-head";
    head.style.color = color;
    head.textContent = `CLUSTER ${cid} · ${count}개 답변`;
    block.appendChild(head);

    samples.forEach((s, i) => {
      if (clusterLabels[i] === cid) {
        const p = document.createElement("div");
        p.className = "cluster-answer";
        p.textContent = s;
        block.appendChild(p);
      }
    });
    container.appendChild(block);
  });
}

// --- 분석 실행 ---
$("analyzeBtn").addEventListener("click", async () => {
  const question = $("question").value.trim();
  if (!question) return;

  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  $("scopeStatus").textContent = "ANALYZING";
  $("readoutRow").style.display = "none";
  $("transcript").innerHTML = "";

  const payload = {
    question,
    n_samples: parseInt($("nSamples").value, 10),
    temperature: parseFloat($("temperature").value),
    uncertainty_threshold: parseFloat($("threshold").value),
  };

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "알 수 없는 오류" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    stopIdleWave();
    renderResultWave(data.samples, data.cluster_labels);
    renderTranscript(data.samples, data.cluster_labels);

    $("roSamples").textContent = data.samples.length;
    $("roClusters").textContent = data.n_clusters;
    $("roEntropy").textContent = data.normalized_entropy_score.toFixed(2);
    $("roStatus").textContent = data.is_uncertain ? "⚠ UNCERTAIN" : "✓ CONFIDENT";

    const statusChip = $("roStatusChip");
    statusChip.classList.toggle("uncertain", data.is_uncertain);
    $("scopeStatus").textContent = data.is_uncertain ? "DIVERGENT" : "CONVERGENT";
    $("readoutRow").style.display = "table";

    $("figureCaption").innerHTML =
      `<b>Figure 1.</b> ${data.samples.length}개 응답이 ${data.n_clusters}개 의미 클러스터로 ` +
      (data.n_clusters <= 1
        ? `수렴했다 (H = ${data.normalized_entropy_score.toFixed(2)}). 모든 곡선이 위상을 공유하며 겹쳐 보인다.`
        : `분산되었다 (H = ${data.normalized_entropy_score.toFixed(2)}). 클러스터 간 위상 차로 간섭무늬가 관찰된다.`);
  } catch (e) {
    $("scopeStatus").textContent = "NO RESPONSE";
    $("transcript").innerHTML = `<div class="cluster-block" style="border-top-color:#7A2E2E;">
      <div class="cluster-block-head" style="color:#7A2E2E;">ERROR</div>
      <div class="cluster-answer">${e.message} — GROQ_API_KEY 설정을 확인하세요.</div>
    </div>`;
    startIdleWave();
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Analysis →";
  }
});
