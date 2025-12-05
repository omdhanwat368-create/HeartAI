// static/js/main.js (shows calibrated metrics + original model table)
document.addEventListener('DOMContentLoaded', () => {
  // load metrics and render them nicely
  fetch('/api/metrics').then(r => r.json()).then(j => {
    const metricsDiv = document.getElementById('metrics');
    if (!metricsDiv) return;

    if (j.error) {
      metricsDiv.innerHTML = '<p class="text-muted">Metrics not available. Train the model to generate metrics.json.</p>';
      return;
    }

    // Build summary header (best model + calibrated metrics if present)
    let headerHtml = `<div class="mb-2"><strong>Best model:</strong> ${j.best_model || '-'} </div>`;

    // Show calibrated metrics when available (accept several possible key names)
    const calibAuc = j.calibrated_test_auc ?? j.calibrated_auc ?? j.calibrated_test_roc_auc ?? null;
    const calibAcc = j.calibrated_test_accuracy ?? j.calibrated_accuracy ?? null;
    if (calibAuc !== null || calibAcc !== null) {
      headerHtml += '<div class="mb-2">';
      if (calibAuc !== null) headerHtml += `<div><strong>Calibrated AUC:</strong> ${Number(calibAuc).toFixed(3)}</div>`;
      if (calibAcc !== null) headerHtml += `<div><strong>Calibrated Accuracy:</strong> ${Number(calibAcc).toFixed(3)}</div>`;
      headerHtml += '</div>';
    }

    // If there is also an uncalibrated best_score_test_roc_auc, show it
    if (j.best_score_test_roc_auc !== undefined) {
      headerHtml += `<div class="mb-2 text-muted"><small>Raw best-test AUC: ${Number(j.best_score_test_roc_auc).toFixed(3)}</small></div>`;
    }

    // Build per-model list (unchanged)
    let modelsHtml = '<div class="card"><div class="card-body p-2">';
    modelsHtml += '<h6 class="card-title mb-2">Model comparison</h6>';
    modelsHtml += '<ul class="list-group list-group-flush">';
    if (j.models) {
      for (const m in j.models) {
        const auc = j.models[m].roc_auc ? Number(j.models[m].roc_auc).toFixed(3) : '-';
        const acc = j.models[m].accuracy ? Number(j.models[m].accuracy).toFixed(3) : '-';
        modelsHtml += `<li class="list-group-item py-1">${m}: AUC ${auc}, acc ${acc}</li>`;
      }
    } else {
      modelsHtml += '<li class="list-group-item py-1 text-muted">No per-model metrics available</li>';
    }
    modelsHtml += '</ul></div></div>';

    metricsDiv.innerHTML = `<div>${headerHtml}</div><div class="mt-2">${modelsHtml}</div>`;
  }).catch((err) => {
    // gracefully fail
    const metricsDiv = document.getElementById('metrics');
    if (metricsDiv) metricsDiv.innerHTML = '<p class="text-muted">Could not load metrics.</p>';
    console.error('Failed to load /api/metrics', err);
  });

  // -----------------------------
  // existing predict logic below
  // -----------------------------
  function safeDestroyChart() {
    try {
      if (window.probChart && typeof window.probChart.destroy === 'function') {
        window.probChart.destroy();
      }
    } catch (err) {
      window.probChart = null;
    }
  }

  // helper: read JSON safely (return null if not JSON)
  async function safeReadJson(response) {
    const txt = await response.text();
    try { return JSON.parse(txt); } catch (e) { return null; }
  }

  const form = document.getElementById('predict-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const dataEntries = Object.fromEntries(new FormData(form).entries());
    // safer numeric casting: only convert fields that look numeric
    for (let k in dataEntries) {
      const v = dataEntries[k];
      const n = Number(v);
      dataEntries[k] = (v === '' || Number.isNaN(n)) ? 0 : n;
    }

    let json;
    try {
      // NOTE: we post to /api/predict (server handler). If your backend uses /predict instead,
      // change the URL to '/predict' here — keep consistent with your app.py.
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(dataEntries)
      });

      // If server returned non-OK, try to extract JSON or text for a helpful error message
      if (!response.ok) {
        const body = await safeReadJson(response) || await response.text();
        const msg = body && body.error ? body.error : (typeof body === 'string' ? body : `Server returned ${response.status}`);
        throw new Error(msg);
      }

      // parse JSON or throw if invalid
      json = await response.json();

    } catch (err) {
      console.error('fetch error', err);
      // show detailed message to user if possible
      alert('Request failed — ' + (err.message || 'check backend or console for details'));
      return;
    }

    // display the result card
    const card = document.getElementById('resultCard');
    const resultText = document.getElementById('resultText');
    if (card) card.style.display = 'block';

    // Show probability and threshold explicitly if present
    const probPct = (json.probability * 100).toFixed(1);
    const threshPct = json.threshold !== undefined ? (json.threshold * 100).toFixed(0) : null;

    if (json.prediction === 1) {
      resultText.className = 'alert alert-danger';
      resultText.innerText = `High risk — probability ${probPct}%` + (threshPct ? ` (threshold ${threshPct}%)` : '');
    } else {
      resultText.className = 'alert alert-success';
      resultText.innerText = `Low risk — probability ${probPct}%` + (threshPct ? ` (threshold ${threshPct}%)` : '');
    }

    // chart
    safeDestroyChart();
    const ctx = document.getElementById('probChart').getContext('2d');
    window.probChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Risk', 'Safe'],
        datasets: [{ data: [json.probability, 1 - json.probability] }]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom' } }
      }
    });
  });
});
