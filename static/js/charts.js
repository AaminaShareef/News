/**
 * charts.js — 6 Chart.js v4 renderers
 * Fully defensive — handles missing/empty fields gracefully.
 */

Chart.defaults.color = '#6b86a4';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 10;

var EVENT_COLORS = {
  Geopolitical: '#b78cf7',
  Logistics: '#38c8f0',
  Weather: '#30d999',
  Labour: '#ff9547',
  Economic: '#f06aad'
};

var RISK_COLORS = {
  'HIGH': '#ff3d54',
  'MEDIUM-HIGH': '#ff6b35',
  'MEDIUM': '#f5a623',
  'LOW-MEDIUM': '#3ecf72',
  'LOW': '#00c8ff'
};

var CHART_RISK_ORDER = ['HIGH', 'MEDIUM-HIGH', 'MEDIUM', 'LOW-MEDIUM', 'LOW'];

var TT = {
  backgroundColor: '#0b1525',
  borderColor: 'rgba(255,255,255,0.08)',
  borderWidth: 1,
  padding: 10,
  titleFont: { family: "'DM Sans', sans-serif", size: 11, weight: '600' },
  bodyFont: { family: "'DM Mono', monospace", size: 10 },
  titleColor: '#ddeef8',
  bodyColor: '#8ba5be',
  cornerRadius: 8,
  displayColors: true,
  boxWidth: 8, boxHeight: 8, boxPadding: 4
};

var GRID = 'rgba(255,255,255,0.04)';
var BDR = 'rgba(255,255,255,0.05)';
var TICK = '#4e6882';

var _charts = { bar: null, pie: null, line: null, hbar: null, stacked: null, country: null };

function _sd(key) { if (_charts[key]) { _charts[key].destroy(); _charts[key] = null; } }

function _showEmpty(canvasId, msg) {
  var el = document.getElementById(canvasId);
  if (!el) return;
  var parent = el.parentNode;
  el.style.display = 'none';
  var existing = parent.querySelector('.chart-empty-msg');
  if (!existing) {
    var div = document.createElement('div');
    div.className = 'chart-empty-msg';
    div.style.cssText = 'color:#4e6882;font-size:0.75rem;text-align:center;padding:40px 10px;';
    div.textContent = msg || 'No data available';
    parent.appendChild(div);
  }
}

function _clearEmpty(canvasId) {
  var el = document.getElementById(canvasId);
  if (!el) return;
  el.style.display = '';
  var parent = el.parentNode;
  var msg = parent.querySelector('.chart-empty-msg');
  if (msg) msg.remove();
}

/* ── 1. Event count by type ── */
function renderBarChart(data) {
  _sd('bar');
  _clearEmpty('bar-chart');
  var counts = {};
  data.forEach(function (d) { if (d.event_type) counts[d.event_type] = (counts[d.event_type] || 0) + 1; });
  var labels = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
  if (!labels.length) { _showEmpty('bar-chart', 'No event type data'); return; }
  var values = labels.map(function (l) { return counts[l]; });
  var colors = labels.map(function (l) { return EVENT_COLORS[l] || '#94a3b8'; });
  var ctx = document.getElementById('bar-chart').getContext('2d');
  _charts.bar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels, datasets: [{
        label: 'Articles', data: values,
        backgroundColor: colors.map(function (c) { return c + '28'; }),
        borderColor: colors, borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
        hoverBackgroundColor: colors.map(function (c) { return c + '55'; })
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { display: false }, tooltip: Object.assign({}, TT, {
          callbacks: {
            title: function (c) { return c[0].label; },
            label: function (c) { return '  ' + c.parsed.y + ' article' + (c.parsed.y !== 1 ? 's' : ''); }
          }
        })
      },
      scales: {
        x: { grid: { color: GRID }, ticks: { color: TICK }, border: { color: BDR } },
        y: {
          grid: { color: GRID }, ticks: { color: TICK, stepSize: 1 }, border: { color: BDR },
          beginAtZero: true, title: { display: true, text: 'Article Count', color: TICK, font: { size: 9 } }
        }
      }
    }
  });
}

/* ── 2. Risk level doughnut ── */
function renderPieChart(data) {
  _sd('pie');
  _clearEmpty('pie-chart');
  var counts = {};
  data.forEach(function (d) {
    var rl = (d.risk_level || '').trim();
    if (rl) counts[rl] = (counts[rl] || 0) + 1;
  });
  var labels = CHART_RISK_ORDER.filter(function (r) { return counts[r]; });
  if (!labels.length) {
    // Fallback: use whatever keys exist
    labels = Object.keys(counts);
  }
  if (!labels.length) { _showEmpty('pie-chart', 'No risk level data'); return; }
  var values = labels.map(function (l) { return counts[l]; });
  var colors = labels.map(function (l) { return RISK_COLORS[l] || '#94a3b8'; });
  var total = values.reduce(function (a, b) { return a + b; }, 0);
  var ctx = document.getElementById('pie-chart').getContext('2d');
  _charts.pie = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels, datasets: [{
        data: values,
        backgroundColor: colors.map(function (c) { return c + '44'; }),
        borderColor: colors, borderWidth: 1.5,
        hoverBorderWidth: 2.5, hoverBackgroundColor: colors.map(function (c) { return c + '77'; })
      }]
    },
    options: {
      responsive: true, cutout: '66%',
      plugins: {
        legend: {
          position: 'bottom', labels: {
            boxWidth: 8, boxHeight: 8, padding: 10, color: '#6b86a4',
            font: { family: "'DM Mono', monospace", size: 9 },
            generateLabels: function (chart) {
              var ds = chart.data.datasets[0];
              return chart.data.labels.map(function (label, i) {
                return {
                  text: label + '  ' + Math.round(ds.data[i] / total * 100) + '%',
                  fillStyle: ds.backgroundColor[i], strokeStyle: ds.borderColor[i], lineWidth: 1, index: i
                };
              });
            }
          }
        },
        tooltip: Object.assign({}, TT, {
          callbacks: {
            label: function (c) { return '  ' + c.label + ': ' + c.parsed + ' (' + Math.round(c.parsed / total * 100) + '%)'; }
          }
        })
      }
    }
  });
}

/* ── 3. Risk score trend + rolling avg ── */
function renderLineChart(data) {
  _sd('line');
  _clearEmpty('line-chart');
  if (!data.length) { _showEmpty('line-chart', 'No trend data'); return; }
  var sorted = data.slice();
  if (sorted[0] && sorted[0].published_at) {
    sorted.sort(function (a, b) { return new Date(a.published_at) - new Date(b.published_at); });
  }
  var labels = sorted.map(function (d, i) {
    return d.published_at
      ? new Date(d.published_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
      : '#' + (i + 1);
  });
  var scores = sorted.map(function (d) { return parseFloat(d.risk_score) || 0; });
  var trend = scores.map(function (_, i) {
    var sl = scores.slice(Math.max(0, i - 1), i + 2);
    return parseFloat((sl.reduce(function (a, b) { return a + b; }, 0) / sl.length).toFixed(3));
  });
  var ctx = document.getElementById('line-chart').getContext('2d');
  var grad = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0, 'rgba(0,200,255,0.28)');
  grad.addColorStop(1, 'rgba(0,200,255,0.01)');
  _charts.line = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels, datasets: [
        {
          label: 'Risk Score', data: scores, borderColor: '#00c8ff', backgroundColor: grad,
          borderWidth: 1.5, pointRadius: 2.5, pointBackgroundColor: '#00c8ff',
          pointHoverRadius: 5, tension: 0.4, fill: true, order: 2
        },
        {
          label: '3-pt Avg', data: trend, borderColor: '#f5a623', backgroundColor: 'transparent',
          borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, pointHoverRadius: 4,
          tension: 0.4, fill: false, order: 1
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true, interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true, position: 'top', align: 'end',
          labels: {
            boxWidth: 20, boxHeight: 2, padding: 12, color: '#6b86a4',
            usePointStyle: false, font: { family: "'DM Mono', monospace", size: 9 }
          }
        },
        tooltip: Object.assign({}, TT, {
          callbacks: {
            label: function (c) { return '  ' + c.dataset.label + ': ' + c.parsed.y.toFixed(2); }
          }
        })
      },
      scales: {
        x: { grid: { color: GRID }, ticks: { color: TICK, maxTicksLimit: 8 }, border: { color: BDR } },
        y: {
          min: 0, max: 1, grid: { color: GRID },
          ticks: { color: TICK, stepSize: 0.2, callback: function (v) { return v.toFixed(1); } },
          border: { color: BDR },
          title: { display: true, text: 'Risk Score (0–1)', color: TICK, font: { size: 9 } }
        }
      }
    }
  });
}

/* ── 4. Avg risk score per event type (horizontal) ── */
function renderAvgRiskChart(data) {
  _sd('hbar');
  _clearEmpty('hbar-chart');
  var sums = {}, cnts = {};
  data.forEach(function (d) {
    if (!d.event_type) return;
    sums[d.event_type] = (sums[d.event_type] || 0) + (parseFloat(d.risk_score) || 0);
    cnts[d.event_type] = (cnts[d.event_type] || 0) + 1;
  });
  var labels = Object.keys(sums).sort(function (a, b) { return (sums[b] / cnts[b]) - (sums[a] / cnts[a]); });
  if (!labels.length) { _showEmpty('hbar-chart', 'No data'); return; }
  var avgs = labels.map(function (l) { return parseFloat((sums[l] / cnts[l]).toFixed(3)); });
  var colors = labels.map(function (l) { return EVENT_COLORS[l] || '#94a3b8'; });
  var ctx = document.getElementById('hbar-chart').getContext('2d');
  _charts.hbar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels, datasets: [{
        label: 'Avg Risk Score', data: avgs,
        backgroundColor: colors.map(function (c) { return c + '30'; }),
        borderColor: colors, borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
        hoverBackgroundColor: colors.map(function (c) { return c + '60'; })
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, TT, {
          callbacks: {
            title: function (c) { return c[0].label; },
            label: function (c) {
              return ['  Avg: ' + c.parsed.x.toFixed(2),
              '  From ' + cnts[c.label] + ' article' + (cnts[c.label] !== 1 ? 's' : '')];
            }
          }
        })
      },
      scales: {
        x: {
          min: 0, max: 1, grid: { color: GRID },
          ticks: { color: TICK, callback: function (v) { return v.toFixed(1); } },
          border: { color: BDR },
          title: { display: true, text: 'Average Risk Score', color: TICK, font: { size: 9 } }
        },
        y: { grid: { color: 'transparent' }, ticks: { color: '#ddeef8', font: { size: 10 } }, border: { color: BDR } }
      }
    }
  });
}

/* ── 5. Risk composition per event type (stacked) ── */
function renderStackedChart(data) {
  _sd('stacked');
  _clearEmpty('stacked-chart');
  var eventTypes = [];
  data.forEach(function (d) {
    if (d.event_type && eventTypes.indexOf(d.event_type) === -1) eventTypes.push(d.event_type);
  });
  var riskLevels = CHART_RISK_ORDER.filter(function (r) { return data.some(function (d) { return d.risk_level === r; }); });
  if (!eventTypes.length || !riskLevels.length) { _showEmpty('stacked-chart', 'No composition data'); return; }
  var matrix = {};
  riskLevels.forEach(function (r) { matrix[r] = {}; eventTypes.forEach(function (et) { matrix[r][et] = 0; }); });
  data.forEach(function (d) {
    if (d.event_type && d.risk_level && matrix[d.risk_level]) matrix[d.risk_level][d.event_type]++;
  });
  var datasets = riskLevels.map(function (r) {
    return {
      label: r, data: eventTypes.map(function (et) { return matrix[r][et]; }),
      backgroundColor: (RISK_COLORS[r] || '#94a3b8') + '88',
      borderColor: RISK_COLORS[r] || '#94a3b8', borderWidth: 1,
      borderRadius: 3, borderSkipped: false
    };
  });
  var ctx = document.getElementById('stacked-chart').getContext('2d');
  _charts.stacked = new Chart(ctx, {
    type: 'bar',
    data: { labels: eventTypes, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: true, interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true, position: 'top', align: 'end',
          labels: {
            boxWidth: 8, boxHeight: 8, padding: 10, color: '#6b86a4',
            font: { family: "'DM Mono', monospace", size: 9 }
          }
        },
        tooltip: Object.assign({}, TT, {
          callbacks: {
            title: function (c) { return 'Event Type: ' + c[0].label; },
            label: function (c) { return '  ' + c.dataset.label + ': ' + c.parsed.y + ' article' + (c.parsed.y !== 1 ? 's' : ''); }
          }
        })
      },
      scales: {
        x: { stacked: true, grid: { color: GRID }, ticks: { color: TICK }, border: { color: BDR } },
        y: {
          stacked: true, grid: { color: GRID }, ticks: { color: TICK, stepSize: 1 },
          border: { color: BDR }, beginAtZero: true,
          title: { display: true, text: 'Article Count', color: TICK, font: { size: 9 } }
        }
      }
    }
  });
}

/* ── 6. Top 8 countries by event count (horizontal) ── */
function renderCountryChart(data) {
  _sd('country');
  _clearEmpty('country-chart');
  var counts = {}, riskMap = {};
  data.forEach(function (d) {
    var key = d.country || d.region || 'Global';
    counts[key] = (counts[key] || 0) + 1;
    if (!riskMap[key]) riskMap[key] = {};
    if (d.risk_level) riskMap[key][d.risk_level] = (riskMap[key][d.risk_level] || 0) + 1;
  });
  var sorted = Object.entries(counts).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 8);
  if (!sorted.length) { _showEmpty('country-chart', 'No geographic data'); return; }
  var labels = sorted.map(function (e) { return e[0]; });
  var values = sorted.map(function (e) { return e[1]; });
  var colors = labels.map(function (l) {
    if (!riskMap[l] || !Object.keys(riskMap[l]).length) return '#94a3b8';
    var dom = Object.entries(riskMap[l]).sort(function (a, b) { return b[1] - a[1]; })[0][0];
    return RISK_COLORS[dom] || '#94a3b8';
  });
  var ctx = document.getElementById('country-chart').getContext('2d');
  _charts.country = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels, datasets: [{
        label: 'Events', data: values,
        backgroundColor: colors.map(function (c) { return c + '30'; }),
        borderColor: colors, borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
        hoverBackgroundColor: colors.map(function (c) { return c + '60'; })
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, TT, {
          callbacks: {
            title: function (c) { return c[0].label; },
            label: function (c) {
              var rm = riskMap[c.label];
              var dom = rm && Object.keys(rm).length
                ? Object.entries(rm).sort(function (a, b) { return b[1] - a[1]; })[0][0] : '—';
              return ['  ' + c.parsed.x + ' article' + (c.parsed.x !== 1 ? 's' : ''),
              '  Dominant risk: ' + dom];
            }
          }
        })
      },
      scales: {
        x: {
          grid: { color: GRID }, ticks: { color: TICK, stepSize: 1 },
          border: { color: BDR }, beginAtZero: true,
          title: { display: true, text: 'Article Count', color: TICK, font: { size: 9 } }
        },
        y: { grid: { color: 'transparent' }, ticks: { color: '#ddeef8', font: { size: 10 } }, border: { color: BDR } }
      }
    }
  });
}