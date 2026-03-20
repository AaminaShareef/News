/**
 * app.js — ProcureGuard dashboard orchestration
 * Handles: API calls, rendering, filters, pagination, export, alerts.
 */

'use strict';

var STATE = {
    allResults: [],
    filtered: [],
    currentPage: 1,
    pageSize: 12
};

function $(id) { return document.getElementById(id); }
function $$(sel) { return document.querySelectorAll(sel); }

function showSpinner(msg) {
    $('spinner-overlay').classList.add('active');
    $('spinner-text').textContent = msg || 'Analysing news data…';
}
function hideSpinner() { $('spinner-overlay').classList.remove('active'); }

function showToast(msg, type) {
    var t = $('toast');
    t.textContent = msg;
    t.className = (type === 'error') ? 'error show' : 'show';
    setTimeout(function() { t.classList.remove('show'); }, 3500);
}

var RISK_ORDER = { 'HIGH': 5, 'MEDIUM-HIGH': 4, 'MEDIUM': 3, 'LOW-MEDIUM': 2, 'LOW': 1 };

function updateKPIs(data) {
    $('kpi-total').textContent = data.length;
    $('kpi-high').textContent = data.filter(function(d) { return d.risk_level === 'HIGH'; }).length;
    $('kpi-medium').textContent = data.filter(function(d) {
        return d.risk_level === 'MEDIUM' || d.risk_level === 'MEDIUM-HIGH';
    }).length;
    $('kpi-low').textContent = data.filter(function(d) {
        return d.risk_level === 'LOW' || d.risk_level === 'LOW-MEDIUM';
    }).length;

    var evCounts = {};
    data.forEach(function(d) {
        if (d.event_type) evCounts[d.event_type] = (evCounts[d.event_type] || 0) + 1;
    });
    var entries = Object.entries(evCounts);
    var dom = entries.length ? entries.sort(function(a, b) { return b[1] - a[1]; })[0] : null;
    $('kpi-event').textContent = dom ? dom[0] : '—';
}

function checkHighRiskAlert(data) {
    var highCount = data.filter(function(d) { return d.risk_level === 'HIGH'; }).length;
    var banner = $('alert-banner');
    if (highCount > 0) {
        $('alert-count').textContent = highCount;
        banner.classList.add('visible');
    } else {
        banner.classList.remove('visible');
    }
}

function populateFilters(data) {
    var sets = { country: new Set(), region: new Set(), event_type: new Set(), risk_level: new Set() };
    data.forEach(function(d) {
        if (d.country) sets.country.add(d.country);
        if (d.region) sets.region.add(d.region);
        if (d.event_type) sets.event_type.add(d.event_type);
        if (d.risk_level) sets.risk_level.add(d.risk_level);
    });
    _buildFilterGroup('filter-country', Array.from(sets.country).sort(), 'country', data);
    _buildFilterGroup('filter-region', Array.from(sets.region).sort(), 'region', data);
    _buildFilterGroup('filter-event-type', Array.from(sets.event_type).sort(), 'event_type', data);
    _buildFilterGroup('filter-risk-level',
        Array.from(sets.risk_level).sort(function(a, b) { return (RISK_ORDER[b] || 0) - (RISK_ORDER[a] || 0); }),
        'risk_level', data);
}

function _buildFilterGroup(containerId, options, field, data) {
    var container = $(containerId);
    if (!container) return;
    container.innerHTML = '';
    options.forEach(function(opt) {
        var count = data.filter(function(d) { return d[field] === opt; }).length;
        var uid = 'chk-' + containerId + '-' + opt.replace(/\s+/g, '-');
        var label = document.createElement('label');
        label.className = 'filter-option';
        label.htmlFor = uid;
        label.innerHTML =
            '<input type="checkbox" id="' + uid + '" data-field="' + field + '" data-value="' + opt + '">' +
            '<span class="opt-label">' + opt + '</span>' +
            '<span class="opt-count">' + count + '</span>';
        container.appendChild(label);
        label.querySelector('input').addEventListener('change', applyFilters);
    });
}

function applyFilters() {
    var active = {};
    $$('.filter-option input[type=checkbox]:checked').forEach(function(chk) {
        var f = chk.dataset.field;
        if (!active[f]) active[f] = new Set();
        active[f].add(chk.dataset.value);
    });
    STATE.filtered = STATE.allResults.filter(function(d) {
        for (var field in active) {
            if (!active[field].has(d[field])) return false;
        }
        return true;
    });
    STATE.currentPage = 1;
    renderPage();
    renderCharts(STATE.filtered.length ? STATE.filtered : STATE.allResults);
}

function clearFilters() {
    $$('.filter-option input[type=checkbox]').forEach(function(c) { c.checked = false; });
    STATE.filtered = STATE.allResults.slice();
    STATE.currentPage = 1;
    renderPage();
    renderCharts(STATE.allResults);
}

function renderPage() {
    var data = STATE.filtered;
    var total = data.length;
    var totalPages = Math.max(1, Math.ceil(total / STATE.pageSize));
    STATE.currentPage = Math.min(STATE.currentPage, totalPages);
    var start = (STATE.currentPage - 1) * STATE.pageSize;
    var slice = data.slice(start, start + STATE.pageSize);
    var grid = $('news-grid');
    grid.innerHTML = '';
    $('results-count').textContent = total + ' article' + (total !== 1 ? 's' : '');
    if (!slice.length) {
        grid.innerHTML = '<div class="empty-state"><div class="es-icon">🔍</div><p>No articles match the current filters.</p></div>';
        $('pagination').innerHTML = '';
        return;
    }
    slice.forEach(function(d) { grid.appendChild(buildCard(d)); });
    buildPagination(totalPages);
}

function buildCard(d) {
    var el = document.createElement('article');
    el.className = 'news-card';
    var rl = d.risk_level || 'LOW';
    var et = d.event_type || '—';
    var score = parseFloat(d.risk_score) || 0;
    var pct = Math.round(score * 100);
    var titleHtml = d.url
        ? '<a href="' + _esc(d.url) + '" target="_blank" rel="noopener">' + _esc(d.title || '—') + '</a>'
        : _esc(d.title || '—');
    var dateStr = d.published_at
        ? new Date(d.published_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
        : '';
    el.innerHTML =
        '<div class="risk-stripe ' + rl + '"></div>' +
        '<div class="card-top">' +
        '<div class="card-title">' + titleHtml + '</div>' +
        '<span class="badge badge-' + rl + '">' + rl + '</span>' +
        '</div>' +
        '<div class="card-meta">' +
        '<span class="badge badge-etype ' + et + '">' + et + '</span>' +
        (d.source ? '<span class="meta-item"><span class="mi-icon">📰</span>' + _esc(d.source) + '</span>' : '') +
        (d.country ? '<span class="meta-item"><span class="mi-icon">🌍</span>' + _esc(d.country) + '</span>' : '') +
        (d.region ? '<span class="meta-item"><span class="mi-icon">🗺️</span>' + _esc(d.region) + '</span>' : '') +
        (dateStr ? '<span class="meta-item"><span class="mi-icon">📅</span>' + dateStr + '</span>' : '') +
        '</div>' +
        '<div class="score-row">' +
        '<div class="score-bar-wrap"><div class="score-bar-fill ' + rl + '" style="width:' + pct + '%"></div></div>' +
        '<span class="score-val">' + score.toFixed(2) + '</span>' +
        '</div>';
    return el;
}

function _esc(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function buildPagination(totalPages) {
    var pg = $('pagination');
    pg.innerHTML = '';
    if (totalPages <= 1) return;
    var prev = document.createElement('button');
    prev.className = 'page-btn';
    prev.innerHTML = '&lsaquo; Prev';
    prev.disabled = STATE.currentPage === 1;
    prev.onclick = function() { STATE.currentPage--; renderPage(); };
    pg.appendChild(prev);
    var maxBtns = 7;
    var start = Math.max(1, STATE.currentPage - 3);
    var end = Math.min(totalPages, start + maxBtns - 1);
    if (end - start < maxBtns - 1) start = Math.max(1, end - maxBtns + 1);
    if (start > 1) { pg.appendChild(_pageBtn(1)); if (start > 2) pg.appendChild(_ellipsis()); }
    for (var i = start; i <= end; i++) pg.appendChild(_pageBtn(i));
    if (end < totalPages) { if (end < totalPages - 1) pg.appendChild(_ellipsis()); pg.appendChild(_pageBtn(totalPages)); }
    var next = document.createElement('button');
    next.className = 'page-btn';
    next.innerHTML = 'Next &rsaquo;';
    next.disabled = STATE.currentPage === totalPages;
    next.onclick = function() { STATE.currentPage++; renderPage(); };
    pg.appendChild(next);
    var info = document.createElement('span');
    info.className = 'page-info';
    info.textContent = 'Page ' + STATE.currentPage + ' of ' + totalPages;
    pg.appendChild(info);
}
function _pageBtn(n) {
    var b = document.createElement('button');
    b.className = 'page-btn' + (n === STATE.currentPage ? ' active' : '');
    b.textContent = n;
    b.onclick = function() { STATE.currentPage = n; renderPage(); };
    return b;
}
function _ellipsis() {
    var s = document.createElement('span');
    s.className = 'page-info';
    s.textContent = '…';
    return s;
}

function renderCharts(data) {
    if (typeof renderBarChart     === 'function') renderBarChart(data);
    if (typeof renderPieChart     === 'function') renderPieChart(data);
    if (typeof renderLineChart    === 'function') renderLineChart(data);
    if (typeof renderAvgRiskChart === 'function') renderAvgRiskChart(data);
    if (typeof renderStackedChart === 'function') renderStackedChart(data);
    if (typeof renderCountryChart === 'function') renderCountryChart(data);
    if (typeof renderMap          === 'function') renderMap(data);
}

function handleResults(records) {
    STATE.allResults = records;
    STATE.filtered = records.slice();
    STATE.currentPage = 1;
    populateFilters(records);
    updateKPIs(records);
    checkHighRiskAlert(records);
    renderCharts(records);
    renderPage();
}

async function runAnalysis() {
    var query = $('search-input').value.trim();
    if (!query) { showToast('Please enter a search query.', 'error'); return; }
    showSpinner('Fetching & analysing news…');
    try {
        var res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, page_size: 30 })
        });
        var json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Server error');
        handleResults(json.results || []);
        var n = (json.results || []).length;
        showToast('✅ ' + n + ' relevant article' + (n !== 1 ? 's' : '') + ' found.');
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    } finally {
        hideSpinner();
    }
}

async function uploadCSV(file) {
    showSpinner('Analysing uploaded CSV…');
    try {
        var fd = new FormData();
        fd.append('file', file);
        var res = await fetch('/api/upload', { method: 'POST', body: fd });
        var json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Upload error');
        handleResults(json.results || []);
        var n = (json.results || []).length;
        showToast('✅ CSV processed — ' + n + ' article' + (n !== 1 ? 's' : '') + ' found.');
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    } finally {
        hideSpinner();
    }
}

async function exportCSV() {
    if (!STATE.filtered.length) { showToast('No data to export.', 'error'); return; }
    showSpinner('Preparing export…');
    try {
        var res = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: STATE.filtered })
        });
        if (!res.ok) throw new Error('Export failed');
        var blob = await res.blob();
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'risk_report_' + new Date().toISOString().slice(0, 10) + '.csv';
        a.click();
        URL.revokeObjectURL(url);
        showToast('📥 CSV exported successfully.');
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    } finally {
        hideSpinner();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    $('analyze-btn').addEventListener('click', runAnalysis);
    $('search-input').addEventListener('keydown', function(e) { if (e.key === 'Enter') runAnalysis(); });
    $('upload-btn').addEventListener('click', function() { $('upload-input').click(); });
    $('upload-input').addEventListener('change', function(e) {
        var file = e.target.files[0];
        if (file) { uploadCSV(file); e.target.value = ''; }
    });
    $('export-btn').addEventListener('click', exportCSV);
    $('clear-filters').addEventListener('click', clearFilters);
    $('close-alert').addEventListener('click', function() { $('alert-banner').classList.remove('visible'); });
    $('news-grid').innerHTML =
        '<div class="empty-state"><div class="es-icon">🛰️</div>' +
        '<p>Enter a query above and click <strong>Analyse</strong> to begin<br>' +
        'or upload a CSV file to process offline data.</p></div>';
});