/**
 * insights.js — AI Risk Insights generator
 * Pure client-side, pattern-based analysis from the results data.
 * Called by app.js after results arrive. Does NOT touch any ML logic.
 */

'use strict';

// ── Risk severity ordering ────────────────────────────────────────────────
const _RISK_WEIGHT = { 'HIGH': 4, 'MEDIUM-HIGH': 3, 'MEDIUM': 2, 'LOW-MEDIUM': 1, 'LOW': 0 };

/**
 * Generate and render the full Insights panel into #insights-body.
 * @param {Array} data — full (unfiltered) results array
 */
function renderInsights(data) {
    const container = document.getElementById('insights-body');
    if (!container) return;

    if (!data || !data.length) {
        container.innerHTML = `
      <div class="insights-placeholder">
        <div class="ip-icon">💡</div>
        <p>Run an analysis to generate AI-powered risk insights and procurement recommendations.</p>
      </div>`;
        return;
    }

    // ── Compute aggregates ──────────────────────────────────────────────────
    const total = data.length;
    const highCount = data.filter(d => d.risk_level === 'HIGH').length;
    const mhCount = data.filter(d => d.risk_level === 'MEDIUM-HIGH').length;
    const medCount = data.filter(d => d.risk_level === 'MEDIUM').length;
    const lowMCount = data.filter(d => d.risk_level === 'LOW-MEDIUM').length;
    const lowCount = data.filter(d => d.risk_level === 'LOW').length;

    const avgScore = data.reduce((s, d) => s + (parseFloat(d.risk_score) || 0), 0) / total;
    const maxScore = Math.max(...data.map(d => parseFloat(d.risk_score) || 0));

    // Event type frequency
    const evFreq = {};
    data.forEach(d => { if (d.event_type) evFreq[d.event_type] = (evFreq[d.event_type] || 0) + 1; });
    const topEvent = Object.entries(evFreq).sort((a, b) => b[1] - a[1])[0] || ['—', 0];

    // Country frequency
    const cFreq = {};
    data.forEach(d => { if (d.country && d.country !== 'Global') cFreq[d.country] = (cFreq[d.country] || 0) + 1; });
    const topCountry = Object.entries(cFreq).sort((a, b) => b[1] - a[1])[0];

    // Region frequency
    const rFreq = {};
    data.forEach(d => { if (d.region) rFreq[d.region] = (rFreq[d.region] || 0) + 1; });
    const topRegion = Object.entries(rFreq).sort((a, b) => b[1] - a[1])[0];

    // Risk concentration ratio (HIGH + MEDIUM-HIGH as % of total)
    const criticalRatio = ((highCount + mhCount) / total * 100).toFixed(0);

    // Determine overall posture
    let posture, postureColor;
    if (highCount / total > 0.4 || maxScore > 0.85) {
        posture = 'CRITICAL'; postureColor = 'bad';
    } else if ((highCount + mhCount) / total > 0.3 || avgScore > 0.65) {
        posture = 'ELEVATED'; postureColor = 'warn';
    } else if (avgScore > 0.5) {
        posture = 'MODERATE'; postureColor = 'warn';
    } else {
        posture = 'LOW'; postureColor = 'good';
    }

    // ── Build insight cards ─────────────────────────────────────────────────
    const insightCards = [];

    // 1. Dominant threat
    insightCards.push({
        type: highCount > 0 ? 'danger' : mhCount > 0 ? 'warning' : 'info',
        icon: highCount > 0 ? '🚨' : '⚠️',
        title: `${topEvent[0]} Disruptions Dominating`,
        body: `<strong>${topEvent[0]}</strong> events account for <strong>${topEvent[1]} of ${total}</strong> detected articles
           (${((topEvent[1] / total) * 100).toFixed(0)}% of total universe).
           ${highCount > 0
                ? `<strong>${highCount} HIGH-risk</strong> event${highCount > 1 ? 's' : ''} require immediate escalation.`
                : 'No critical-level events detected at this time.'}`,
        meterVal: Math.round(topEvent[1] / total * 100),
        meterLabel: 'Dominance',
    });

    // 2. Geographic concentration
    if (topCountry) {
        const countryCritical = data.filter(d => d.country === topCountry[0] && ['HIGH', 'MEDIUM-HIGH'].includes(d.risk_level)).length;
        insightCards.push({
            type: countryCritical > 0 ? 'warning' : 'info',
            icon: '🌍',
            title: `Geographic Concentration — ${topCountry[0]}`,
            body: `<strong>${topCountry[0]}</strong> (${topRegion ? topRegion[0] : ''}) is the most-affected geography
             with <strong>${topCountry[1]} event${topCountry[1] > 1 ? 's' : ''}</strong> detected.
             ${countryCritical > 0
                    ? `<strong>${countryCritical}</strong> of these are HIGH or MEDIUM-HIGH severity, indicating a localised risk cluster.`
                    : 'Risk levels from this region are currently manageable.'}`,
            meterVal: Math.round(topCountry[1] / total * 100),
            meterLabel: 'Exposure',
        });
    }

    // 3. Average risk posture
    insightCards.push({
        type: avgScore >= 0.7 ? 'danger' : avgScore >= 0.55 ? 'warning' : 'success',
        icon: '📊',
        title: `Portfolio Average Risk Score — ${avgScore.toFixed(2)}`,
        body: `The weighted average risk score across all <strong>${total} articles</strong> is
           <strong>${avgScore.toFixed(2)}</strong> (scale 0–1).
           ${avgScore >= 0.7
                ? 'This is a <strong>high-risk environment</strong>. Immediate procurement contingency plans should be activated.'
                : avgScore >= 0.55
                    ? 'The environment is <strong>moderately stressed</strong>. Monitor closely and pre-position alternate sourcing.'
                    : 'Overall risk posture is <strong>manageable</strong>. Maintain standard monitoring cadence.'}`,
        meterVal: Math.round(avgScore * 100),
        meterLabel: 'Avg Score',
    });

    // 4. Critical concentration
    insightCards.push({
        type: parseInt(criticalRatio) > 40 ? 'danger' : parseInt(criticalRatio) > 20 ? 'warning' : 'success',
        icon: '🎯',
        title: `${criticalRatio}% of Events Are High-Severity`,
        body: `<strong>${highCount} HIGH</strong> and <strong>${mhCount} MEDIUM-HIGH</strong> events represent
           <strong>${criticalRatio}%</strong> of detected articles.
           ${parseInt(criticalRatio) > 40
                ? 'Concentration of high-severity signals warrants <strong>boardroom-level attention</strong> and supply chain stress testing.'
                : parseInt(criticalRatio) > 20
                    ? 'Elevated share of critical events. <strong>Increase monitoring frequency</strong> on affected categories.'
                    : 'Severity distribution is healthy — most events are low-to-medium impact.'}`,
        meterVal: parseInt(criticalRatio),
        meterLabel: 'Critical %',
    });

    // ── Build recommendations ───────────────────────────────────────────────
    const recommendations = [];

    if (highCount > 0) {
        recommendations.push({
            priority: 'high',
            action: `Execute Contingency Sourcing for ${topEvent[0]} Disruptions`,
            detail: `${highCount} HIGH-risk event${highCount > 1 ? 's are' : ' is'} active in the ${topEvent[0]} category. 
               Activate pre-qualified alternate suppliers and notify category managers immediately.`,
        });
    }

    if (evFreq['Geopolitical'] > 0) {
        recommendations.push({
            priority: highCount > 0 ? 'high' : 'medium',
            action: 'Review Trade Route Dependencies & Sanction Exposure',
            detail: `${evFreq['Geopolitical']} geopolitical event${evFreq['Geopolitical'] > 1 ? 's' : ''} detected. 
               Audit suppliers in affected countries for dual-use risk, trade sanctions, and export control compliance.`,
        });
    }

    if (evFreq['Logistics'] > 0 || evFreq['Weather'] > 0) {
        const logCount = (evFreq['Logistics'] || 0) + (evFreq['Weather'] || 0);
        recommendations.push({
            priority: 'medium',
            action: 'Increase Safety Stock & Review Lead-Time Buffers',
            detail: `${logCount} logistics/weather disruption event${logCount > 1 ? 's' : ''} indicate potential shipping delays. 
               Review buffer stock levels for A-category items from affected regions.`,
        });
    }

    if (evFreq['Labour'] > 0) {
        recommendations.push({
            priority: 'medium',
            action: 'Monitor Labour Disputes — Pre-Qualify Backup Suppliers',
            detail: `${evFreq['Labour']} labour-related event${evFreq['Labour'] > 1 ? 's' : ''} detected. 
               Proactively identify and qualify secondary suppliers in non-affected geographies to ensure business continuity.`,
        });
    }

    if (evFreq['Economic'] > 0) {
        recommendations.push({
            priority: parseInt(criticalRatio) > 30 ? 'medium' : 'low',
            action: 'Review FX Exposure & Commodity Price Hedging',
            detail: `${evFreq['Economic']} economic disruption signal${evFreq['Economic'] > 1 ? 's' : ''} detected. 
               Assess open purchase orders for FX sensitivity and review commodity price hedging positions.`,
        });
    }

    recommendations.push({
        priority: 'low',
        action: 'Schedule Weekly Risk Review Cadence',
        detail: `With ${total} active risk signals, establish a structured weekly review with category managers, 
             logistics leads, and finance to track evolving exposures.`,
    });

    // ── Render ──────────────────────────────────────────────────────────────
    container.innerHTML = `
    <!-- Summary stats bar -->
    <div class="insights-summary">
      <div class="ins-stat">
        <div class="ins-stat-label">Overall Posture</div>
        <div class="ins-stat-value ${postureColor}">${posture}</div>
      </div>
      <div class="ins-stat">
        <div class="ins-stat-label">Avg Risk Score</div>
        <div class="ins-stat-value ${avgScore >= 0.7 ? 'bad' : avgScore >= 0.55 ? 'warn' : 'good'}">${avgScore.toFixed(2)}</div>
      </div>
      <div class="ins-stat">
        <div class="ins-stat-label">Critical Events</div>
        <div class="ins-stat-value ${(highCount + mhCount) > 0 ? 'bad' : 'good'}">${highCount + mhCount}</div>
      </div>
      <div class="ins-stat">
        <div class="ins-stat-label">Top Threat Type</div>
        <div class="ins-stat-value" style="font-size:0.9rem">${topEvent[0]}</div>
      </div>
      <div class="ins-stat">
        <div class="ins-stat-label">Most Exposed Region</div>
        <div class="ins-stat-value" style="font-size:0.9rem">${topRegion ? topRegion[0] : '—'}</div>
      </div>
      <div class="ins-stat">
        <div class="ins-stat-label">Max Risk Score</div>
        <div class="ins-stat-value ${maxScore >= 0.8 ? 'bad' : maxScore >= 0.6 ? 'warn' : 'good'}">${maxScore.toFixed(2)}</div>
      </div>
    </div>

    <!-- Insight cards -->
    <div class="insights-grid">
      ${insightCards.map(c => _insightCardHTML(c)).join('')}
    </div>

    <!-- Recommendations panel -->
    <div class="recommendations-panel">
      <div class="rec-header">
        <div class="rec-title">📋 Procurement Recommendations</div>
        <span class="rec-badge">${recommendations.length} Actions</span>
      </div>
      <div class="rec-list">
        ${recommendations.map(r => _recItemHTML(r)).join('')}
      </div>
    </div>
  `;
}

function _insightCardHTML(c) {
    return `
    <div class="insight-card card-${c.type}">
      <div class="insight-card-header">
        <div class="insight-icon-wrap ${c.type}">${c.icon}</div>
        <div class="insight-card-meta">
          <div class="insight-card-title">${c.title}</div>
          <div class="insight-card-type ${c.type}">${c.type.replace('-', '').toUpperCase()} SIGNAL</div>
        </div>
      </div>
      <div class="insight-card-body">${c.body}</div>
      <div class="insight-meter">
        <span class="insight-meter-label">${c.meterLabel}</span>
        <div class="insight-meter-bar">
          <div class="insight-meter-fill ${c.type}" style="width:${Math.min(c.meterVal, 100)}%"></div>
        </div>
        <span class="insight-meter-val">${c.meterVal}%</span>
      </div>
    </div>`;
}

function _recItemHTML(r) {
    return `
    <div class="rec-item">
      <span class="rec-priority ${r.priority}">${r.priority}</span>
      <div class="rec-content">
        <div class="rec-action">${r.action}</div>
        <div class="rec-detail">${r.detail}</div>
      </div>
    </div>`;
}
