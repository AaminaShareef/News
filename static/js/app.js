'use strict';
var STATE={allResults:[],filtered:[],currentPage:1,pageSize:12};
function $(id){return document.getElementById(id);}
function $$(sel){return document.querySelectorAll(sel);}
function showSpinner(msg){$('spinner-overlay').classList.add('active');$('spinner-text').textContent=msg||'Analysing…';}
function hideSpinner(){$('spinner-overlay').classList.remove('active');}
function showToast(msg,type){var t=$('toast');t.textContent=msg;t.className=(type==='error')?'error show':'show';setTimeout(function(){t.classList.remove('show');},3500);}

var RISK_ORDER={'HIGH':5,'MEDIUM-HIGH':4,'MEDIUM':3,'LOW-MEDIUM':2,'LOW':1};

function updateKPIs(data){
    $('kpi-total').textContent=data.length;
    $('kpi-high').textContent=data.filter(function(d){return(d.risk_level||'').trim()==='HIGH';}).length;
    $('kpi-medium').textContent=data.filter(function(d){var r=(d.risk_level||'').trim();return r==='MEDIUM'||r==='MEDIUM-HIGH';}).length;
    $('kpi-low').textContent=data.filter(function(d){var r=(d.risk_level||'').trim();return r==='LOW'||r==='LOW-MEDIUM';}).length;
    var evCounts={};
    data.forEach(function(d){var k=(d.event_type||d.domain||'').trim();if(k)evCounts[k]=(evCounts[k]||0)+1;});
    var entries=Object.entries(evCounts);
    $('kpi-event').textContent=entries.length?entries.sort(function(a,b){return b[1]-a[1];})[0][0]:'—';
    var neg=data.filter(function(d){return(d.sentiment||'').trim()==='Negative';}).length;
    var pos=data.filter(function(d){return(d.sentiment||'').trim()==='Positive';}).length;
    var el=$('kpi-sentiment');
    if(el){el.textContent=neg+' / '+pos;el.className='kpi-value'+(neg>pos?' high':neg<pos?' low':' medium');}
}

function checkHighRiskAlert(data){
    var n=data.filter(function(d){return(d.risk_level||'').trim()==='HIGH';}).length;
    var b=$('alert-banner');
    if(n>0){$('alert-count').textContent=n;b.classList.add('visible');}else b.classList.remove('visible');
}

function populateFilters(data){
    var sets={country:new Set(),region:new Set(),event_type:new Set(),risk_level:new Set(),sentiment:new Set()};
    data.forEach(function(d){
        if(d.country) sets.country.add(d.country.trim());
        if(d.region)  sets.region.add(d.region.trim());
        var k=(d.event_type||d.domain||'').trim(); if(k) sets.event_type.add(k);
        var r=(d.risk_level||'').trim(); if(r) sets.risk_level.add(r);
        var s=(d.sentiment||'').trim(); if(s) sets.sentiment.add(s);
    });
    _buildFilterGroup('filter-country',Array.from(sets.country).sort(),'country',data);
    _buildFilterGroup('filter-region', Array.from(sets.region).sort(), 'region', data);
    _buildFilterGroup('filter-event-type',Array.from(sets.event_type).sort(),'event_type',data);
    _buildFilterGroup('filter-risk-level',
        Array.from(sets.risk_level).sort(function(a,b){return(RISK_ORDER[b]||0)-(RISK_ORDER[a]||0);}),
        'risk_level',data);
    _buildFilterGroup('filter-sentiment',
        ['Negative','Neutral','Positive'].filter(function(s){return sets.sentiment.has(s);}),
        'sentiment',data);
}

function _buildFilterGroup(cid,options,field,data){
    var c=$(cid); if(!c) return; c.innerHTML='';
    options.forEach(function(opt){
        var count=data.filter(function(d){
            var v=(d[field]||d.domain||'').trim(); return v===opt;
        }).length;
        var uid='chk-'+cid+'-'+opt.replace(/\s+/g,'-');
        var label=document.createElement('label');
        label.className='filter-option'; label.htmlFor=uid;
        label.innerHTML='<input type="checkbox" id="'+uid+'" data-field="'+field+'" data-value="'+opt+'">'
            +'<span class="opt-label">'+opt+'</span><span class="opt-count">'+count+'</span>';
        c.appendChild(label);
        label.querySelector('input').addEventListener('change',applyFilters);
    });
}

function applyFilters(){
    var active={};
    $$('.filter-option input:checked').forEach(function(chk){
        var f=chk.dataset.field; if(!active[f]) active[f]=new Set(); active[f].add(chk.dataset.value);
    });
    STATE.filtered=STATE.allResults.filter(function(d){
        for(var field in active){
            var val=(d[field]||d.domain||'').trim();
            if(!active[field].has(val)) return false;
        }
        return true;
    });
    STATE.currentPage=1; renderPage();
    renderCharts(STATE.filtered.length?STATE.filtered:STATE.allResults);
}

function clearFilters(){
    $$('.filter-option input').forEach(function(c){c.checked=false;});
    STATE.filtered=STATE.allResults.slice(); STATE.currentPage=1;
    renderPage(); renderCharts(STATE.allResults);
}

function renderPage(){
    var data=STATE.filtered, total=data.length;
    var totalPages=Math.max(1,Math.ceil(total/STATE.pageSize));
    STATE.currentPage=Math.min(STATE.currentPage,totalPages);
    var start=(STATE.currentPage-1)*STATE.pageSize;
    var slice=data.slice(start,start+STATE.pageSize);
    var grid=$('news-grid'); grid.innerHTML='';
    $('results-count').textContent=total+' article'+(total!==1?'s':'');
    if(!slice.length){
        grid.innerHTML='<div class="empty-state"><div class="es-icon">🔍</div><p>No articles match filters.</p></div>';
        $('pagination').innerHTML=''; return;
    }
    slice.forEach(function(d){grid.appendChild(buildCard(d));});
    buildPagination(totalPages);
}

var SENT_ICON={'Negative':'🔴','Neutral':'🟡','Positive':'🟢'};
function buildCard(d){
    var el=document.createElement('article'); el.className='news-card';
    var rl=(d.risk_level||'LOW').trim();
    var et=(d.event_type||d.domain||'—').trim();
    var score=parseFloat(d.risk_score)||0;
    var pct=Math.min(Math.round(score*300),100); // scale 0-0.33 to 0-100%
    var titleHtml=d.url
        ?'<a href="'+_esc(d.url)+'" target="_blank" rel="noopener">'+_esc(d.title||'—')+'</a>'
        :_esc(d.title||'—');
    var dateStr=d.published_at?new Date(d.published_at).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}):'';
    var sent=(d.sentiment||'').trim();
    el.innerHTML=
        '<div class="risk-stripe '+rl+'"></div>'+
        '<div class="card-top"><div class="card-title">'+titleHtml+'</div>'+
        '<span class="badge badge-'+rl+'">'+rl+'</span></div>'+
        '<div class="card-meta">'+
        '<span class="badge badge-etype '+et+'">'+et+'</span>'+
        (sent?'<span class="sentiment-pill '+sent+'">'+(SENT_ICON[sent]||'')+' '+sent+'</span>':'')+
        (d.source?'<span class="meta-item"><span class="mi-icon">📰</span>'+_esc(d.source)+'</span>':'')+
        (d.country?'<span class="meta-item"><span class="mi-icon">🌍</span>'+_esc(d.country)+'</span>':'')+
        (dateStr?'<span class="meta-item"><span class="mi-icon">📅</span>'+dateStr+'</span>':'')+
        '</div>'+
        '<div class="score-row"><span class="score-label">RISK</span>'+
        '<div class="score-bar-wrap"><div class="score-bar-fill '+rl+'" style="width:'+pct+'%"></div></div>'+
        '<span class="score-val">'+score.toFixed(3)+'</span></div>';
    return el;
}
function _esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

function buildPagination(totalPages){
    var pg=$('pagination'); pg.innerHTML=''; if(totalPages<=1) return;
    var prev=document.createElement('button');
    prev.className='page-btn'; prev.innerHTML='&lsaquo; Prev'; prev.disabled=STATE.currentPage===1;
    prev.onclick=function(){STATE.currentPage--;renderPage();}; pg.appendChild(prev);
    var s=Math.max(1,STATE.currentPage-3), e=Math.min(totalPages,s+6);
    if(e-s<6) s=Math.max(1,e-6);
    if(s>1){pg.appendChild(_pb(1));if(s>2)pg.appendChild(_el());}
    for(var i=s;i<=e;i++) pg.appendChild(_pb(i));
    if(e<totalPages){if(e<totalPages-1)pg.appendChild(_el());pg.appendChild(_pb(totalPages));}
    var next=document.createElement('button');
    next.className='page-btn'; next.innerHTML='Next &rsaquo;'; next.disabled=STATE.currentPage===totalPages;
    next.onclick=function(){STATE.currentPage++;renderPage();}; pg.appendChild(next);
    var info=document.createElement('span'); info.className='page-info';
    info.textContent='Page '+STATE.currentPage+' of '+totalPages; pg.appendChild(info);
}
function _pb(n){var b=document.createElement('button');b.className='page-btn'+(n===STATE.currentPage?' active':'');b.textContent=n;b.onclick=function(){STATE.currentPage=n;renderPage();};return b;}
function _el(){var s=document.createElement('span');s.className='page-info';s.textContent='…';return s;}

function renderCharts(data){
    if(typeof renderBarChart==='function')       renderBarChart(data);
    if(typeof renderPieChart==='function')       renderPieChart(data);
    if(typeof renderLineChart==='function')      renderLineChart(data);
    if(typeof renderAvgRiskChart==='function')   renderAvgRiskChart(data);
    if(typeof renderStackedChart==='function')   renderStackedChart(data);
    if(typeof renderCountryChart==='function')   renderCountryChart(data);
    if(typeof renderSentimentChart==='function') renderSentimentChart(data);
    if(typeof renderMap==='function')            renderMap(data);
}

function handleResults(records){
    STATE.allResults=records; STATE.filtered=records.slice(); STATE.currentPage=1;
    populateFilters(records); updateKPIs(records); checkHighRiskAlert(records);
    renderCharts(records); renderPage();
}

async function runAnalysis(){
    var query=$('search-input').value.trim();
    if(!query){showToast('Please enter a search query.','error');return;}
    hideDropdown();
    showSpinner('Fetching & analysing news…');
    try{
        var res=await fetch('/api/analyze',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({query:query,page_size:30})});
        var json=await res.json();
        if(!res.ok) throw new Error(json.error||'Server error');
        handleResults(json.results||[]);
        showToast('✅ '+(json.results||[]).length+' articles found.');
    }catch(e){showToast('Error: '+e.message,'error');}
    finally{hideSpinner();}
}

async function uploadCSV(file){
    showSpinner('Analysing CSV…');
    try{
        var fd=new FormData(); fd.append('file',file);
        var res=await fetch('/api/upload',{method:'POST',body:fd});
        var json=await res.json();
        if(!res.ok) throw new Error(json.error||'Upload error');
        handleResults(json.results||[]);
        showToast('✅ CSV: '+(json.results||[]).length+' articles found.');
    }catch(e){showToast('Error: '+e.message,'error');}
    finally{hideSpinner();}
}

async function exportCSV(){
    if(!STATE.filtered.length){showToast('No data to export.','error');return;}
    showSpinner('Exporting…');
    try{
        var res=await fetch('/api/export',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({results:STATE.filtered})});
        if(!res.ok) throw new Error('Export failed');
        var blob=await res.blob();
        var url=URL.createObjectURL(blob);
        var a=document.createElement('a'); a.href=url;
        a.download='risk_report_'+new Date().toISOString().slice(0,10)+'.csv';
        a.click(); URL.revokeObjectURL(url);
        showToast('📥 Exported.');
    }catch(e){showToast('Error: '+e.message,'error');}
    finally{hideSpinner();}
}

/* ── Search dropdown logic ── */
function showDropdown(){ var d=$('search-dropdown'); if(d) d.classList.add('open'); }
function hideDropdown(){ var d=$('search-dropdown'); if(d) d.classList.remove('open'); }

document.addEventListener('DOMContentLoaded',function(){
    $('analyze-btn').addEventListener('click',runAnalysis);

    var inp=$('search-input');
    inp.addEventListener('focus', showDropdown);
    inp.addEventListener('keydown',function(e){
        if(e.key==='Enter') runAnalysis();
        if(e.key==='Escape') hideDropdown();
    });
    inp.addEventListener('input',function(){
        if(inp.value.length>0) hideDropdown(); else showDropdown();
    });

    // Dropdown item click
    $$('.dropdown-item').forEach(function(item){
        item.addEventListener('click',function(){
            inp.value=item.getAttribute('data-query');
            hideDropdown();
            runAnalysis();
        });
    });

    // Clear button
    $('search-clear').addEventListener('click',function(){
        inp.value=''; inp.focus(); showDropdown();
    });

    // Click outside closes dropdown
    document.addEventListener('click',function(e){
        var wrap=$('search-container');
        var dd=$('search-dropdown');
        if(wrap&&dd&&!wrap.contains(e.target)&&!dd.contains(e.target)) hideDropdown();
    });

    $('upload-btn').addEventListener('click',function(){$('upload-input').click();});
    $('upload-input').addEventListener('change',function(e){
        var file=e.target.files[0]; if(file){uploadCSV(file);e.target.value='';}
    });
    $('export-btn').addEventListener('click',exportCSV);
    $('clear-filters').addEventListener('click',clearFilters);
    $('close-alert').addEventListener('click',function(){$('alert-banner').classList.remove('visible');});

    $('news-grid').innerHTML='<div class="empty-state"><div class="es-icon">🛰️</div>'
        +'<p>Choose a topic from the search bar or type your own query,<br>then click <strong>Analyse</strong>.</p></div>';
});