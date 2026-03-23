/**
 * charts.js — 7 Chart.js v4 renderers
 * Maximally defensive — accepts both 'event_type' and 'domain' field names,
 * trims all string values, and logs data summary on every render call.
 */

Chart.defaults.color = '#6b86a4';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 10;

var EVENT_COLORS = {
    Geopolitical:'#b78cf7', Logistics:'#38c8f0',
    Weather:'#30d999', Labour:'#ff9547', Economic:'#f06aad'
};
var RISK_COLORS = {
    'HIGH':'#ff3d54','MEDIUM-HIGH':'#ff6b35',
    'MEDIUM':'#f5a623','LOW-MEDIUM':'#3ecf72','LOW':'#00c8ff'
};
var SENTIMENT_COLORS = {'Negative':'#ff3d54','Neutral':'#f5a623','Positive':'#3ecf72'};
var CHART_RISK_ORDER = ['HIGH','MEDIUM-HIGH','MEDIUM','LOW-MEDIUM','LOW'];
var CHART_SENTIMENT_ORDER = ['Negative','Neutral','Positive'];

var TT = {
    backgroundColor:'#0b1525', borderColor:'rgba(255,255,255,0.08)',
    borderWidth:1, padding:10,
    titleFont:{family:"'DM Sans',sans-serif",size:11,weight:'600'},
    bodyFont:{family:"'DM Mono',monospace",size:10},
    titleColor:'#ddeef8', bodyColor:'#8ba5be',
    cornerRadius:8, displayColors:true, boxWidth:8, boxHeight:8, boxPadding:4
};
var G='rgba(255,255,255,0.04)', B='rgba(255,255,255,0.05)', K='#4e6882';

var _ch = {};
function _sd(k){ if(_ch[k]){ _ch[k].destroy(); _ch[k]=null; } }

/* Helper: get domain value from a record */
function _dom(d){ return (d.event_type||d.domain||'').trim(); }

/* Helper: get risk level, trimmed */
function _rl(d){ return (d.risk_level||'').trim(); }

/* Helper: get sentiment, trimmed */
function _sent(d){ return (d.sentiment||'').trim(); }

function _showEmpty(id, msg){
    var el=document.getElementById(id); if(!el) return;
    el.style.display='none';
    var p=el.parentNode;
    var ex=p.querySelector('.chart-empty-msg');
    if(ex) ex.remove();
    var div=document.createElement('div');
    div.className='chart-empty-msg';
    div.style.cssText='color:#4e6882;font-size:0.75rem;text-align:center;padding:40px 10px;';
    div.textContent=msg||'No data'; p.appendChild(div);
}
function _clearEmpty(id){
    var el=document.getElementById(id); if(!el) return;
    el.style.display='';
    var ex=el.parentNode.querySelector('.chart-empty-msg');
    if(ex) ex.remove();
}

/* ── 1. Events by domain ── */
function renderBarChart(data){
    _sd('bar'); _clearEmpty('bar-chart');
    var counts={};
    data.forEach(function(d){ var k=_dom(d); if(k) counts[k]=(counts[k]||0)+1; });
    var labels=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a];});
    if(!labels.length){_showEmpty('bar-chart','No data');return;}
    var values=labels.map(function(l){return counts[l];});
    var colors=labels.map(function(l){return EVENT_COLORS[l]||'#94a3b8';});
    var ctx=document.getElementById('bar-chart').getContext('2d');
    _ch.bar=new Chart(ctx,{type:'bar',
        data:{labels:labels,datasets:[{label:'Articles',data:values,
            backgroundColor:colors.map(function(c){return c+'28';}),
            borderColor:colors,borderWidth:1.5,borderRadius:5,borderSkipped:false,
            hoverBackgroundColor:colors.map(function(c){return c+'55';})}]},
        options:{responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:Object.assign({},TT,{callbacks:{
                title:function(c){return c[0].label;},
                label:function(c){return '  '+c.parsed.y+' article'+(c.parsed.y!==1?'s':'');}
            }})},
            scales:{
                x:{grid:{color:G},ticks:{color:K},border:{color:B}},
                y:{grid:{color:G},ticks:{color:K,stepSize:1},border:{color:B},
                   beginAtZero:true,title:{display:true,text:'Articles',color:K,font:{size:9}}}
            }}
    });
}

/* ── 2. Risk level doughnut ── */
function renderPieChart(data){
    _sd('pie'); _clearEmpty('pie-chart');
    var counts={};
    data.forEach(function(d){ var r=_rl(d); if(r) counts[r]=(counts[r]||0)+1; });
    // Use RISK_ORDER first, fallback to whatever keys exist
    var labels=CHART_RISK_ORDER.filter(function(r){return counts[r]>0;});
    if(!labels.length) labels=Object.keys(counts).filter(function(k){return k;});
    if(!labels.length){_showEmpty('pie-chart','No risk data');return;}
    var values=labels.map(function(l){return counts[l]||0;});
    var colors=labels.map(function(l){return RISK_COLORS[l]||'#94a3b8';});
    var total=values.reduce(function(a,b){return a+b;},0);
    var ctx=document.getElementById('pie-chart').getContext('2d');
    _ch.pie=new Chart(ctx,{type:'doughnut',
        data:{labels:labels,datasets:[{data:values,
            backgroundColor:colors.map(function(c){return c+'44';}),
            borderColor:colors,borderWidth:1.5,hoverBorderWidth:2.5,
            hoverBackgroundColor:colors.map(function(c){return c+'77';})}]},
        options:{responsive:true,maintainAspectRatio:false,cutout:'64%',plugins:{
            legend:{position:'bottom',labels:{boxWidth:8,boxHeight:8,padding:10,color:'#6b86a4',
                font:{family:"'DM Mono',monospace",size:9},
                generateLabels:function(chart){
                    var ds=chart.data.datasets[0];
                    return chart.data.labels.map(function(label,i){
                        return{text:label+'  '+Math.round((ds.data[i]/total)*100)+'%',
                            fillStyle:ds.backgroundColor[i],strokeStyle:ds.borderColor[i],lineWidth:1,index:i};
                    });
                }}},
            tooltip:Object.assign({},TT,{callbacks:{
                label:function(c){return '  '+c.label+': '+c.parsed+' ('+Math.round((c.parsed/total)*100)+'%)';}
            }})
        }}
    });
}

/* ── 3. Risk score trend ── */
function renderLineChart(data){
    _sd('line'); _clearEmpty('line-chart');
    if(!data.length){_showEmpty('line-chart','No data');return;}
    var sorted=data.slice();
    if(sorted[0]&&sorted[0].published_at)
        sorted.sort(function(a,b){return new Date(a.published_at)-new Date(b.published_at);});
    var labels=sorted.map(function(d,i){
        return d.published_at
            ?new Date(d.published_at).toLocaleDateString('en-GB',{day:'2-digit',month:'short'})
            :'#'+(i+1);
    });
    var scores=sorted.map(function(d){return parseFloat(d.risk_score)||0;});
    var trend=scores.map(function(_,i){
        var sl=scores.slice(Math.max(0,i-1),i+2);
        return parseFloat((sl.reduce(function(a,b){return a+b;},0)/sl.length).toFixed(3));
    });
    var ctx=document.getElementById('line-chart').getContext('2d');
    var grad=ctx.createLinearGradient(0,0,0,180);
    grad.addColorStop(0,'rgba(0,200,255,0.28)'); grad.addColorStop(1,'rgba(0,200,255,0.01)');
    _ch.line=new Chart(ctx,{type:'line',
        data:{labels:labels,datasets:[
            {label:'Score',data:scores,borderColor:'#00c8ff',backgroundColor:grad,
             borderWidth:1.5,pointRadius:2.5,pointBackgroundColor:'#00c8ff',
             pointHoverRadius:5,tension:0.4,fill:true,order:2},
            {label:'Avg',data:trend,borderColor:'#f5a623',backgroundColor:'transparent',
             borderWidth:1.5,borderDash:[4,3],pointRadius:0,tension:0.4,fill:false,order:1}
        ]},
        options:{responsive:true,maintainAspectRatio:false,
            interaction:{mode:'index',intersect:false},
            plugins:{
                legend:{display:true,position:'top',align:'end',labels:{
                    boxWidth:16,boxHeight:2,padding:10,color:'#6b86a4',
                    font:{family:"'DM Mono',monospace",size:9}}},
                tooltip:Object.assign({},TT,{callbacks:{
                    label:function(c){return '  '+c.dataset.label+': '+c.parsed.y.toFixed(3);}
                }})},
            scales:{
                x:{grid:{color:G},ticks:{color:K,maxTicksLimit:8},border:{color:B}},
                y:{min:0,grid:{color:G},ticks:{color:K,callback:function(v){return v.toFixed(2);}},
                   border:{color:B},title:{display:true,text:'Risk Score',color:K,font:{size:9}}}
            }}
    });
}

/* ── 4. Avg risk per domain ── */
function renderAvgRiskChart(data){
    _sd('hbar'); _clearEmpty('hbar-chart');
    var sums={},cnts={};
    data.forEach(function(d){
        var k=_dom(d); if(!k) return;
        sums[k]=(sums[k]||0)+(parseFloat(d.risk_score)||0);
        cnts[k]=(cnts[k]||0)+1;
    });
    var labels=Object.keys(sums).sort(function(a,b){return(sums[b]/cnts[b])-(sums[a]/cnts[a]);});
    if(!labels.length){_showEmpty('hbar-chart','No data');return;}
    var avgs=labels.map(function(l){return parseFloat((sums[l]/cnts[l]).toFixed(3));});
    var colors=labels.map(function(l){return EVENT_COLORS[l]||'#94a3b8';});
    var ctx=document.getElementById('hbar-chart').getContext('2d');
    _ch.hbar=new Chart(ctx,{type:'bar',
        data:{labels:labels,datasets:[{label:'Avg Risk',data:avgs,
            backgroundColor:colors.map(function(c){return c+'30';}),
            borderColor:colors,borderWidth:1.5,borderRadius:5,borderSkipped:false,
            hoverBackgroundColor:colors.map(function(c){return c+'60';})}]},
        options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:Object.assign({},TT,{callbacks:{
                title:function(c){return c[0].label;},
                label:function(c){return['  Avg: '+c.parsed.x.toFixed(3),
                    '  '+cnts[c.label]+' article'+(cnts[c.label]!==1?'s':'')];}
            }})},
            scales:{
                x:{min:0,grid:{color:G},ticks:{color:K,callback:function(v){return v.toFixed(2);}},
                   border:{color:B},title:{display:true,text:'Avg Risk Score',color:K,font:{size:9}}},
                y:{grid:{color:'transparent'},ticks:{color:'#ddeef8',font:{size:10}},border:{color:B}}
            }}
    });
}

/* ── 5. Risk mix per domain (stacked) ── */
function renderStackedChart(data){
    _sd('stacked'); _clearEmpty('stacked-chart');
    var domains=[];
    data.forEach(function(d){
        var k=_dom(d); if(k&&domains.indexOf(k)===-1) domains.push(k);
    });
    var risks=CHART_RISK_ORDER.filter(function(r){return data.some(function(d){return _rl(d)===r;});});
    if(!domains.length){_showEmpty('stacked-chart','No data');return;}
    if(!risks.length){
        // fallback: use whatever risk levels exist
        data.forEach(function(d){ var r=_rl(d); if(r&&risks.indexOf(r)===-1) risks.push(r); });
    }
    if(!risks.length){_showEmpty('stacked-chart','No risk levels');return;}
    var matrix={};
    risks.forEach(function(r){ matrix[r]={}; domains.forEach(function(d){matrix[r][d]=0;}); });
    data.forEach(function(d){
        var k=_dom(d), r=_rl(d);
        if(k&&r&&matrix[r]) matrix[r][k]=(matrix[r][k]||0)+1;
    });
    var datasets=risks.map(function(r){
        return{label:r,data:domains.map(function(d){return matrix[r][d]||0;}),
            backgroundColor:(RISK_COLORS[r]||'#94a3b8')+'88',
            borderColor:RISK_COLORS[r]||'#94a3b8',borderWidth:1,borderRadius:3,borderSkipped:false};
    });
    var ctx=document.getElementById('stacked-chart').getContext('2d');
    _ch.stacked=new Chart(ctx,{type:'bar',
        data:{labels:domains,datasets:datasets},
        options:{responsive:true,maintainAspectRatio:false,
            interaction:{mode:'index',intersect:false},
            plugins:{
                legend:{display:true,position:'top',align:'end',labels:{
                    boxWidth:8,boxHeight:8,padding:8,color:'#6b86a4',
                    font:{family:"'DM Mono',monospace",size:9}}},
                tooltip:Object.assign({},TT,{callbacks:{
                    title:function(c){return c[0].label;},
                    label:function(c){return '  '+c.dataset.label+': '+c.parsed.y;}
                }})},
            scales:{
                x:{stacked:true,grid:{color:G},ticks:{color:K},border:{color:B}},
                y:{stacked:true,grid:{color:G},ticks:{color:K,stepSize:1},border:{color:B},
                   beginAtZero:true,title:{display:true,text:'Count',color:K,font:{size:9}}}
            }}
    });
}

/* ── 6. Top regions ── */
function renderCountryChart(data){
    _sd('country'); _clearEmpty('country-chart');
    var counts={},riskMap={};
    data.forEach(function(d){
        var key=(d.country||d.region||'Global').trim();
        counts[key]=(counts[key]||0)+1;
        if(!riskMap[key]) riskMap[key]={};
        var r=_rl(d); if(r) riskMap[key][r]=(riskMap[key][r]||0)+1;
    });
    var sorted=Object.entries(counts).sort(function(a,b){return b[1]-a[1];}).slice(0,8);
    if(!sorted.length){_showEmpty('country-chart','No geographic data');return;}
    var labels=sorted.map(function(e){return e[0];});
    var values=sorted.map(function(e){return e[1];});
    var colors=labels.map(function(l){
        var rm=riskMap[l]; if(!rm||!Object.keys(rm).length) return '#94a3b8';
        var dom=Object.entries(rm).sort(function(a,b){return b[1]-a[1];})[0][0];
        return RISK_COLORS[dom]||'#94a3b8';
    });
    var ctx=document.getElementById('country-chart').getContext('2d');
    _ch.country=new Chart(ctx,{type:'bar',
        data:{labels:labels,datasets:[{label:'Events',data:values,
            backgroundColor:colors.map(function(c){return c+'30';}),
            borderColor:colors,borderWidth:1.5,borderRadius:5,borderSkipped:false,
            hoverBackgroundColor:colors.map(function(c){return c+'60';})}]},
        options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:Object.assign({},TT,{callbacks:{
                title:function(c){return c[0].label;},
                label:function(c){
                    var rm=riskMap[c.label];
                    var dom=rm&&Object.keys(rm).length
                        ?Object.entries(rm).sort(function(a,b){return b[1]-a[1];})[0][0]:'—';
                    return['  '+c.parsed.x+' article'+(c.parsed.x!==1?'s':''),'  Dominant risk: '+dom];
                }
            }})},
            scales:{
                x:{grid:{color:G},ticks:{color:K,stepSize:1},border:{color:B},beginAtZero:true,
                   title:{display:true,text:'Articles',color:K,font:{size:9}}},
                y:{grid:{color:'transparent'},ticks:{color:'#ddeef8',font:{size:10}},border:{color:B}}
            }}
    });
}

/* ── 7. Sentiment per domain (stacked) ── */
function renderSentimentChart(data){
    _sd('sentiment'); _clearEmpty('sentiment-chart');
    var domains=[];
    data.forEach(function(d){
        var k=_dom(d); if(k&&domains.indexOf(k)===-1) domains.push(k);
    });
    var sents=CHART_SENTIMENT_ORDER.filter(function(s){return data.some(function(d){return _sent(d)===s;});});
    if(!domains.length||!sents.length){_showEmpty('sentiment-chart','No sentiment data');return;}
    var matrix={};
    sents.forEach(function(s){ matrix[s]={}; domains.forEach(function(d){matrix[s][d]=0;}); });
    data.forEach(function(d){
        var k=_dom(d), s=_sent(d);
        if(k&&s&&matrix[s]) matrix[s][k]=(matrix[s][k]||0)+1;
    });
    var datasets=sents.map(function(s){
        return{label:s,data:domains.map(function(d){return matrix[s][d]||0;}),
            backgroundColor:(SENTIMENT_COLORS[s]||'#94a3b8')+'88',
            borderColor:SENTIMENT_COLORS[s]||'#94a3b8',borderWidth:1,borderRadius:3,borderSkipped:false};
    });
    var ctx=document.getElementById('sentiment-chart').getContext('2d');
    _ch.sentiment=new Chart(ctx,{type:'bar',
        data:{labels:domains,datasets:datasets},
        options:{responsive:true,maintainAspectRatio:false,
            interaction:{mode:'index',intersect:false},
            plugins:{
                legend:{display:true,position:'top',align:'end',labels:{
                    boxWidth:8,boxHeight:8,padding:8,color:'#6b86a4',
                    font:{family:"'DM Mono',monospace",size:9}}},
                tooltip:Object.assign({},TT,{callbacks:{
                    title:function(c){return c[0].label;},
                    label:function(c){return '  '+c.dataset.label+': '+c.parsed.y;}
                }})},
            scales:{
                x:{stacked:true,grid:{color:G},ticks:{color:K},border:{color:B}},
                y:{stacked:true,grid:{color:G},ticks:{color:K,stepSize:1},border:{color:B},
                   beginAtZero:true,title:{display:true,text:'Count',color:K,font:{size:9}}}
            }}
    });
}