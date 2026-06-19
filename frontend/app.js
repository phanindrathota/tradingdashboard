const tickersInput = document.querySelector('#tickers');
const refreshButton = document.querySelector('#refresh');
const body = document.querySelector('#dashboard-body');
const errors = document.querySelector('#error-panel');
const label = document.querySelector('#connection-label');
const updatedAt = document.querySelector('#updated-at');
const dot = document.querySelector('#connection-dot');

const timeframes = ['W', 'D', '4H', '65m/1H', '30m', '15m', '10m', '5m'];

function cssToken(value) { return String(value || 'WAIT').toLowerCase().replaceAll(' ', '-').replaceAll('/', '-'); }
function badge(value) { return `<span class="badge ${cssToken(value)}">${value || 'WAIT'}</span>`; }
function setStatus(text, mode = 'wait') { label.textContent = text; dot.className = `dot ${mode}`; }

function renderRows(rows) {
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="15" class="empty">No rows returned. Check the ticker symbols and try again.</td></tr>';
    return;
  }
  body.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.ticker}<br><small>$${row.lastPrice ? row.lastPrice.toFixed(2) : 'n/a'}</small></td>
      ${timeframes.map((tf) => `<td>${badge(row.timeframes[tf])}</td>`).join('')}
      <td title="High: ${row.orb.high?.toFixed?.(2) ?? 'n/a'} Low: ${row.orb.low?.toFixed?.(2) ?? 'n/a'} Range: ${row.orb.range?.toFixed?.(2) ?? 'n/a'}">${badge(row.orb.status)}</td>
      <td>${badge(row.vwapSide)}</td>
      <td>${badge(row.sma200Side)}</td>
      <td class="score">${row.score}</td>
      <td>${badge(row.bias)}</td>
      <td>${badge(row.entryStatus)}</td>
    </tr>`).join('');
}

async function refresh() {
  refreshButton.disabled = true;
  setStatus('Scanning...', 'wait');
  errors.classList.add('hidden');
  try {
    const params = new URLSearchParams({ tickers: tickersInput.value });
    const response = await fetch(`/api/scan?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    renderRows(data.rows);
    updatedAt.textContent = `Updated ${new Date(data.updatedAt).toLocaleString()}`;
    setStatus('Live', 'bull');
    if (data.errors?.length) {
      errors.innerHTML = data.errors.map((error) => `<div><strong>${error.ticker}</strong>: ${error.message}</div>`).join('');
      errors.classList.remove('hidden');
    }
  } catch (error) {
    setStatus('Error', 'bear');
    errors.textContent = error.message;
    errors.classList.remove('hidden');
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener('click', refresh);
setInterval(refresh, 60_000);
refresh();
