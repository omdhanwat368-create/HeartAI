// static/js/chat.js
document.addEventListener('DOMContentLoaded', () => {
  const chatWindow = document.getElementById('chatWindow');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const quickArea = document.getElementById('quickArea');
  const statusEl = document.getElementById('chatStatus');

  function appendBotMessage(htmlContent) {
    const row = document.createElement('div');
    row.className = 'row-bot';
    row.innerHTML = `<div class="msg bot">${htmlContent}</div>`;
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'row-user';
    row.innerHTML = `<div class="msg user">${escapeHtml(text)}</div>`;
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function escapeHtml(s){ return (s+'').replace(/[&<>"']/g, c=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  function renderCard(data) {
    // build a small card: title, summary, bullets
    let html = `<div class="card-box">`;
    if (data.title) html += `<h5>${escapeHtml(data.title)}</h5>`;
    if (data.summary) html += `<div class="small-muted">${escapeHtml(data.summary)}</div>`;
    if (Array.isArray(data.bullets) && data.bullets.length) {
      html += `<ul>`;
      for (const b of data.bullets) html += `<li>${escapeHtml(b)}</li>`;
      html += `</ul>`;
    }
    html += `</div>`;
    appendBotMessage(html);
  }

  function renderQuickReplies(list) {
    quickArea.innerHTML = '';
    if (!Array.isArray(list) || !list.length) return;
    list.forEach((q,i) => {
      const chip = document.createElement('button');
      chip.className = 'chip' + (i===0 ? ' primary' : '');
      chip.innerText = q;
      chip.addEventListener('click', () => { submitMessage(q); });
      quickArea.appendChild(chip);
    });
  }

  async function submitMessage(text) {
    if (!text || !text.trim()) return;
    appendUserMessage(text);
    chatInput.value = '';
    statusEl.innerText = 'Thinking...';
    renderQuickReplies([]); // clear
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({message: text})
      });
      const j = await res.json();
      statusEl.innerText = 'Ready';

      // handle emergency specially
      if (j.type === 'emergency') {
        appendBotMessage(`<strong>Emergency advice:</strong><div>${escapeHtml(j.text)}</div>`);
        renderQuickReplies(j.quick_replies || []);
        return;
      }

      if (j.type === 'card') {
        renderCard(j);
      } else {
        appendBotMessage(escapeHtml(j.text || ''));
      }

      renderQuickReplies(j.quick_replies || []);
    } catch (err) {
      statusEl.innerText = 'Error';
      appendBotMessage('Sorry — chat service failed. Try again later.');
      console.error('chat error', err);
    }
  }

  // send button + enter
  sendBtn.addEventListener('click', () => submitMessage(chatInput.value));
  chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submitMessage(chatInput.value); } });

  // initial greeting from bot
  (async function init(){
    // small welcome
    appendBotMessage("<strong>Hello — I'm HeartAssist.</strong> I can explain causes, prevention, tests, and when to seek help. Try: <em>Causes</em>, <em>Prevention tips</em>, or <em>When to see a doctor</em>.");
    renderQuickReplies(["Causes","Prevention tips","When to see a doctor"]);
  })();
});
    