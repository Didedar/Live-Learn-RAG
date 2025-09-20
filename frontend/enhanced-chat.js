// Enhanced Chat with Learning System — integrated with UI helpers (theme, sidebar, input UX)
class EnhancedChat {
  constructor() {
    // ---- API base (can be overridden via <meta name="api-base" content="/api">) ----
    const metaBase = document.querySelector('meta[name="api-base"]')?.content?.trim();
    this.apiBase = metaBase || 'http://localhost:8000/api/v1';

    this.currentMessageId = null;
    this.currentSessionId = this.generateSessionId();
    this.messages = [];
    this.learningStats = {};

    this.init();
  }

  init() {
    // Apply saved theme on boot (from user's snippet)
    this.applySavedTheme();

    this.bindEvents();
    this.loadLearningStats();
    this.setupAutoResize();
    this.loadChatHistory();
    this.updateCharCount();

    // Focus input on load
    window.addEventListener('load', () => {
      document.getElementById('messageInput')?.focus();
    });

    // Refresh stats every 30s
    setInterval(() => this.loadLearningStats(), 30000);
  }

  // ===== THEME =====
  applySavedTheme() {
    const html = document.documentElement;
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.classList.toggle('dark', savedTheme === 'dark');
  }

  toggleTheme() {
    const html = document.documentElement;
    html.classList.toggle('dark');
    localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
  }

  // ===== EVENTS =====
  bindEvents() {
    // Send message
    document.getElementById('sendBtn')?.addEventListener('click', () => this.sendMessage());
    document.getElementById('messageInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // New chat
    document.getElementById('newChatBtn')?.addEventListener('click', () => this.newChat());

    // Theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', () => this.toggleTheme());

    // Sidebar toggle (mobile)
    document.getElementById('sidebarToggle')?.addEventListener('click', () => this.toggleSidebar());

    // Sidebar overlay click-to-close (optional element)
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    sidebarOverlay?.addEventListener('click', () => this.toggleSidebar(true));

    // Stats refresh
    document.getElementById('refreshStatsBtn')?.addEventListener('click', () => this.loadLearningStats());

    // Feedback modal
    this.bindFeedbackEvents();

    // Char counter & input UX
    const input = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    input?.addEventListener('input', () => {
      this.updateCharCount();
      // Disable send when empty/too long
      if (sendBtn) {
        const len = input.value.length;
        sendBtn.disabled = len === 0 || len > 2000;
      }
    });

    // Paste clamp to 2000 (from user's snippet)
    input?.addEventListener('paste', () => {
      setTimeout(() => {
        if (input.value.length > 2000) {
          input.value = input.value.substring(0, 2000);
          this.updateCharCount();
          // Optional: toast
          this.showToast('Текст обрезан до 2000 символов', 'warning');
        }
      }, 0);
    });

    // ESC closes feedback modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const modal = document.getElementById('feedbackModal');
        if (modal && modal.classList.contains('flex')) this.closeFeedbackModal();
      }
    });

    // Ripple effect for gradient buttons (optional visual from user's snippet)
    document.querySelectorAll('.bg-gradient-to-r').forEach(btn => {
      btn.addEventListener('click', function (e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');
        this.style.position = 'relative';
        this.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
      });
    });

    // Inject ripple CSS once
    if (!document.getElementById('ripple-style')) {
      const style = document.createElement('style');
      style.id = 'ripple-style';
      style.textContent = `
        .ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,.6);transform:scale(0);animation:ripple-animation .6s linear;pointer-events:none}
        @keyframes ripple-animation{to{transform:scale(4);opacity:0}}
      `;
      document.head.appendChild(style);
    }
  }

  bindFeedbackEvents() {
    const modal = document.getElementById('feedbackModal');
    const cancelBtn = document.getElementById('cancelFeedbackBtn');
    const submitBtn = document.getElementById('submitFeedbackBtn');

    cancelBtn?.addEventListener('click', () => this.closeFeedbackModal());
    modal?.addEventListener('click', (e) => { if (e.target === modal) this.closeFeedbackModal(); });

    document.querySelectorAll('.feedback-type-btn').forEach(btn => {
      btn.addEventListener('click', () => this.selectFeedbackType(btn));
    });

    submitBtn?.addEventListener('click', () => this.submitFeedback());
  }

  setupAutoResize() {
    const textarea = document.getElementById('messageInput');
    textarea?.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    });
  }

  generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  // ===== NETWORK =====
  async sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input?.value.trim();
    if (!message) return;

    // Disable send while processing
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.disabled = true;

    // Add user message
    this.addMessage('user', message);
    input.value = '';
    input.style.height = 'auto';
    this.updateCharCount();

    // Typing indicator
    this.showTypingIndicator();

    try {
      // NOTE: endpoints preserved from the original file
      const response = await fetch(`${this.apiBase}/feedback/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: message, session_id: this.currentSessionId, top_k: 6 })
      });

      const data = await response.json();

      if (response.ok) {
        this.currentMessageId = data.message_id;
        this.addMessage('assistant', data.answer, data.contexts);
        this.updateTokenCounter(message + (data.answer || ''));
      } else {
        throw new Error(data.detail || 'Ошибка сервера');
      }
    } catch (error) {
      console.error('Error:', error);
      this.addMessage('assistant', `Извините, произошла ошибка: ${error.message}`, []);
    } finally {
      this.hideTypingIndicator();
      if (sendBtn) sendBtn.disabled = false;
    }
  }

  // ===== MESSAGES =====
    // ===== MESSAGES (pretty) =====
  addMessage(role, content, sources = []) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    const messageId = 'msg_' + Date.now();

    messageDiv.className = `message ${role} message-animation`;

    // meta (время + роль)
    const ts = new Date();
    const metaHtml = `
      <div class="msg-meta">
        <span>${this.prettyTime(ts)}</span>
        ${role === 'assistant' ? '<span>• ИИ-ассистент</span>' : '<span>• Вы</span>'}
        ${this.renderMessageActions(messageId)}
      </div>
    `;

    // аватар
    const avatar = `
      <div class="message-avatar ${role}">
        ${role === 'user'
          ? '<svg class="size-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M12 12c2.2 0 4-1.8 4-4s-1.8-4-4-4-4 1.8-4 4 1.8 4 4 4zm0 2c-2.7 0-8 1.3-8 4v2h16v-2c0-2.7-5.3-4-8-4z"/></svg>'
          : '<svg class="size-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>'}
      </div>
    `;

    // сам пузырь
    const bubble = `
      <div class="bubble ${role}">
        <div class="prose">${this.formatMessage(content)}</div>
        ${sources && sources.length ? this.renderSources(sources) : ''}
      </div>
    `;

    messageDiv.innerHTML = `
      ${avatar}
      <div class="message-content ${role}">
        ${bubble}
        ${metaHtml}
      </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // сохраняем
    this.messages.push({ id: messageId, role, content, sources, timestamp: ts.getTime() });
  }

  // Красивые источники/контекст
  renderSources(sources) {
    try {
      const chips = sources.slice(0, 6).map((s, i) => {
        // поддержим разные формы: строка / {metadata:{...}} / {source:...}
        let label =
          (s?.metadata?.file_name) ||
          (s?.metadata?.source_path) ||
          (s?.metadata?.source) ||
          (s?.source) ||
          (typeof s === 'string' ? s : null) ||
          `Источник ${i+1}`;
        label = String(label).split('/').pop();
        return `
          <span class="chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M21 7L9 19l-5.5-5.5 1.41-1.41L9 16.17 19.59 5.59z"/></svg>
            ${this.escapeHtml(label)}
          </span>
        `;
      }).join('');

      return `<div class="sources">${chips}</div>`;
    } catch {
      return '';
    }
  }

  // Красивое форматирование: **bold**, *italic*, `inline`, ```code```, списки, ссылки
  formatMessage(content) {
    let html = String(content || '');

    // code fences ```lang\n...\n```
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_m, lang, code) => {
      const safe = this.escapeHtml(code);
      const langLabel = this.escapeHtml(lang || 'text');
      return `
        <div class="codeblock">
          <div class="code-header">
            <span>${langLabel}</span>
            <button class="action-btn" onclick="navigator.clipboard.writeText(\`${safe.replace(/`/g,'\\`')}\`)">Копировать</button>
          </div>
          <pre><code>${safe}</code></pre>
        </div>
      `;
    });

    // inline code
    html = html.replace(/`([^`]+?)`/g, (_m, code) => `<code>${this.escapeHtml(code)}</code>`);

    // bold / italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // links
    html = html.replace(/(?<!["'>])(https?:\/\/[^\s)]+)(?![^<]*>)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

    // simple lists (lines starting with -, *, or digits.)
    html = html
      .replace(/(^|\n)\s*[-*]\s+(.*)/g, '$1• $2') // normalize bullet
      .replace(/\n/g, '<br>');

    return html;
  }

  renderMessageActions(messageId) {
    return `
      <div class="message-actions">
        <button class="action-btn" onclick="chat.openFeedbackModal('${messageId}')" title="Фидбэк">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        </button>
        <button class="action-btn" onclick="chat.copyMessage('${messageId}')" title="Копировать">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2z"/></svg>
        </button>
        <button class="action-btn" onclick="chat.regenerateResponse('${messageId}')" title="Перегенерировать">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4 7.58 4 4.01 7.58 4.01 12S7.58 20 12 20c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6S8.69 6 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
        </button>
      </div>
    `;
  }

  showTypingIndicator() {
    const messagesContainer = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message assistant message-animation';
    typingDiv.innerHTML = `
      <div class="message-avatar assistant">
        <svg class="size-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div class="message-content assistant">
        <div class="bubble assistant">
          <div class="typing">
            <span>ИИ печатает</span>
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
        </div>
        <div class="msg-meta"><span>${this.prettyTime(new Date())}</span><span>• ИИ-ассистент</span></div>
      </div>`;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // ===== helpers for the pretty render =====
  prettyTime(d) {
    try {
      const hh = String(d.getHours()).padStart(2,'0');
      const mm = String(d.getMinutes()).padStart(2,'0');
      return `${hh}:${mm}`;
    } catch { return ''; }
  }

  escapeHtml(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');
  }


  renderMessageActions(messageId) {
    return `
      <div class="message-actions">
        <button class="action-btn" onclick="chat.openFeedbackModal('${messageId}')" title="Оставить фидбэк">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        </button>
        <button class="action-btn" onclick="chat.copyMessage('${messageId}')" title="Копировать">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
        </button>
        <button class="action-btn" onclick="chat.regenerateResponse('${messageId}')" title="Повторить">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
        </button>
      </div>
    `;
  }
  ensureTypingStyles() {
    if (document.getElementById('typing-pretty-style')) return;
    const style = document.createElement('style');
    style.id = 'typing-pretty-style';
    style.textContent = `
      .bubble.loading{position:relative;overflow:hidden}
      .bubble.loading::before{
        content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;
        background:linear-gradient(90deg,#60a5fa,#a78bfa,#60a5fa);
        -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;mask-composite: exclude;
        background-size:200% 100%;animation:borderFlow 2.6s linear infinite
      }
      @keyframes borderFlow{0%{background-position:0%}100%{background-position:200%}}
      .loading-progress{
        position:absolute;left:0;top:-1px;width:100%;height:2px;
        background:linear-gradient(90deg,#60a5fa,#a78bfa,#10b981);
        background-size:200% 100%;animation:progress 2.2s linear infinite
      }
      @keyframes progress{0%{background-position:0%}100%{background-position:200%}}
  
      .skeleton-line{
        height:12px;border-radius:6px;margin:.45rem 0;
        background:linear-gradient(90deg,rgba(148,163,184,.25),rgba(148,163,184,.12),rgba(148,163,184,.25));
        background-size:200% 100%;animation:shimmer 1.6s ease-in-out infinite
      }
      .dark .skeleton-line{
        background:linear-gradient(90deg,rgba(148,163,184,.25),rgba(148,163,184,.10),rgba(148,163,184,.25))
      }
      @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
  
      .skeleton-code{
        margin:.55rem 0;border-radius:.6rem;border:1px solid rgba(148,163,184,.25);
        height:36px;overflow:hidden;position:relative
      }
      .skeleton-code::after{
        content:"";position:absolute;inset:0;
        background:linear-gradient(90deg,rgba(148,163,184,.15),rgba(148,163,184,.08),rgba(148,163,184,.15));
        background-size:200% 100%;animation:shimmer 1.6s ease-in-out infinite
      }
  
      .typing-head{display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem;color:#64748b}
      .dark .typing-head{color:#94a3b8}
      .typing-badge{
        font-size:.65rem;padding:.18rem .55rem;border-radius:999px;
        border:1px solid rgba(99,102,241,.35);background:rgba(59,130,246,.08)
      }
      .dark .typing-badge{background:rgba(99,102,241,.12)}
  
      .typing-dots{display:inline-flex;gap:.3rem;align-items:center}
      .typing-dots i{
        width:6px;height:6px;border-radius:999px;background:currentColor;opacity:.8;
        transform:translateY(0);animation:bounce 1.15s infinite
      }
      .typing-dots i:nth-child(2){animation-delay:.15s}
      .typing-dots i:nth-child(3){animation-delay:.3s}
      @keyframes bounce{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-3px);opacity:1}}
  
      .avatar-pulse{position:relative}
      .avatar-pulse::after{
        content:"";position:absolute;inset:-4px;border-radius:12px;
        border:2px solid rgba(99,102,241,.35);animation:avatarPulse 1.8s ease-out infinite
      }
      @keyframes avatarPulse{0%{opacity:.6;transform:scale(.95)}100%{opacity:0;transform:scale(1.2)}}
  
      .fade-out{animation:fadeOut .18s ease both}
      @keyframes fadeOut{to{opacity:0;transform:translateY(4px)}}
    `;
    document.head.appendChild(style);
  }
  // ===== TYPING INDICATOR =====
  showTypingIndicator() {
    this.ensureTypingStyles();
    const messagesContainer = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message assistant message-animation';
  
    // случайная ширина строк скелетона для живости
    const lines = Array.from({length: 3 + Math.floor(Math.random()*2)})
      .map((_,i) => {
        const w = 85 - i*12 - Math.floor(Math.random()*8); // %
        return `<div class="skeleton-line" style="width:${w}%"></div>`;
      }).join('');
  
    typingDiv.innerHTML = `
      <div class="message-avatar assistant avatar-pulse">
        <svg class="size-4 text-white" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="message-content assistant">
        <div class="bubble assistant loading">
          <div class="loading-progress"></div>
          <div class="typing-head">
            <span class="typing-badge">думаю</span>
            <div class="typing-dots"><i></i><i></i><i></i></div>
          </div>
          ${lines}
          <div class="skeleton-code"></div>
        </div>
        <div class="msg-meta"><span>${this.prettyTime(new Date())}</span><span>• ИИ-ассистент</span></div>
      </div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
  
  // Плавное исчезание индикатора
  hideTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
    if (!el) return;
  }

  // ===== LEARNING STATS =====
  async loadLearningStats() {
    try {
      const response = await fetch(`${this.apiBase}/feedback/stats`);
      const data = await response.json();
      if (response.ok) {
        this.learningStats = data.statistics || data;
        this.updateLearningStatsDisplay();
        this.updateSystemHealth();
      }
    } catch (error) {
      console.error('Error loading learning stats:', error);
    }
  }

  updateLearningStatsDisplay() {
    const stats = this.learningStats;
    const applied = stats.learning_statistics?.applied_feedback || 0;
    const applicationRate = stats.system_health?.application_rate || 0;
    const filterRate = stats.system_health?.filter_rate || 0;

    const appliedEl = document.getElementById('appliedCount');
    const systemHealthEl = document.getElementById('systemHealth');
    const trustEl = document.getElementById('trustLevel');

    if (appliedEl) appliedEl.textContent = applied;

    if (systemHealthEl) {
      if (applicationRate > 0.7) { systemHealthEl.textContent = 'Отлично'; systemHealthEl.className = 'stat-value good'; }
      else if (applicationRate > 0.4) { systemHealthEl.textContent = 'Хорошо'; systemHealthEl.className = 'stat-value warning'; }
      else { systemHealthEl.textContent = 'Низкое'; systemHealthEl.className = 'stat-value error'; }
    }

    if (trustEl) {
      if (filterRate < 0.2) { trustEl.textContent = 'Высокий'; trustEl.className = 'stat-value good'; }
      else if (filterRate < 0.5) { trustEl.textContent = 'Средний'; trustEl.className = 'stat-value warning'; }
      else { trustEl.textContent = 'Низкий'; trustEl.className = 'stat-value error'; }
    }
  }

  updateSystemHealth() {
    const indicator = document.getElementById('healthIndicator');
    if (!indicator) return;
    const stats = this.learningStats.system_health || {};
    const applicationRate = stats.application_rate || 0;
    const filterRate = stats.filter_rate || 0;

    if (applicationRate > 0.5 && filterRate < 0.3) {
      indicator.innerHTML = `<div class="size-2 rounded-full bg-green-500 animate-pulse"></div><span>Система обучается</span>`;
      indicator.className = 'health-indicator healthy';
    } else if (applicationRate > 0.2) {
      indicator.innerHTML = `<div class="size-2 rounded-full bg-yellow-500 animate-pulse"></div><span>Требует внимания</span>`;
      indicator.className = 'health-indicator warning';
    } else {
      indicator.innerHTML = `<div class="size-2 rounded-full bg-red-500 animate-pulse"></div><span>Проблемы с обучением</span>`;
      indicator.className = 'health-indicator error';
    }
  }

  // ===== FEEDBACK =====
  openFeedbackModal(messageId) {
    this.currentFeedbackMessageId = messageId;
    const modal = document.getElementById('feedbackModal');
    modal?.classList.remove('hidden');
    modal?.classList.add('flex');
    this.resetFeedbackForm();
  }

  closeFeedbackModal() {
    const modal = document.getElementById('feedbackModal');
    modal?.classList.add('hidden');
    modal?.classList.remove('flex');
    this.currentFeedbackMessageId = null;
  }

  resetFeedbackForm() {
    document.querySelectorAll('.feedback-type-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('correctionSection')?.classList.add('hidden');
    const corr = document.getElementById('correctionText');
    const reason = document.getElementById('feedbackReason');
    if (corr) corr.value = '';
    if (reason) reason.value = '';
  }

  selectFeedbackType(button) {
    document.querySelectorAll('.feedback-type-btn').forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
    const type = button.dataset.type;
    const correctionSection = document.getElementById('correctionSection');
    if (correctionSection) {
      if (type === 'incorrect') correctionSection.classList.remove('hidden');
      else correctionSection.classList.add('hidden');
    }
  }

  async submitFeedback() {
    const activeBtn = document.querySelector('.feedback-type-btn.active');
    if (!activeBtn) return this.showToast('Выберите тип фидбэка', 'warning');

    const type = activeBtn.dataset.type;
    const correction = document.getElementById('correctionText')?.value.trim() || '';
    const reason = document.getElementById('feedbackReason')?.value.trim() || '';

    if (type === 'incorrect' && !correction) return this.showToast('Введите правильный ответ', 'warning');

    const btn = document.getElementById('submitFeedbackBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Отправляется...'; }

    try {
      const res = await fetch(`${this.apiBase}/feedback/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: this.currentMessageId,
          user_feedback: { label: type, correction_text: correction || null, scope: 'chunk', reason: reason || null }
        })
      });
      const data = await res.json();
      if (res.ok) {
        this.closeFeedbackModal();
        if (data.status === 'applied') this.showLearningToast('success', 'Фидбэк применен!', 'Система стала умнее 🧠');
        else if (data.status === 'queued') this.showLearningToast('warning', 'Фидбэк на проверке', 'Будет рассмотрен экспертами');
        else if (data.status === 'rejected') this.showLearningToast('warning', 'Фидбэк отклонен', data.message || 'Низкий уровень доверия');
        else this.showLearningToast('info', 'Фидбэк получен', 'Спасибо за помощь!');
        setTimeout(() => this.loadLearningStats(), 1000);
      } else {
        throw new Error(data.detail || 'Ошибка отправки фидбэка');
      }
    } catch (e) {
      this.showToast(`Ошибка: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'Отправить фидбэк'; }
    }
  }

  // ===== TOASTS =====
  showLearningToast(type, title, message) {
    const toast = document.getElementById('learningToast');
    if (!toast) return;
    const colors = { success: 'bg-green-600', warning: 'bg-yellow-600', error: 'bg-red-600', info: 'bg-blue-600' };
    toast.innerHTML = `
      <div class="${colors[type] || colors.info} text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 max-w-sm">
        <div class="size-6 rounded-full bg-white/20 flex items-center justify-center">
          <svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4"/></svg>
        </div>
        <div class="flex-1"><div class="font-medium text-sm">${title}</div><div class="text-xs opacity-90">${message}</div></div>
        <button class="text-white/80 hover:text-white" onclick="hideLearningToast()"><svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg></button>
      </div>`;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 5000);
  }

  showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="flex items-center gap-3"><div class="flex-1">${message}</div><button onclick="this.parentElement.parentElement.remove()"><svg class="size-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg></button></div>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }

  // ===== UTIL =====
  copyMessage(messageId) {
    const m = this.messages.find(x => x.id === messageId);
    if (m) navigator.clipboard.writeText(m.content).then(() => this.showToast('Сообщение скопировано', 'success'));
  }

  async regenerateResponse(messageId) {
    const idx = this.messages.findIndex(m => m.id === messageId);
    if (idx > 0) {
      const userMessage = this.messages[idx - 1];
      if (userMessage.role === 'user') {
        this.showTypingIndicator();
        try {
          const res = await fetch(`${this.apiBase}/feedback/ask`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: userMessage.content, session_id: this.currentSessionId, top_k: 6 })
          });
          const data = await res.json();
          if (res.ok) {
            this.currentMessageId = data.message_id;
            this.addMessage('assistant', data.answer, data.contexts);
          } else { throw new Error(data.detail || 'Ошибка сервера'); }
        } catch (e) {
          this.addMessage('assistant', `Извините, произошла ошибка: ${e.message}`, []);
        } finally { this.hideTypingIndicator(); }
      }
    }
  }

  newChat() {
    this.currentSessionId = this.generateSessionId();
    this.messages = [];
    const messages = document.getElementById('messages');
    if (messages) messages.innerHTML = '';
    // Show a welcome bubble
    this.addMessage('assistant', 'Привет! Я ИИ-ассистент с системой обучения. Задавайте вопросы — а если ответ неточен, оставьте фидбэк, и я стану умнее!');
  }

  toggleSidebar(forceHide = false) {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (!sidebar) return;
    if (forceHide) sidebar.classList.add('hidden');
    else sidebar.classList.toggle('hidden');
    if (overlay) overlay.classList.toggle('hidden');
  }

  updateCharCount() {
    const input = document.getElementById('messageInput');
    const counter = document.getElementById('charCount');
    if (!input || !counter) return;
    const length = input.value.length;
    counter.textContent = `${length} / 2000`;
    if (length > 1800) counter.classList.add('text-red-500'); else counter.classList.remove('text-red-500');
  }

  updateTokenCounter(text) {
    const tokens = Math.ceil((text || '').length / 4);
    const el = document.getElementById('tokenCounter');
    if (el) el.textContent = `${tokens} ток.`;
  }

  loadChatHistory() {
    // Placeholder for future localStorage history
  }
}

// Global helpers for inline onclick
function hideLearningToast() { document.getElementById('learningToast')?.classList.add('hidden'); }

// Init
const chat = new EnhancedChat();

// Respect saved theme also on cold start (in case HTML was served with no class)
(() => {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
  }
})();
// --- Pretty chat bubble styles (inject once) ---
if (!document.getElementById('chat-pretty-style')) {
  const style = document.createElement('style');
  style.id = 'chat-pretty-style';
  style.textContent = `
  .message{display:flex;gap:.75rem;align-items:flex-start}
  .message.message-animation{animation:fade-in .25s ease-out}
  @keyframes fade-in{from{transform:translateY(6px);opacity:0}to{transform:none;opacity:1}}

  .message-avatar{flex:0 0 2rem;height:2rem;border-radius:.8rem;display:flex;align-items:center;justify-content:center;
    box-shadow:0 6px 18px rgba(0,0,0,.08)}
  .message-avatar.user{background:linear-gradient(135deg,#10b981,#059669)}
  .message-avatar.assistant{background:linear-gradient(135deg,#3b82f6,#8b5cf6)}

  .message-content{flex:1;min-width:0;display:grid;gap:.375rem}
  .bubble{position:relative;border-radius:1rem;padding:.875rem 1rem;border:1px solid transparent;
    background:linear-gradient(180deg, rgba(255,255,255,.85), rgba(255,255,255,.75));
    backdrop-filter: blur(8px);
    box-shadow:0 8px 24px rgba(15,23,42,.06)}
  .dark .bubble{background:linear-gradient(180deg, rgba(17,24,39,.7), rgba(31,41,55,.65));}

  .bubble.user{border-image:linear-gradient(90deg,#10b981,#06b6d4) 1}
  .bubble.assistant{border-image:linear-gradient(90deg,#3b82f6,#8b5cf6) 1}

  .bubble .prose{line-height:1.55;color:#111827}
  .dark .bubble .prose{color:#e5e7eb}
  .bubble .prose p{margin:.25rem 0}
  .bubble .prose h1,.bubble .prose h2,.bubble .prose h3{font-weight:700;margin:.5rem 0 .25rem}
  .bubble .prose ul{margin:.25rem 0 .25rem 1rem}
  .bubble .prose li{margin:.125rem 0}
  .bubble .prose a{color:#2563eb;text-decoration:underline}
  .dark .bubble .prose a{color:#93c5fd}

  /* inline code */
  .bubble .prose code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;
    background:rgba(2,6,23,.06); padding:.1rem .35rem;border-radius:.35rem;font-size:.9em}
  .dark .bubble .prose code{background:rgba(255,255,255,.08)}

  /* code blocks */
  .codeblock{margin:.5rem 0;border-radius:.8rem;border:1px solid rgba(148,163,184,.25);overflow:auto}
  .codeblock pre{margin:0;padding:.75rem 1rem;font-size:.9rem;line-height:1.5}
  .codeblock .code-header{display:flex;justify-content:space-between;align-items:center;padding:.45rem .7rem;
    font-size:.75rem;border-bottom:1px solid rgba(148,163,184,.25);
    background:linear-gradient(180deg,rgba(241,245,249,.8),rgba(226,232,240,.6))}
  .dark .codeblock .code-header{background:linear-gradient(180deg,rgba(17,24,39,.7),rgba(31,41,55,.6))}

  .msg-meta{display:flex;gap:.5rem;align-items:center;font-size:.72rem;color:#64748b}
  .dark .msg-meta{color:#94a3b8}

  .message-actions{display:flex;gap:.25rem;margin-left:.25rem}
  .action-btn{padding:.35rem;border-radius:.5rem;border:1px solid rgba(148,163,184,.25);background:transparent}
  .action-btn:hover{background:rgba(2,6,23,.04)}
  .dark .action-btn:hover{background:rgba(255,255,255,.06)}

  .sources{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
  .chip{display:inline-flex;align-items:center;gap:.35rem;border-radius:999px;padding:.25rem .55rem;
    border:1px dashed rgba(148,163,184,.45);font-size:.72rem;white-space:nowrap}
  .chip svg{opacity:.7}

  /* typing indicator */
  .typing{display:inline-flex;align-items:center;gap:.5rem}
  .typing .dot{width:.4rem;height:.4rem;border-radius:999px;background:currentColor;opacity:.6;animation:blink 1.1s infinite}
  .typing .dot:nth-child(2){animation-delay:.15s}
  .typing .dot:nth-child(3){animation-delay:.3s}
  @keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

  /* alignment */
  .message.user{flex-direction:row-reverse}
  .message.user .message-content{align-items:flex-end}
  .message.user .bubble{background:linear-gradient(180deg, rgba(236,253,245,.9), rgba(219,234,254,.85))}
  .dark .message.user .bubble{background:linear-gradient(180deg, rgba(6,78,59,.45), rgba(30,58,138,.45))}

  `;
  document.head.appendChild(style);
}
