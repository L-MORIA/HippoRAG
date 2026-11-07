"use strict";

const HISTORY_KEY = "hipporag.history.v1";
const API_KEY_STORAGE = "hipporag.apiKey";
const MAX_HISTORY = 50;

const FALLBACK_MODES = [
  {
    id: "5",
    label: "Быстрый",
    description: "BM25 + русский cross-encoder. Обычный поиск по книгам.",
    expert: false,
    latency: "~0,2 с",
    needs_llm: false,
  },
  {
    id: "2",
    label: "Граф",
    description: "Entity linking + PPR по графу знаний. Для связей между понятиями.",
    expert: true,
    latency: "~0,7 с",
    needs_llm: false,
  },
  {
    id: "1",
    label: "Максимальный охват",
    description: "Multi-vector + переформулировка запроса через LLM. Медленнее, но находит больше.",
    expert: true,
    latency: "~6–13 с",
    needs_llm: true,
  },
];

const state = {
  config: null,
  modes: [],
  history: [],
  current: null,
  apiKey: "",
  pendingRetry: false,
  lastFeedbackRating: null,
  loading: false,
};

const $ = (selector) => document.querySelector(selector);

const elements = {};

function cacheElements() {
  Object.assign(elements, {
    status: $("#server-status"),
    settingsButton: $("#settings-btn"),
    form: $("#ask-form"),
    question: $("#question"),
    modePicker: $("#mode-picker"),
    askButton: $("#ask-btn"),
    errorBox: $("#error-box"),
    result: $("#result"),
    resultMode: $("#result-mode"),
    confidence: $("#confidence-badge"),
    latency: $("#latency"),
    abstainBanner: $("#abstain-banner"),
    refusalReason: $("#refusal-reason"),
    answer: $("#answer"),
    reformulated: $("#reformulated"),
    exportMd: $("#export-md"),
    exportJson: $("#export-json"),
    copyAnswer: $("#copy-answer"),
    citations: $("#citations"),
    fragments: $("#fragments"),
    fragmentsDetails: $("#fragments-details"),
    feedbackBlock: $("#feedback-block"),
    feedbackCommentWrap: $("#feedback-comment-wrap"),
    feedbackComment: $("#feedback-comment"),
    feedbackSend: $("#feedback-send"),
    feedbackStatus: $("#feedback-status"),
    historyList: $("#history-list"),
    clearHistory: $("#clear-history"),
    exportHistoryJson: $("#export-history-json"),
    exportHistoryMd: $("#export-history-md"),
    keyDialog: $("#key-dialog"),
    keyForm: $("#key-form"),
    keyInput: $("#key-input"),
    keyError: $("#key-error"),
    keyCancel: $("#key-cancel"),
  });
}

function safeGet(storage, key) {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(storage, key, value) {
  try {
    storage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} мс`;
  return `${seconds.toFixed(2).replace(".", ",")} с`;
}

function setStatus(text, kind = "muted") {
  if (!elements.status) return;
  elements.status.textContent = text;
  elements.status.className = `status status-${kind}`;
}

function showError(message) {
  if (!elements.errorBox) return;
  elements.errorBox.textContent = message;
  elements.errorBox.classList.remove("hidden");
}

function clearError() {
  elements.errorBox?.classList.add("hidden");
  if (elements.errorBox) elements.errorBox.textContent = "";
}

function setLoading(value) {
  state.loading = value;
  elements.askButton?.classList.toggle("is-loading", value);
  if (elements.askButton) elements.askButton.disabled = value;
  if (elements.question) elements.question.disabled = value;
}

async function fetchJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch (error) {
    const networkError = new Error("Сервер недоступен. Проверьте, что он запущен.");
    networkError.cause = error;
    throw networkError;
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail = payload?.detail || payload?.error || `HTTP ${response.status}`;
    const error = new Error(String(detail));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function handleAuthError(error) {
  if (error?.status !== 401) return false;
  openKeyDialog(true);
  return true;
}

function loadHistory() {
  const raw = safeGet(localStorage, HISTORY_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.slice(0, MAX_HISTORY) : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  safeSet(localStorage, HISTORY_KEY, JSON.stringify(state.history.slice(0, MAX_HISTORY)));
}

function renderModes(defaultMode) {
  const modes = state.modes.length ? state.modes : FALLBACK_MODES;
  elements.modePicker.innerHTML = "";
  const selected = modes.some((mode) => mode.id === defaultMode) ? defaultMode : modes[0]?.id;

  modes.forEach((mode) => {
    const card = document.createElement("label");
    card.className = "mode-card";

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "mode";
    input.value = mode.id;
    input.checked = mode.id === selected;

    const top = document.createElement("span");
    top.className = "mode-card-top";
    const title = document.createElement("strong");
    title.textContent = mode.label;
    const badge = document.createElement("span");
    badge.className = "mode-badge";
    badge.textContent = mode.expert ? "эксперт" : "по умолчанию";
    top.append(title, badge);

    const description = document.createElement("small");
    description.textContent = mode.description;

    const latency = document.createElement("small");
    latency.textContent = `Режим ${mode.id} · ${mode.latency}`;

    card.append(input, top, description, latency);
    elements.modePicker.append(card);
  });
}

function getSelectedMode() {
  const checked = elements.modePicker?.querySelector('input[name="mode"]:checked');
  return checked?.value || state.config?.default_mode || "5";
}

function setSelectedMode(modeId) {
  const input = elements.modePicker?.querySelector(`input[name="mode"][value="${CSS.escape(modeId)}"]`);
  if (input) input.checked = true;
}

function updateKeyButton() {
  if (!elements.settingsButton) return;
  elements.settingsButton.textContent = state.apiKey ? "Ключ задан" : "Ключ доступа";
}

function openKeyDialog(retry = false) {
  state.pendingRetry = retry;
  if (!elements.keyDialog) return;
  if (elements.keyInput) elements.keyInput.value = state.apiKey || "";
  elements.keyError?.classList.add("hidden");
  if (typeof elements.keyDialog.showModal === "function") {
    if (!elements.keyDialog.open) elements.keyDialog.showModal();
  } else {
    elements.keyDialog.setAttribute("open", "");
  }
  elements.keyInput?.focus();
}

function closeKeyDialog() {
  if (!elements.keyDialog) return;
  if (typeof elements.keyDialog.close === "function" && elements.keyDialog.open) {
    elements.keyDialog.close();
  } else {
    elements.keyDialog.removeAttribute("open");
  }
}

async function saveKey(event) {
  event.preventDefault();
  const key = elements.keyInput?.value.trim() || "";
  if (!key) {
    if (elements.keyError) {
      elements.keyError.textContent = "Введите ключ.";
      elements.keyError.classList.remove("hidden");
    }
    return;
  }

  const previous = state.apiKey;
  state.apiKey = key;
  try {
    await fetchJson("/health");
  } catch (error) {
    state.apiKey = previous;
    if (elements.keyError) {
      elements.keyError.textContent =
        error.status === 401 ? "Ключ не подошёл. Проверьте API_KEY." : error.message;
      elements.keyError.classList.remove("hidden");
    }
    return;
  }

  safeSet(sessionStorage, API_KEY_STORAGE, key);
  updateKeyButton();
  closeKeyDialog();
  await refreshHealth();
  if (state.pendingRetry) {
    state.pendingRetry = false;
    submitQuestion();
  }
}

async function refreshHealth() {
  if (!state.config) return;
  try {
    const health = await fetchJson("/health");
    setStatus(`Индекс: ${health.chunks} чанков`, "ok");
  } catch (error) {
    if (error.status === 401) {
      setStatus("Нужен ключ доступа", "warn");
      if (!state.apiKey) openKeyDialog(false);
    } else if (error.status === 503) {
      setStatus("Сервер без API_KEY", "bad");
    } else {
      setStatus("Сервер недоступен", "bad");
    }
  }
}

function renderAnswer(text) {
  const escaped = escapeHtml(text || "Нет текста ответа.");
  const withRefs = escaped.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_match, group) =>
    group
      .split(",")
      .map((part) => {
        const id = part.trim();
        return `<button type="button" class="cite-ref" data-cite="${id}" aria-label="Источник ${id}">[${id}]</button>`;
      })
      .join(""),
  );
  return withRefs
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function renderCitations(data) {
  elements.citations.innerHTML = "";
  if (!data.citations?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Ответ не содержит использованных цитат.";
    elements.citations.append(empty);
    return;
  }

  data.citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    card.dataset.cite = String(citation.id);

    const number = document.createElement("span");
    number.className = "citation-number";
    number.textContent = citation.id;

    const body = document.createElement("div");
    const title = document.createElement("p");
    title.className = "citation-title";
    title.textContent = citation.doc_title || citation.source_file || "Источник";

    const meta = document.createElement("p");
    meta.className = "citation-meta";
    meta.textContent = `стр. ${citation.page_num || "—"}${
      citation.section ? ` · ${citation.section}` : ""
    }`;

    const snippet = document.createElement("p");
    snippet.className = "citation-snippet";
    snippet.textContent = citation.text_snippet || "";

    body.append(title, meta, snippet);
    card.append(number, body);
    card.addEventListener("click", () => scrollToCitation(citation.id));
    elements.citations.append(card);
  });
}

function renderFragments(data) {
  elements.fragments.innerHTML = "";
  if (!data.fragments?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Фрагменты не найдены.";
    elements.fragments.append(empty);
    return;
  }

  data.fragments.forEach((fragment) => {
    const card = document.createElement("article");
    card.className = "fragment-card";
    card.id = `fragment-${fragment.id}`;

    const head = document.createElement("div");
    head.className = "fragment-head";
    const source = document.createElement("span");
    source.textContent = `[${fragment.id}] ${fragment.doc_title || "Источник"}, стр. ${
      fragment.page_num || "—"
    }`;
    const score = document.createElement("span");
    score.textContent = `score ${Number(fragment.score || 0).toFixed(3)}`;
    head.append(source, score);

    const text = document.createElement("p");
    text.className = "fragment-text";
    text.textContent = fragment.text || "";

    card.append(head, text);
    elements.fragments.append(card);
  });
}

function scrollToCitation(citationId) {
  document.querySelectorAll(".citation-card").forEach((card) => {
    card.classList.toggle("highlight", card.dataset.cite === String(citationId));
  });
  const card = document.querySelector(`.citation-card[data-cite="${CSS.escape(String(citationId))}"]`);
  card?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function confidenceView(data) {
  if (data.abstained) return { text: "Отказ", className: "badge badge-bad" };
  if (Number(data.confidence) >= 1) return { text: "Заземлено", className: "badge badge-ok" };
  if (Number(data.confidence) > 0) return { text: "Есть замечания", className: "badge badge-warn" };
  return { text: "Нет цитат", className: "badge badge-bad" };
}

function resetFeedback() {
  state.lastFeedbackRating = null;
  elements.feedbackBlock?.querySelectorAll(".feedback-button").forEach((button) => {
    button.classList.remove("is-selected");
  });
  elements.feedbackCommentWrap?.classList.add("hidden");
  if (elements.feedbackComment) elements.feedbackComment.value = "";
  if (elements.feedbackStatus) elements.feedbackStatus.textContent = "";
}

// F-8: причина отказа — из grounding_errors, а не общий текст для всех случаев.
const REFUSAL_REASONS = {
  no_fragments:
    "По запросу не нашлось подходящих фрагментов — в книгах нет этого факта. Попробуйте переформулировать вопрос.",
  llm_unavailable_or_empty:
    "Языковая модель временно недоступна или ответила пустотой — попробуйте ещё раз через минуту.",
  answer_not_verified:
    "Ответ не удалось проверить по источникам — покажите фрагменты ниже и перепроверьте вручную.",
};

function renderRefusalReason(data) {
  if (!elements.refusalReason) return;
  const errors = Array.isArray(data.grounding_errors) ? data.grounding_errors : [];
  if (!data.abstained || errors.length === 0) {
    elements.refusalReason.classList.add("hidden");
    elements.refusalReason.textContent = "";
    return;
  }
  const known = errors.find((code) => REFUSAL_REASONS[code]);
  const reason = known
    ? REFUSAL_REASONS[known]
    : "Ответ не удалось проверить по источникам — покажите фрагменты ниже и перепроверьте вручную.";
  elements.refusalReason.textContent = `Причина: ${reason}`;
  elements.refusalReason.classList.remove("hidden");
}

function renderResult(data) {
  state.current = data;
  elements.result.classList.remove("hidden");
  elements.resultMode.textContent = `Режим ${data.mode} · ${data.mode_label}${
    data.expert ? " · эксперт" : ""
  }`;
  elements.answer.innerHTML = renderAnswer(data.answer);

  const confidence = confidenceView(data);
  elements.confidence.textContent = confidence.text;
  elements.confidence.className = confidence.className;
  elements.latency.textContent = `${formatSeconds(data.processing_time)} · поиск ${formatSeconds(
    data.retrieval_time,
  )}`;

  elements.abstainBanner.classList.toggle("hidden", !data.abstained);
  renderRefusalReason(data);
  elements.reformulated.classList.toggle(
    "hidden",
    !data.reformulated_query || data.reformulated_query === data.query,
  );
  if (data.reformulated_query && data.reformulated_query !== data.query) {
    elements.reformulated.textContent = `Поиск шёл по переформулировке: «${data.reformulated_query}»`;
  }

  renderCitations(data);
  renderFragments(data);
  elements.fragmentsDetails.open = false;
  resetFeedback();
}

function historyEntry(data) {
  return {
    ...data,
    history_id:
      data.history_id ||
      (globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : String(Date.now())),
    history_timestamp: data.history_timestamp || new Date().toISOString(),
    feedback: data.feedback || null,
  };
}

function addHistory(data) {
  const entry = historyEntry(data);
  state.current = entry;
  state.history = [entry, ...state.history.filter((item) => item.history_id !== entry.history_id)];
  state.history = state.history.slice(0, MAX_HISTORY);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  elements.historyList.innerHTML = "";
  if (!state.history.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Пока пусто. Задайте первый вопрос.";
    elements.historyList.append(empty);
    return;
  }

  state.history.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";

    const query = document.createElement("span");
    query.className = "history-query";
    query.textContent = entry.query;

    const meta = document.createElement("span");
    meta.className = "history-meta";
    const mode = document.createElement("span");
    mode.textContent = `Режим ${entry.mode} · ${entry.mode_label}`;
    const date = document.createElement("span");
    date.textContent = `${formatDate(entry.history_timestamp)}${
      entry.feedback === "up" ? " · 👍" : entry.feedback === "down" ? " · 👎" : ""
    }`;
    meta.append(mode, date);

    button.append(query, meta);
    button.addEventListener("click", () => restoreHistory(entry));
    elements.historyList.append(button);
  });
}

function restoreHistory(entry) {
  state.current = entry;
  elements.question.value = entry.query || "";
  setSelectedMode(entry.mode);
  renderResult(entry);
  elements.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitQuestion() {
  const question = elements.question.value.trim();
  if (!question) {
    showError("Введите вопрос.");
    elements.question.focus();
    return;
  }

  clearError();
  setLoading(true);
  try {
    const data = await fetchJson("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        mode: getSelectedMode(),
        top_k: 5,
      }),
    });
    renderResult(data);
    addHistory(data);
    if (window.innerWidth < 980) {
      elements.result.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    if (!handleAuthError(error)) showError(error.message);
  } finally {
    setLoading(false);
  }
}

function markHistoryFeedback(rating) {
  if (!state.current?.history_id) return;
  const entry = state.history.find((item) => item.history_id === state.current.history_id);
  if (entry) entry.feedback = rating;
  state.current.feedback = rating;
  saveHistory();
  renderHistory();
}

async function sendFeedback() {
  if (!state.current) return;
  const rating = state.lastFeedbackRating;
  if (!rating) {
    if (elements.feedbackStatus) elements.feedbackStatus.textContent = "Сначала выберите оценку.";
    return;
  }

  elements.feedbackSend.disabled = true;
  try {
    await fetchJson("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        rating,
        query: state.current.query,
        answer: state.current.answer || "",
        mode: state.current.mode || "5",
        comment: elements.feedbackComment.value.trim(),
        citations: state.current.citations || [],
      }),
    });
    markHistoryFeedback(rating);
    if (elements.feedbackStatus) elements.feedbackStatus.textContent = "Спасибо! Оценка сохранена.";
    elements.feedbackComment.value = "";
    elements.feedbackCommentWrap.classList.add("hidden");
  } catch (error) {
    if (!handleAuthError(error) && elements.feedbackStatus) {
      elements.feedbackStatus.textContent = `Не удалось сохранить: ${error.message}`;
    }
  } finally {
    elements.feedbackSend.disabled = false;
  }
}

function download(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function answerToMarkdown(data) {
  const lines = [
    `# ${data.query || "Вопрос"}`,
    "",
    `**Режим:** ${data.mode} — ${data.mode_label}`,
    `**Дата:** ${formatDate(data.history_timestamp || data.timestamp)}`,
    `**Заземление:** ${data.abstained ? "отказ" : `${Math.round(Number(data.confidence) * 100)} %`}`,
    "",
    "## Ответ",
    "",
    data.answer || "Нет текста ответа.",
    "",
    "## Источники",
    "",
  ];

  if (data.citations?.length) {
    data.citations.forEach((citation) => {
      lines.push(
        `${citation.id}. **${citation.doc_title || citation.source_file}**, стр. ${
          citation.page_num || "—"
        }${citation.section ? `, ${citation.section}` : ""}`,
      );
      if (citation.text_snippet) lines.push(`   > ${citation.text_snippet}`);
    });
  } else {
    lines.push("Нет использованных цитат.");
  }

  return lines.join("\n");
}

function exportCurrentMarkdown() {
  if (!state.current) return;
  const stamp = new Date().toISOString().slice(0, 10);
  download(`hipporag-answer-${stamp}.md`, answerToMarkdown(state.current), "text/markdown;charset=utf-8");
}

function exportCurrentJson() {
  if (!state.current) return;
  const stamp = new Date().toISOString().slice(0, 10);
  download(
    `hipporag-answer-${stamp}.json`,
    JSON.stringify(state.current, null, 2),
    "application/json;charset=utf-8",
  );
}

function exportHistoryJson() {
  const stamp = new Date().toISOString().slice(0, 10);
  download(
    `hipporag-history-${stamp}.json`,
    JSON.stringify(state.history, null, 2),
    "application/json;charset=utf-8",
  );
}

function exportHistoryMarkdown() {
  const stamp = new Date().toISOString().slice(0, 10);
  const body = state.history.map(answerToMarkdown).join("\n\n---\n\n");
  download(`hipporag-history-${stamp}.md`, body || "История пуста.", "text/markdown;charset=utf-8");
}

async function copyAnswer() {
  if (!state.current?.answer) return;
  const text = state.current.answer;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const label = elements.copyAnswer.textContent;
  elements.copyAnswer.textContent = "Скопировано";
  window.setTimeout(() => {
    elements.copyAnswer.textContent = label;
  }, 1400);
}

function bindEvents() {
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion();
  });

  elements.question.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      submitQuestion();
    }
  });

  elements.answer.addEventListener("click", (event) => {
    const button = event.target.closest(".cite-ref");
    if (button) scrollToCitation(button.dataset.cite);
  });

  elements.settingsButton.addEventListener("click", () => openKeyDialog(false));
  elements.keyForm.addEventListener("submit", saveKey);
  elements.keyCancel.addEventListener("click", closeKeyDialog);

  elements.exportMd.addEventListener("click", exportCurrentMarkdown);
  elements.exportJson.addEventListener("click", exportCurrentJson);
  elements.copyAnswer.addEventListener("click", copyAnswer);
  elements.exportHistoryJson.addEventListener("click", exportHistoryJson);
  elements.exportHistoryMd.addEventListener("click", exportHistoryMarkdown);

  elements.clearHistory.addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });

  elements.feedbackBlock.addEventListener("click", (event) => {
    const button = event.target.closest(".feedback-button");
    if (!button) return;
    state.lastFeedbackRating = button.dataset.rating;
    elements.feedbackBlock.querySelectorAll(".feedback-button").forEach((item) => {
      item.classList.toggle("is-selected", item === button);
    });
    elements.feedbackCommentWrap.classList.remove("hidden");
    if (elements.feedbackStatus) {
      elements.feedbackStatus.textContent = "Оценка выбрана — добавьте комментарий или отправьте.";
    }
  });

  elements.feedbackSend.addEventListener("click", sendFeedback);
}

async function init() {
  cacheElements();
  state.apiKey = safeGet(sessionStorage, API_KEY_STORAGE) || "";
  state.history = loadHistory();
  renderHistory();
  bindEvents();
  updateKeyButton();

  try {
    const config = await fetchJson("/ui/config");
    state.config = config;
    state.modes = config.modes?.length ? config.modes : FALLBACK_MODES;
    renderModes(config.default_mode);
    updateKeyButton();
    await refreshHealth();
  } catch (error) {
    state.modes = FALLBACK_MODES;
    renderModes("5");
    setStatus("Сервер недоступен", "bad");
    showError(error.message);
  }
}

init();
