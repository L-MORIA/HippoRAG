"use strict";

const API_KEY_STORAGE = "hipporag.apiKey";
const REFRESH_MS = 5000;

const state = {
  apiKey: "",
  timer: null,
};

const $ = (selector) => document.querySelector(selector);

const elements = {};

function cacheElements() {
  Object.assign(elements, {
    status: $("#dashboard-status"),
    refresh: $("#refresh-btn"),
    keyPanel: $("#key-panel"),
    keyInput: $("#key-input"),
    keySave: $("#key-save"),
    keyError: $("#key-error"),
    cards: $("#metric-cards"),
    healthRows: $("#health-rows"),
    routeRows: $("#route-rows"),
    stageRows: $("#stage-rows"),
    emptyReasons: $("#empty-reasons"),
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

function formatNumber(value) {
  const number = Number(value ?? 0);
  return new Intl.NumberFormat("ru-RU").format(Number.isFinite(number) ? number : 0);
}

function formatMs(value) {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return "—";
  if (number >= 1000) return `${(number / 1000).toFixed(2)} с`;
  return `${number.toFixed(1)} мс`;
}

function formatUptime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return `${hours} ч ${minutes} мин`;
}

function setStatus(text, kind) {
  elements.status.textContent = text;
  elements.status.className = `status status-${kind}`;
}

function showKeyPanel(message) {
  elements.keyPanel.classList.remove("hidden");
  if (message) {
    elements.keyError.textContent = message;
    elements.keyError.classList.remove("hidden");
  }
}

function hideKeyPanel() {
  elements.keyPanel.classList.add("hidden");
  elements.keyError.classList.add("hidden");
}

async function fetchJson(path) {
  const headers = {};
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  const response = await fetch(path, { headers });
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function addCard(label, value, hint) {
  const card = document.createElement("article");
  card.className = "metric-card";

  const valueNode = document.createElement("p");
  valueNode.className = "metric-value";
  valueNode.textContent = value;

  const labelNode = document.createElement("p");
  labelNode.className = "metric-label";
  labelNode.textContent = label;

  card.append(valueNode, labelNode);
  if (hint) {
    const hintNode = document.createElement("p");
    hintNode.className = "metric-hint";
    hintNode.textContent = hint;
    card.append(hintNode);
  }
  elements.cards.append(card);
}

function renderCards(health, metrics) {
  const latency = metrics.latency || {};
  elements.cards.innerHTML = "";
  addCard("Запросов", formatNumber(metrics.requests_total), `uptime ${formatUptime(metrics.uptime_seconds)}`);
  addCard("Ошибок", formatNumber(metrics.errors_total), `5xx: ${formatNumber(metrics.server_errors_total)}`);
  addCard("Медленных", formatNumber(metrics.slow_requests_total), `порог ${formatMs(metrics.slow_threshold_ms)}`);
  addCard("Таймаутов", formatNumber(metrics.timeouts_total));
  addCard("Пустых ответов", formatNumber(metrics.empty_answers_total));
  addCard("p50", formatMs(latency.p50_ms), `выборка ${formatNumber(latency.count)}`);
  addCard("p95", formatMs(latency.p95_ms), `максимум ${formatMs(latency.max_ms)}`);
  addCard("Чанков", formatNumber(health.chunks), `${formatNumber(health.documents)} документов`);
}

function appendRow(tbody, values) {
  const row = document.createElement("tr");
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  tbody.append(row);
}

function renderHealth(health) {
  elements.healthRows.innerHTML = "";
  appendRow(elements.healthRows, ["Статус", health.status]);
  appendRow(elements.healthRows, ["Uptime", health.uptime]);
  appendRow(elements.healthRows, ["Индекс", `v${health.index_version ?? "—"} · ${health.index_state}`]);
  appendRow(elements.healthRows, ["Операция", health.index_operation]);
  appendRow(elements.healthRows, ["Создан", health.index_created_at || "—"]);
  appendRow(elements.healthRows, ["LLM", health.llm_model || "—"]);
  appendRow(elements.healthRows, ["Эмбеддинги", health.embedding_model || "—"]);
  appendRow(elements.healthRows, ["Документы", formatNumber(health.documents)]);
  appendRow(elements.healthRows, ["Чанки / факты", `${formatNumber(health.chunks)} / ${formatNumber(health.facts)}`]);
  appendRow(elements.healthRows, ["Отказы авторизации", formatNumber(health.auth_failures)]);
}

function renderRoutes(routes) {
  elements.routeRows.innerHTML = "";
  const entries = Object.entries(routes || {});
  if (!entries.length) {
    appendRow(elements.routeRows, ["Пока нет запросов", "—", "—", "—", "—"]);
    return;
  }
  for (const [route, stats] of entries) {
    appendRow(elements.routeRows, [
      route,
      formatNumber(stats.requests),
      formatNumber(stats.errors),
      formatMs(stats.avg_ms),
      formatMs(stats.max_ms),
    ]);
  }
}

function renderStages(stages) {
  elements.stageRows.innerHTML = "";
  const entries = Object.entries(stages || {});
  if (!entries.length) {
    appendRow(elements.stageRows, ["Пока нет данных", "—", "—", "—"]);
    return;
  }
  for (const [stage, stats] of entries) {
    appendRow(elements.stageRows, [
      stage,
      formatNumber(stats.count),
      formatMs(stats.avg_ms),
      formatMs(stats.max_ms),
    ]);
  }
}

function renderEmptyReasons(reasons) {
  elements.emptyReasons.innerHTML = "";
  const entries = Object.entries(reasons || {});
  if (!entries.length) {
    const item = document.createElement("li");
    item.textContent = "Пустых ответов не было.";
    elements.emptyReasons.append(item);
    return;
  }
  for (const [reason, count] of entries) {
    const item = document.createElement("li");
    item.textContent = `${reason}: ${formatNumber(count)}`;
    elements.emptyReasons.append(item);
  }
}

async function refresh() {
  try {
    const [health, metrics] = await Promise.all([fetchJson("/health"), fetchJson("/metrics")]);
    hideKeyPanel();
    setStatus("Сервер доступен", "ok");
    renderCards(health, metrics);
    renderHealth(health);
    renderRoutes(metrics.routes);
    renderStages(metrics.stages);
    renderEmptyReasons(metrics.empty_answers_by_reason);
  } catch (error) {
    if (error.status === 401) {
      setStatus("Нужен ключ доступа", "warn");
      showKeyPanel("Ключ не подошёл или ещё не введён.");
    } else if (error.status === 503) {
      setStatus("Сервер без API_KEY", "bad");
      showKeyPanel("Сервер запущен без API_KEY — запросы не обслуживаются.");
    } else {
      setStatus("Сервер недоступен", "bad");
      showKeyPanel(`Не удалось получить данные: ${error.message}`);
    }
  }
}

function schedule() {
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(() => {
    if (!document.hidden) refresh();
  }, REFRESH_MS);
}

function bindEvents() {
  elements.refresh.addEventListener("click", refresh);
  elements.keySave.addEventListener("click", () => {
    const key = elements.keyInput.value.trim();
    if (!key) {
      showKeyPanel("Введите ключ.");
      return;
    }
    state.apiKey = key;
    safeSet(sessionStorage, API_KEY_STORAGE, key);
    refresh();
  });
  elements.keyInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") elements.keySave.click();
  });
}

function init() {
  cacheElements();
  state.apiKey = safeGet(sessionStorage, API_KEY_STORAGE) || "";
  bindEvents();
  schedule();
  refresh();
}

init();
