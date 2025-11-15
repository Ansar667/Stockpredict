const initialState = (() => {
    const payloadEl = document.getElementById('app-state');
    if (!payloadEl) {
        return {};
    }
    try {
        return JSON.parse(payloadEl.textContent || '{}') || {};
    } catch (error) {
        console.warn('Failed to parse initial payload', error);
        return {};
    }
})();

const AppState = (() => {
    const payloadEl = document.getElementById('app-state');
    let state = initialState || {};
    const syncWindow = () => {
        window.APP_STATE = state;
    };
    syncWindow();
    return {
        get: () => state,
        set: (next) => {
            state = next || {};
            if (payloadEl) {
                payloadEl.textContent = JSON.stringify(state);
            }
            syncWindow();
        },
    };
})();

const alertContainer = document.getElementById('alertContainer');
const metricsCard = document.getElementById('metricCard');
const metricsGrid = document.getElementById('summaryMetrics');
const tickerField = document.querySelector('input[name="ticker"]');
const startField = document.querySelector('input[name="start"]');
const endField = document.querySelector('input[name="end"]');
const tableWrapper = document.getElementById('tableWrapper');
const tableBody = document.getElementById('tableBody');
const tableEmpty = document.getElementById('tableEmptyState');
const tableExpandButton = document.getElementById('expandTableBtn');
const predictionHeading = document.getElementById('predictionHeading');
const exportBtn = document.getElementById('exportPdfBtn');
const formElement = document.getElementById('predict-form');
const errorMetricsCard = document.getElementById('errorMetricsCard');
const errorMetricsBody = document.getElementById('errorMetricsBody');
const recentList = document.getElementById('recentList');
const recentEmpty = document.getElementById('recentEmpty');
const METRIC_LABELS = { mape: 'MAPE', rmse: 'RMSE', mae: 'MAE', mse: 'MSE' };
const RECENT_STORAGE_KEY = 'stock_forecast_recent';
const RECENT_LIMIT = 5;
let recentItems = [];

const formatNumber = (value, fallback = 'вЂ”') => {
    return typeof value === 'number' && !Number.isNaN(value) ? value.toFixed(2) : fallback;
};

const renderAlerts = (payload) => {
    if (!alertContainer) return;
    alertContainer.innerHTML = '';
    const prefix = (window.APP_STATE && window.APP_STATE.summaryPrefix) || 'Summary';
    const appendAlert = (cls, text) => {
        const wrapper = document.createElement('div');
        wrapper.className = `alert ${cls}`;
        wrapper.setAttribute('role', 'status');
        const strong = document.createElement('strong');
        strong.textContent = `${prefix}.`;
        const span = document.createElement('span');
        span.textContent = text;
        wrapper.append(strong, span);
        alertContainer.appendChild(wrapper);
    };
    if (payload.error) {
        appendAlert('alert-error', payload.error);
    }
    if (payload.summaryText) {
        appendAlert('alert-info summary-alert', payload.summaryText);
    }
};

const renderMetrics = (metrics) => {
    if (!metricsCard || !metricsGrid) return;
    metricsGrid.innerHTML = '';
    const hasMetrics = Array.isArray(metrics) && metrics.length > 0;
    metricsCard.hidden = !hasMetrics;
    if (!hasMetrics) {
        return;
    }
    const fallback = AppState.get().noDataLabel || 'вЂ”';
    metrics.forEach((metric) => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        const valueText = metric && metric.value !== null && metric.value !== undefined ? formatNumber(metric.value, fallback) : fallback;
        card.innerHTML = `<span>${metric.label || ''}</span><strong>${valueText}</strong>`;
        metricsGrid.appendChild(card);
    });
};

const renderErrorMetrics = (metricsMap) => {
    if (!errorMetricsCard || !errorMetricsBody) return;
    const entries = Object.entries(metricsMap || {}).filter(([, stats]) => stats);
    if (!entries.length) {
        errorMetricsCard.hidden = true;
        errorMetricsBody.innerHTML = '';
        return;
    }
    errorMetricsCard.hidden = false;
    const fragments = entries
        .map(([modelKey, stats]) => {
            const modelName = modelKey.replace('Pred_', '') || modelKey;
            const metrics = Object.entries(METRIC_LABELS)
                .map(([key, label]) => {
                    const value = stats && typeof stats[key] === 'number' ? stats[key] : null;
                    if (value === null) {
                        return `<span>${label}: вЂ”</span>`;
                    }
                    const formatted = key === 'mape' ? `${value.toFixed(2)}%` : value.toFixed(2);
                    return `<span>${label}: ${formatted}</span>`;
                })
                .join('');
            return `<div class="error-row"><strong>${modelName}</strong>${metrics}</div>`;
        })
        .join('');
    errorMetricsBody.innerHTML = fragments;
};

const loadRecentSearches = () => {
    try {
        const raw = localStorage.getItem(RECENT_STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.filter(
            (item) =>
                item &&
                typeof item.ticker === 'string' &&
                typeof item.start === 'string' &&
                typeof item.end === 'string',
        );
    } catch (error) {
        console.warn('Failed to load recent searches', error);
        return [];
    }
};

const persistRecentSearches = () => {
    try {
        localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(recentItems.slice(0, RECENT_LIMIT)));
    } catch (error) {
        console.warn('Failed to save recent searches', error);
    }
};

const renderRecentSearches = () => {
    if (!recentList || !recentEmpty) return;
    recentList.innerHTML = '';
    if (!recentItems.length) {
        recentEmpty.hidden = false;
        return;
    }
    recentEmpty.hidden = true;
    recentItems.slice(0, RECENT_LIMIT).forEach((item) => {
        const li = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.ticker = item.ticker;
        button.dataset.start = item.start;
        button.dataset.end = item.end;
        button.innerHTML = `<strong>${item.ticker}</strong><span>${item.start} → ${item.end}</span>`;
        li.appendChild(button);
        recentList.appendChild(li);
    });
};

const addRecentSearch = (entry) => {
    if (!entry || !entry.ticker || !entry.start || !entry.end) return;
    const normalizedTicker = entry.ticker.trim().toUpperCase();
    const normalizedStart = entry.start.trim();
    const normalizedEnd = entry.end.trim();
    if (!normalizedTicker || !normalizedStart || !normalizedEnd) return;
    recentItems = recentItems.filter(
        (item) =>
            !(
                item.ticker === normalizedTicker &&
                item.start === normalizedStart &&
                item.end === normalizedEnd
            ),
    );
    recentItems.unshift({
        ticker: normalizedTicker,
        start: normalizedStart,
        end: normalizedEnd,
        timestamp: Date.now(),
    });
    if (recentItems.length > RECENT_LIMIT) {
        recentItems = recentItems.slice(0, RECENT_LIMIT);
    }
    persistRecentSearches();
    renderRecentSearches();
};

recentItems = loadRecentSearches();
renderRecentSearches();
if (recentList) {
    recentList.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-ticker]');
        if (!button) return;
        if (tickerField) {
            tickerField.value = button.dataset.ticker || '';
            tickerField.dispatchEvent(new Event('input'));
        }
        if (startField) {
            startField.value = button.dataset.start || '';
        }
        if (endField) {
            endField.value = button.dataset.end || '';
        }
    });
}

const recordRecentFromForm = () => {
    if (!tickerField || !startField || !endField) return;
    const ticker = tickerField.value.trim();
    const startValue = startField.value;
    const endValue = endField.value;
    if (!ticker || !startValue || !endValue) return;
    addRecentSearch({ ticker, start: startValue, end: endValue });
};

const tableToggleController = (() => {
    if (!tableWrapper || !tableExpandButton) {
        return { reset: () => {} };
    }
    let expanded = false;
    const sync = () => {
        if (!expanded) {
            tableWrapper.classList.remove('table-expanded');
            tableWrapper.style.maxHeight = '320px';
        } else {
            tableWrapper.classList.add('table-expanded');
            tableWrapper.style.maxHeight = '';
        }
        const disableToggle = tableWrapper.scrollHeight <= tableWrapper.clientHeight + 5;
        tableExpandButton.disabled = disableToggle;
        tableExpandButton.textContent = expanded
            ? tableExpandButton.dataset.collapse || 'Collapse'
            : tableExpandButton.dataset.expand || 'Show all';
    };
    tableExpandButton.addEventListener('click', () => {
        if (tableExpandButton.disabled) return;
        expanded = !expanded;
        sync();
    });
    sync();
    return {
        reset: () => {
            expanded = false;
            sync();
        },
    };
})();

const renderTable = (rows, predictionColumn, headingText) => {
    if (!tableBody) return;
    const noDataLabel = AppState.get().noDataLabel || 'вЂ”';
    tableBody.innerHTML = '';
    const hasRows = Array.isArray(rows) && rows.length > 0;
    if (tableWrapper) {
        tableWrapper.style.display = hasRows ? '' : 'none';
    }
    if (tableEmpty) {
        tableEmpty.hidden = hasRows;
    }
    if (predictionHeading && headingText) {
        predictionHeading.textContent = headingText;
    }
    if (!hasRows) {
        if (tableExpandButton) {
            tableExpandButton.disabled = true;
        }
        tableToggleController.reset();
        return;
    }
    const columnKey = predictionColumn || 'Pred_LSTM';
    rows.forEach((row) => {
        const date = row.Date || row.date || '-';
        const closeCell = typeof row.Close === 'number' ? `<span class="value">${row.Close.toFixed(2)}</span>` : `<span class="placeholder">${noDataLabel}</span>`;
        const predictionValue = row[columnKey];
        const predictionCell =
            typeof predictionValue === 'number'
                ? `<span class="value">${predictionValue.toFixed(2)}</span>`
                : `<span class="placeholder">${noDataLabel}</span>`;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${date}</td>
            <td>${closeCell}</td>
            <td>${predictionCell}</td>
        `;
        tableBody.appendChild(tr);
    });
    tableToggleController.reset();
};

const chartController = (() => {
    const canvas = document.getElementById('priceChart');
    const chartCard = document.getElementById('chartCard');
    const emptyState = document.getElementById('chartEmptyState');
    const captionEl = document.querySelector('.chart-header .caption');
    const captionTemplate = captionEl ? captionEl.dataset.template : null;
    const slider = document.getElementById('stepSlider');
    const sliderValue = document.getElementById('stepValue');
    const rangeButtons = document.querySelectorAll('[data-range]');
    const rangeContainer = document.getElementById('rangeButtons');
    const DEFAULT_RANGE = 60;
    let fullData = { labels: [], series: {} };
    let prepared = { labels: [], series: {} };
    let selectedRange = 0; // 0 means "all"
    let hasData = false;
    if (!canvas) {
        return { update: () => {} };
    }
    const ctx = canvas.getContext('2d');
    const getCssVar = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
    const getChartThemeColors = () => {
        const styles = getComputedStyle(document.body);
        const fallback = (value, alt) => (value && value.length ? value : alt);
        return {
            axisColor: fallback(styles.getPropertyValue('--chart-axis').trim(), '#475467'),
            gridColor: fallback(styles.getPropertyValue('--chart-grid').trim(), 'rgba(148,163,184,0.24)'),
            legendColor: fallback(styles.getPropertyValue('--chart-axis').trim(), '#475467'),
            tooltipBackground: fallback(styles.getPropertyValue('--card').trim(), 'rgba(255,255,255,0.92)'),
            tooltipText: fallback(styles.getPropertyValue('--text').trim(), '#101828'),
            closeLine: fallback(styles.getPropertyValue('--chart-close').trim(), '#2563eb'),
            lstmLine: fallback(styles.getPropertyValue('--chart-lstm').trim(), '#00c389'),
            xgbLine: fallback(styles.getPropertyValue('--chart-xgb').trim(), '#f97316'),
            rfLine: fallback(styles.getPropertyValue('--chart-rf').trim(), '#a855f7'),
        };
    };
    let themeColors = getChartThemeColors();
    const seriesConfig = [
        { key: 'Close', labelKey: 'closeLabel', colorKey: 'closeLine', defaultColor: '#4e5bff', fill: false },
        { key: 'Pred_LSTM', labelKey: 'predictionLabel', colorKey: 'lstmLine', defaultColor: '#00c389', fill: true },
        { key: 'Pred_XGB', label: 'XGB', colorKey: 'xgbLine', defaultColor: '#f97316', dashed: true },
        { key: 'Pred_RF', label: 'RF', colorKey: 'rfLine', defaultColor: '#a855f7', dashed: true },
    ];
    let stride = slider ? Math.max(parseInt(slider.value, 10) || 1, 1) : 1;

    const parseColor = (color) => {
        if (!color) return { r: 255, g: 255, b: 255 };
        const hexMatch = color.trim().match(/^#([0-9a-f]{3,8})$/i);
        if (hexMatch) {
            let hex = hexMatch[1];
            if (hex.length === 3) {
                hex = hex.split('').map((c) => c + c).join('');
            } else if (hex.length === 4) {
                hex = hex.split('').map((c, idx) => (idx < 3 ? c + c : '')).join('');
            }
            const intVal = parseInt(hex.substring(0, 6), 16);
            return {
                r: (intVal >> 16) & 255,
                g: (intVal >> 8) & 255,
                b: intVal & 255,
            };
        }
        const rgbMatch = color.trim().match(/^rgba?\(([^)]+)\)/i);
        if (rgbMatch) {
            const parts = rgbMatch[1].split(',').map((part) => parseFloat(part));
            return {
                r: parts[0] || 255,
                g: parts[1] || 255,
                b: parts[2] || 255,
            };
        }
        return { r: 255, g: 255, b: 255 };
    };

    const buildGradient = (color) => {
        const { r, g, b } = parseColor(color);
        const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height || 400);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.18)`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
        return gradient;
    };

    const getEmptyMessage = () => {
        const state = AppState.get() || {};
        const parts = [];
        if (state.noDataLabel) parts.push(state.noDataLabel);
        if (state.chartEmptyHint) parts.push(state.chartEmptyHint);
        return parts.join('. ');
    };

    const toggleControls = (enabled) => {
        hasData = enabled;
        const disabledClass = 'control-disabled';
        if (rangeButtons.length) {
            rangeButtons.forEach((button) => {
                button.disabled = !enabled;
                button.classList.toggle(disabledClass, !enabled);
            });
            if (rangeContainer) {
                rangeContainer.classList.toggle(disabledClass, !enabled);
            }
        }
        if (slider) {
            slider.disabled = !enabled;
            slider.setAttribute('aria-disabled', String(!enabled));
            slider.classList.toggle(disabledClass, !enabled);
        }
        if (!enabled && sliderValue) {
            sliderValue.textContent = '-';
        } else if (sliderValue) {
            sliderValue.textContent = stride;
        }
    };

    const toggleEmptyState = (isEmpty, message) => {
        if (chartCard) {
            chartCard.classList.toggle('is-empty', isEmpty);
        }
        if (emptyState) {
            emptyState.hidden = !isEmpty;
            if (message) {
                emptyState.textContent = message;
            }
        }
        toggleControls(!isEmpty);
    };

    const updateCaption = (count) => {
        if (!captionEl) return;
        if (!hasData) {
            captionEl.hidden = true;
            return;
        }
        captionEl.hidden = false;
        if (captionTemplate) {
            captionEl.innerHTML = captionTemplate.replace('{points}', `<strong>${count}</strong>`);
        } else {
            captionEl.innerHTML = `РџРѕСЃР»РµРґРЅРёРµ <strong>${count}</strong> С‚РѕС‡РµРє`;
        }
    };

    const prepareSeries = (rows) => {
        const labels = [];
        const series = {
            Close: [],
            Pred_LSTM: [],
            Pred_XGB: [],
            Pred_RF: [],
        };
        (rows || []).forEach((row) => {
            labels.push(row.Date || row.date || '');
            Object.keys(series).forEach((key) => {
                const value = row[key];
                series[key].push(typeof value === 'number' ? value : null);
            });
        });
        return { labels, series };
    };

    const chart = new Chart(canvas, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 18 },
                },
                tooltip: {
                    borderWidth: 1,
                    borderColor: 'rgba(148, 163, 184, 0.2)',
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.formattedValue}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false, color: 'rgba(148, 163, 184, 0.24)' },
                    ticks: { maxRotation: 0, color: '#475467' },
                },
                y: {
                    beginAtZero: false,
                    grid: { color: 'rgba(148, 163, 184, 0.24)' },
                    ticks: { color: '#475467' },
                },
            },
        },
    });

    const applyChartTheme = () => {
        themeColors = getChartThemeColors();
        chart.options.scales.x.ticks.color = themeColors.axisColor;
        chart.options.scales.y.ticks.color = themeColors.axisColor;
        chart.options.scales.x.grid.color = themeColors.gridColor;
        chart.options.scales.y.grid.color = themeColors.gridColor;
        chart.options.plugins.legend.labels.color = themeColors.legendColor;
        if (chart.options.plugins.tooltip) {
            chart.options.plugins.tooltip.backgroundColor = themeColors.tooltipBackground;
            chart.options.plugins.tooltip.titleColor = themeColors.tooltipText;
            chart.options.plugins.tooltip.bodyColor = themeColors.tooltipText;
            chart.options.plugins.tooltip.borderColor = themeColors.gridColor;
        }
    };

    const applyStride = () => {
        if (!prepared.labels.length) {
            chart.data.labels = [];
            chart.data.datasets = [];
            chart.update();
            toggleEmptyState(true, getEmptyMessage());
            updateCaption(0);
            return;
        }
        const indexes = [];
        for (let i = 0; i < prepared.labels.length; i += stride) {
            indexes.push(i);
        }
        const lastIndex = prepared.labels.length - 1;
        if (indexes[indexes.length - 1] !== lastIndex) {
            indexes.push(lastIndex);
        }
        const sampledLabels = indexes.map((i) => prepared.labels[i]);
        const datasets = seriesConfig
            .map((cfg) => {
                const dataSource = prepared.series[cfg.key];
                if (!dataSource || !dataSource.length) return null;
                const values = indexes.map((i) => dataSource[i]);
                const hasValue = values.some((val) => typeof val === 'number' && !Number.isNaN(val));
                if (!hasValue) return null;
                const labelText = cfg.label || AppState.get()[cfg.labelKey] || cfg.key;
                const color = themeColors[cfg.colorKey] || cfg.defaultColor;
                return {
                    label: labelText,
                    data: values,
                    borderColor: color,
                    backgroundColor: cfg.fill ? buildGradient(color) : color,
                    borderWidth: cfg.fill ? 2.5 : 2,
                    fill: cfg.fill ? 'start' : false,
                    pointRadius: cfg.fill ? 3 : 2,
                    pointHoverRadius: 4,
                    tension: 0.35,
                    borderDash: cfg.dashed ? [6, 4] : undefined,
                    spanGaps: true,
                };
            })
            .filter(Boolean);
        chart.data.labels = sampledLabels;
        chart.data.datasets = datasets;
        const bounds = AppState.get().chartBounds || null;
        chart.options.scales.y.min = bounds && typeof bounds.min === 'number' ? bounds.min : undefined;
        chart.options.scales.y.max = bounds && typeof bounds.max === 'number' ? bounds.max : undefined;
        chart.update();
        toggleEmptyState(!datasets.length, getEmptyMessage());
        updateCaption(sampledLabels.length);
    };

    const applyRange = () => {
        if (!fullData.labels.length) {
            prepared = { labels: [], series: {} };
            applyStride();
            return;
        }
        const total = fullData.labels.length;
        const limit = selectedRange > 0 ? Math.min(selectedRange, total) : total;
        const start = Math.max(total - limit, 0);
        prepared = {
            labels: fullData.labels.slice(start),
            series: Object.fromEntries(
                Object.entries(fullData.series).map(([key, values]) => [key, values.slice(start)]),
            ),
        };
        applyStride();
    };

    const parseRangeValue = (value) => {
        if (!value || value === 'all') return 0;
        const numeric = parseInt(value, 10);
        return Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
    };

    const updateRangeButtons = () => {
        if (!rangeButtons.length) return;
        rangeButtons.forEach((button) => {
            const buttonValue = parseRangeValue(button.dataset.range);
            button.classList.toggle('active', buttonValue === selectedRange);
        });
    };

    const setRange = (value) => {
        selectedRange = value;
        updateRangeButtons();
        applyRange();
    };

    const update = (rows) => {
        fullData = prepareSeries(Array.isArray(rows) ? rows : []);
        toggleEmptyState(!fullData.labels.length, getEmptyMessage());
        const total = fullData.labels.length;
        const defaultRange = total > DEFAULT_RANGE ? DEFAULT_RANGE : 0;
        selectedRange = defaultRange;
        updateRangeButtons();
        if (sliderValue) {
            sliderValue.textContent = stride;
        }
        applyRange();
    };

    if (slider) {
        slider.addEventListener('input', (event) => {
            stride = Math.max(parseInt(event.target.value, 10) || 1, 1);
            if (sliderValue) {
                sliderValue.textContent = stride;
            }
            applyStride();
        });
        if (sliderValue) {
            sliderValue.textContent = stride;
        }
    }

    if (rangeButtons.length) {
        rangeButtons.forEach((button) => {
            button.addEventListener('click', () => {
                if (button.disabled) return;
                const value = parseRangeValue(button.dataset.range);
                if (value === selectedRange) return;
                setRange(value);
            });
        });
        updateRangeButtons();
    }

    applyChartTheme();

    return { update };
})();

window.updateChart = (rows) => {
    chartController.update(rows || []);
};

const triggerPdfDownload = async () => {
    if (!formElement || !exportBtn) return;
    const tickerInput = formElement.querySelector('input[name="ticker"]');
    const startInput = formElement.querySelector('input[name="start"]');
    const endInput = formElement.querySelector('input[name="end"]');
    const ticker = tickerInput?.value.trim();
    const start = startInput?.value;
    const end = endInput?.value;
    if (!ticker || !start || !end) {
        renderAlerts({ error: 'РЈРєР°Р¶РёС‚Рµ С‚РёРєРµСЂ Рё РїРµСЂРёРѕРґ РїРµСЂРµРґ РІС‹РіСЂСѓР·РєРѕР№ РѕС‚С‡РµС‚Р°.' });
        return;
    }
    try {
        exportBtn.disabled = true;
        const payload = new FormData();
        payload.append('ticker', ticker);
        payload.append('start', start);
        payload.append('end', end);
        const response = await fetch('/export_pdf', { method: 'POST', body: payload });
        const contentType = response.headers.get('Content-Type') || '';
        if (!response.ok || !contentType.includes('application/pdf')) {
            let message = 'РќРµ СѓРґР°Р»РѕСЃСЊ СЃС„РѕСЂРјРёСЂРѕРІР°С‚СЊ РѕС‚С‡РµС‚.';
            try {
                if (contentType.includes('application/json')) {
                    const data = await response.json();
                    if (data && data.error) message = data.error;
                } else {
                    message = await response.text();
                }
            } catch (err) {
                console.error(err);
            }
            renderAlerts({ error: message });
            return;
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const disposition = response.headers.get('Content-Disposition') || '';
        const match = disposition.match(/filename="?([^";]+)"?/i);
        const filename = match ? match[1] : `report_${ticker}.pdf`;
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error(error);
        renderAlerts({ error: 'РќРµ СѓРґР°Р»РѕСЃСЊ СЃС„РѕСЂРјРёСЂРѕРІР°С‚СЊ РѕС‚С‡РµС‚.' });
    } finally {
        if (exportBtn) {
            exportBtn.disabled = !(Array.isArray(AppState.get().rows) && AppState.get().rows.length);
        }
    }
};

const applyState = (payload = {}) => {
    AppState.set(payload);
    renderAlerts(payload);
    renderMetrics(payload.summaryMetrics || []);
    renderTable(payload.rows || [], payload.predictionColumn, payload.predictionHeading);
    chartController.update(payload.rows || []);
    renderErrorMetrics(payload.modelMetrics || {});
    const hasRows = Array.isArray(payload.rows) && payload.rows.length;
    if (!payload.error && hasRows) {
        recordRecentFromForm();
    }
    if (exportBtn) {
        exportBtn.disabled = !hasRows;
    }
};

const parsePayloadFromHTML = (html) => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const payloadEl = doc.getElementById('app-state');
    if (!payloadEl) return null;
    try {
        return JSON.parse(payloadEl.textContent || '{}') || {};
    } catch (error) {
        console.warn('Unable to parse payload from response', error);
        return null;
    }
};

(function formHandler() {
    const form = document.getElementById('predict-form');
    if (!form) return;
    const submitBtn = form.querySelector('.submit-btn');
    const setLoading = (state) => {
        if (!submitBtn) return;
        submitBtn.disabled = state;
        submitBtn.classList.toggle('is-loading', state);
    };
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        try {
            setLoading(true);
            const response = await fetch(form.action || window.location.pathname, {
                method: 'POST',
                body: formData,
                headers: { 'X-Requested-With': 'XMLHttpRequest' },
            });
            if (!response.ok) {
                throw new Error('Network error');
            }
            const html = await response.text();
            const payload = parsePayloadFromHTML(html);
            if (payload) {
                applyState(payload);
            } else {
                throw new Error('Invalid payload');
            }
        } catch (error) {
            renderAlerts({ error: 'РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±РЅРѕРІРёС‚СЊ РґР°РЅРЅС‹Рµ' });
            console.error(error);
        } finally {
            setLoading(false);
        }
    });
})();

(function tickerHelpers() {
    const tickerInput = document.querySelector('input[name="ticker"]');
    const datalist = document.getElementById('ticker-suggestions');
    const startInput = document.querySelector('input[name="start"]');
    const endInput = document.querySelector('input[name="end"]');
    const periodButtons = document.querySelectorAll('#period-buttons button');
    const quickTickerButtons = document.querySelectorAll('[data-ticker]');
    if (!tickerInput || !datalist || !startInput || !endInput) {
        return;
    }

    let controller = null;
    const MIN_LENGTH = 2;

    const clearOptions = () => {
        datalist.innerHTML = '';
    };

    const renderOptions = (items) => {
        clearOptions();
        items.forEach((item) => {
            const option = document.createElement('option');
            const label = item.name ? `${item.symbol} - ${item.name}` : item.symbol;
            option.value = item.symbol;
            option.textContent = label;
            datalist.appendChild(option);
        });
    };

    const fetchSuggestions = async (value) => {
        if (value.length < MIN_LENGTH) {
            clearOptions();
            return;
        }
        try {
            if (controller) {
                controller.abort();
            }
            controller = new AbortController();
            const response = await fetch(`/search_tickers?q=${encodeURIComponent(value)}`, {
                signal: controller.signal,
            });
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            renderOptions(payload || []);
        } catch (error) {
            if (error.name === 'AbortError') {
                return;
            }
            clearOptions();
        }
    };

    const updateStartDate = (days) => {
        if (!days) return;
        const endDate = endInput.value ? new Date(endInput.value) : new Date();
        const startDate = new Date(endDate);
        startDate.setDate(endDate.getDate() - Number(days));
        const iso = startDate.toISOString().split('T')[0];
        startInput.value = iso;
        if (!endInput.value) {
            endInput.value = endDate.toISOString().split('T')[0];
        }
    };

    tickerInput.addEventListener('input', (event) => {
        const value = event.target.value.trim();
        fetchSuggestions(value);
    });

    periodButtons.forEach((button) => {
        button.addEventListener('click', () => {
            periodButtons.forEach((btn) => btn.classList.remove('active'));
            button.classList.add('active');
            updateStartDate(button.dataset.days);
        });
    });

    quickTickerButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const symbol = button.dataset.ticker;
            tickerInput.value = symbol;
            tickerInput.dispatchEvent(new Event('input'));
            tickerInput.focus();
        });
    });
})();

(function themeSwitcher() {
    const toggle = document.querySelector('[data-theme-toggle]');
    if (!toggle) return;
    const STORAGE_KEY = 'stockpredict-theme';

    const applyTheme = (theme) => {
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem(STORAGE_KEY, theme);
    };

    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
        applyTheme(saved);
    }

    toggle.addEventListener('click', () => {
        const current = document.body.getAttribute('data-theme') || 'light';
        const next = current === 'light' ? 'dark' : 'light';
        applyTheme(next);
    });
})();

if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        if (!exportBtn.disabled) {
            triggerPdfDownload();
        }
    });
}

applyState(initialState);

