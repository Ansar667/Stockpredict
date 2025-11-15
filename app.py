# app.py
import io
import os
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model import predict_stock
from ml_utils import search_tickers

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

LANG_CHOICES = [("en", "EN"), ("ru", "RU")]
TRANSLATIONS = {
    "en": {
        "title": "Stock Forecast Dashboard",
        "subtitle": "Select a ticker and period to see how our LSTM model projects the price trajectory.",
        "ticker_label": "Ticker",
        "start_label": "Start date",
        "end_label": "End date",
        "quick_title": "Quick ranges",
        "quick_buttons": [
            {"value": 30, "label": "1 month"},
            {"value": 90, "label": "3 months"},
            {"value": 180, "label": "6 months"},
            {"value": 365, "label": "1 year"},
        ],
        "popular_title": "Popular tickers",
        "helper_text": "The backend fetches pricing data from Yahoo Finance, prepares features and applies the LSTM model trained offline.",
        "run_btn": "Run forecast",
        "summary_prefix": "Summary",
        "summary_template": "Latest close: {close} | {model}: {pred}",
        "no_data": "N/A",
        "chart_empty_hint": "Select a ticker and press “Run forecast”.",
        "how_to_title": "How to use",
        "how_to_list": [
            "Pick a popular ticker or type your own.",
            "Use quick ranges or custom dates for historical windows.",
            "Compare the LSTM projection with the actual close price.",
        ],
        "trending_title": "Trending tickers",
        "close_label": "Close",
        "prediction_label": "LSTM",
        "table_title": "Latest records",
        "table_show_all": "Show all",
        "table_hide": "Collapse",
        "table_prediction_heading": "Pred {model}",
        "chart_title": "Price vs prediction",
        "chart_caption": "Last {points} points",
        "language_label": "Language",
        "theme_toggle": "Switch theme",
        "step_title": "Sampling step",
        "step_hint": "Adjust how many data points to skip (1 = every day).",
        "export_btn": "Download report (PDF)",
        "error_title": "Model errors",
        "forecast_generic_error": "We could not run the forecast. Please try again later.",
        "ticker_required_error": "Please enter a ticker.",
        "dates_required_error": "Please provide both start and end dates.",
        "dates_invalid_error": "Enter valid dates in the YYYY-MM-DD format.",
        "range_invalid_error": "The start date must be earlier than the end date.",
        "range_limit_error": "Please select a shorter date range (max {years} years).",
        "summary_error_message": "An error occurred while building the summary. Please try again.",
        "timeout_message": "The server is taking too long to respond. Please try again with a shorter date range.",
        "request_error_message": "An unexpected error occurred. Please try again.",
        "range_label": "Chart range",
        "recent_title": "Recent searches",
        "recent_empty": "No recent searches yet.",
        "nav_home": "Home",
        "nav_dashboard": "Dashboard",
        "landing_title_text": "Forecast smarter with AI",
        "landing_subtitle_text": "Interactive dashboard that blends LSTM forecasts with classic price analysis.",
        "landing_cta": "Open dashboard",
        "landing_how_title": "How it works",
        "landing_how_list": [
            "We load historical prices from Yahoo Finance.",
            "We build features and feed them into a trained LSTM model.",
            "We compare the LSTM projection with the actual close price on an interactive chart."
        ],
        "landing_features_title": "Key features",
        "landing_features_list": [
            "LSTM-based forecast for any supported ticker and date range.",
            "Interactive chart with range and sampling controls.",
            "Model error metrics (MAPE, RMSE, MAE, MSE).",
            "One-click PDF report export."
        ],
        "landing_preview_title": "Dashboard preview",
        "landing_preview_button": "Run",
        "landing_audience_title": "Who is this for",
        "landing_audience_cards": [
            {
                "title": "Students and interns",
                "bullets": [
                    "Practice with time series and ML.",
                    "A ready pet project for your portfolio.",
                ],
            },
            {
                "title": "Developers",
                "bullets": [
                    "Full-stack example with Python + Flask + JS.",
                    "Clean architecture and polished frontend.",
                ],
            },
            {
                "title": "Investor enthusiasts",
                "bullets": [
                    "Convenient view of historical prices and dynamics.",
                    "Honest model without unrealistic promises.",
                ],
            },
        ],
        "landing_stack_title": "Under the hood",
        "landing_stack_body": "The dashboard is built with Python and Flask, uses an LSTM model for forecasting and Yahoo Finance data. The UI is powered by JavaScript and Chart.js.",
        "landing_stack_badges": [
            "Python",
            "Flask",
            "LSTM",
            "Yahoo Finance",
            "Chart.js",
            "Docker",
        ],
        "landing_honest_title": "Honest forecast, not a magic crystal ball",
        "landing_honest_text": "Stock markets are hard to predict, so the model does not promise miracles or guaranteed profits. Instead, it transparently shows how the LSTM sees price dynamics and what errors it produces on historical data.",
    },
    "ru": {
        "title": "Панель прогноза акций",
        "subtitle": "Выберите тикер и период, чтобы увидеть, как наша модель LSTM прогнозирует траекторию цены.",
        "ticker_label": "Тикер",
        "start_label": "Дата начала",
        "end_label": "Дата окончания",
        "quick_title": "Быстрые периоды",
        "quick_buttons": [
            {"value": 30, "label": "1 месяц"},
            {"value": 90, "label": "3 месяца"},
            {"value": 180, "label": "6 месяцев"},
            {"value": 365, "label": "1 год"},
        ],
        "popular_title": "Популярные тикеры",
        "helper_text": "Сервис загружает котировки из Yahoo Finance, готовит признаки и применяет обученную LSTM-модель.",
        "run_btn": "Запустить прогноз",
        "summary_prefix": "Сводка",
        "summary_template": "Последняя цена: {close} | {model}: {pred}",
        "no_data": "Нет данных",
        "chart_empty_hint": "Выберите тикер и нажмите «Запустить прогноз».",
        "how_to_title": "Как пользоваться",
        "how_to_list": [
            "Выберите популярный тикер или введите свой.",
            "Используйте быстрые периоды или задайте даты вручную.",
            "Сравните прогноз LSTM с фактической ценой закрытия.",
        ],
        "trending_title": "Трендовые тикеры",
        "close_label": "Закрытие",
        "prediction_label": "LSTM",
        "table_title": "Последние записи",
        "table_show_all": "Показать все",
        "table_hide": "Свернуть",
        "table_prediction_heading": "Прогноз {model}",
        "chart_title": "Цена и прогноз",
        "chart_caption": "Последние {points} точек",
        "language_label": "Язык",
        "theme_toggle": "Сменить тему",
        "step_title": "Шаг выборки",
        "step_hint": "Настройте, сколько точек пропускать (1 = каждый день).",
        "export_btn": "Скачать отчёт (PDF)",
        "error_title": "Ошибки модели",
        "forecast_generic_error": "Не удалось запустить прогноз. Попробуйте позже.",
        "ticker_required_error": "Введите тикер.",
        "dates_required_error": "Укажите дату начала и окончания.",
        "dates_invalid_error": "Введите корректные даты в формате ГГГГ-ММ-ДД.",
        "range_invalid_error": "Дата начала должна быть раньше даты окончания.",
        "range_limit_error": "Сократите период — максимум {years} лет.",
        "summary_error_message": "Произошла ошибка при формировании сводки. Попробуйте ещё раз.",
        "timeout_message": "Сервер слишком долго не отвечает. Попробуйте ещё раз с более коротким периодом.",
        "request_error_message": "Произошла ошибка. Попробуйте ещё раз.",
        "range_label": "Диапазон графика",
        "recent_title": "Недавние запросы",
        "recent_empty": "Недавние запросы отсутствуют.",
        "nav_home": "Главная",
        "nav_dashboard": "Панель",
        "landing_title_text": "Панель прогноза акций",
        "landing_subtitle_text": "Выберите тикер и период, чтобы увидеть, как наша модель LSTM прогнозирует траекторию цены.",
        "landing_cta": "Открыть панель",
        "landing_how_title": "Как это работает",
        "landing_how_list": [
            "Мы загружаем котировки из Yahoo Finance.",
            "Мы строим признаки и пропускаем их через обученную LSTM-модель.",
            "Мы сравниваем прогноз и фактическую цену на интерактивном графике."
        ],
        "landing_features_title": "Возможности панели",
        "landing_features_list": [
            "Прогноз на основе LSTM по выбранному тикеру и периоду.",
            "Интерактивный график с выбором диапазона и шага выборки.",
            "Ошибки модели (MAPE, RMSE, MAE, MSE) в удобном виде.",
            "Экспорт отчёта в PDF одним кликом."
        ],
        "landing_preview_title": "Превью дашборда",
        "landing_preview_button": "Запустить",
        "landing_audience_title": "Для кого эта панель",
        "landing_audience_cards": [
            {
                "title": "Студенты и интерны",
                "bullets": [
                    "Практика работы с временными рядами и ML.",
                    "Готовый pet-project для портфолио.",
                ],
            },
            {
                "title": "Разработчики",
                "bullets": [
                    "Full-stack пример на Python + Flask + JS.",
                    "Чистая архитектура и аккуратный фронтенд.",
                ],
            },
            {
                "title": "Инвесторы-энтузиасты",
                "bullets": [
                    "Удобный просмотр истории цены и динамики.",
                    "Честная модель без обещаний сверхприбыли.",
                ],
            },
        ],
        "landing_stack_title": "Что под капотом",
        "landing_stack_body": "Панель построена на Python и Flask, использует LSTM-модель для прогноза и данные Yahoo Finance. Интерфейс работает на JavaScript и Chart.js.",
        "landing_stack_badges": [
            "Python",
            "Flask",
            "LSTM",
            "Yahoo Finance",
            "Chart.js",
            "Docker",
        ],
        "landing_honest_title": "Честный прогноз, а не магический шар",
        "landing_honest_text": "Рынок акций слабо предсказуем, поэтому модель не обещает чудес и ракет на графике. Вместо этого она аккуратно показывает, как LSTM видит динамику цены и какие ошибки даёт на истории.",
    },
}

PREDICTION_COLUMN = "Pred_LSTM"
SEARCH_CACHE_TTL = 60 * 5
SEARCH_CACHE: dict[str, dict[str, object]] = {}

POPULAR_TICKERS = [
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "NVDA", "name": "Nvidia"},
    {"symbol": "META", "name": "Meta"},
    {"symbol": "AMZN", "name": "Amazon"},
]

MAX_RANGE_YEARS = 10
MAX_RANGE_DAYS = MAX_RANGE_YEARS * 365


def _get_ui(lang: str) -> dict[str, object]:
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"])


def _validate_request_params(ticker: str, start: str, end: str, ui: dict) -> tuple[str, datetime, datetime]:
    ticker_clean = (ticker or "").strip().upper()
    if not ticker_clean:
        raise ValueError(ui["ticker_required_error"])
    if not start or not end:
        raise ValueError(ui["dates_required_error"])
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError(ui["dates_invalid_error"])
    if start_dt >= end_dt:
        raise ValueError(ui["range_invalid_error"])
    if (end_dt - start_dt).days > MAX_RANGE_DAYS:
        raise ValueError(ui["range_limit_error"].format(years=MAX_RANGE_YEARS))
    return ticker_clean, start_dt, end_dt


def _build_state_payload(
    ui: dict,
    lang: str,
    *,
    rows=None,
    summary_metrics=None,
    summary_text=None,
    model_metrics=None,
    chart_bounds=None,
    prediction_label=None,
    prediction_column=None,
    summary_raw=None,
    error=None,
):
    prediction_column = prediction_column or PREDICTION_COLUMN
    display_label = prediction_label or prediction_column.replace("Pred_", "") or ui["prediction_label"]
    return {
        "rows": rows or [],
        "summaryMetrics": summary_metrics or [],
        "summaryText": summary_text,
        "summary": summary_raw,
        "error": error,
        "chartBounds": chart_bounds,
        "predictionLabel": display_label,
        "predictionColumn": prediction_column,
        "predictionHeading": ui["table_prediction_heading"].format(model=display_label),
        "closeLabel": ui["close_label"],
        "noDataLabel": ui["no_data"],
        "chartEmptyHint": ui["chart_empty_hint"],
        "summaryPrefix": ui["summary_prefix"],
        "modelMetrics": model_metrics or {},
        "summaryErrorMessage": ui["summary_error_message"],
        "timeoutMessage": ui["timeout_message"],
        "requestErrorMessage": ui["request_error_message"],
        "lang": lang,
    }


def _generate_forecast_bundle(ticker: str, start: str, end: str, lang: str) -> dict:
    ui = _get_ui(lang)
    ticker_clean, *_ = _validate_request_params(ticker, start, end, ui)
    df, summary, model_metrics = predict_stock(ticker_clean, start, end)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = df["Date"].astype(str)

    expected_columns = ["Date", "Close", "Pred_LSTM", "Pred_XGB", "Pred_RF"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data is missing required columns: {', '.join(missing_columns)}.")

    prediction_column = PREDICTION_COLUMN
    prediction_label = prediction_column.replace("Pred_", "")

    table = df.to_dict(orient="records")

    chart_bounds = None
    close_values = [row.get("Close") for row in table if isinstance(row.get("Close"), (int, float))]
    if close_values:
        close_min = float(min(close_values))
        close_max = float(max(close_values))
        close_range = max(close_max - close_min, 1.0)
        padding = max(close_range * 0.25, close_max * 0.01)
        chart_bounds = {"min": close_min - padding, "max": close_max + padding}

    summary_metrics = []
    if table:
        last_row = table[-1]
        summary_metrics = [
            {"label": ui["close_label"], "value": last_row.get("Close")},
            {
                "label": prediction_label or ui["prediction_label"],
                "value": last_row.get(prediction_column),
            },
        ]

    summary_text = None
    if summary_metrics:
        close_val = summary_metrics[0].get("value")
        pred_val = summary_metrics[1].get("value") if len(summary_metrics) > 1 else None
        if close_val is not None and pred_val is not None:
            summary_text = ui["summary_template"].format(
                close=f"{close_val:.2f}",
                model=prediction_label or ui["prediction_label"],
                pred=f"{pred_val:.2f}",
            )

    state_payload = _build_state_payload(
        ui,
        lang,
        rows=table,
        summary_metrics=summary_metrics,
        summary_text=summary_text,
        model_metrics=model_metrics,
        chart_bounds=chart_bounds,
        prediction_label=prediction_label,
        prediction_column=prediction_column,
        summary_raw=summary,
        error=None,
    )

    return {
        "table": table,
        "summary_metrics": summary_metrics,
        "summary_text": summary_text,
        "summary": summary,
        "model_metrics": model_metrics,
        "chart_bounds": chart_bounds,
        "prediction_label": prediction_label,
        "prediction_column": prediction_column,
        "state": state_payload,
    }


def resolve_language():
    lang = session.get("lang", "en")
    lang_query = request.args.get("lang")
    lang_form = request.form.get("lang")
    json_payload = request.get_json(silent=True)
    lang_json = json_payload.get("lang") if isinstance(json_payload, dict) else None
    if lang_query in TRANSLATIONS:
        session["lang"] = lang_query
        return lang_query
    if lang_form in TRANSLATIONS:
        session["lang"] = lang_form
        return lang_form
    if lang_json in TRANSLATIONS:
        session["lang"] = lang_json
        return lang_json
    return lang


@app.get("/")
def index():
    lang = resolve_language()
    ui = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return render_template(
        "landing.html",
        ui=ui,
        lang=lang,
        languages=LANG_CHOICES,
        page_title=ui["title"],
        active_route="index",
    )


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    lang = resolve_language()
    ui = _get_ui(lang)
    error = None
    result: dict[str, object] = {}

    if request.method == "POST":
        ticker = request.form.get("ticker")
        start = request.form.get("start")
        end = request.form.get("end")

        try:
            result = _generate_forecast_bundle(ticker, start, end, lang)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            app.logger.exception("Prediction failed for %s", ticker, exc_info=exc)
            error = ui["forecast_generic_error"]

    table = result.get("table")
    summary_metrics = result.get("summary_metrics")
    summary_text = result.get("summary_text")
    summary = result.get("summary")
    model_metrics = result.get("model_metrics")
    chart_bounds = result.get("chart_bounds")
    prediction_label = result.get("prediction_label") or ui["prediction_label"]
    prediction_column = result.get("prediction_column") or PREDICTION_COLUMN

    state_payload = dict(result.get("state") or {})
    if not state_payload:
        state_payload = _build_state_payload(
            ui,
            lang,
            rows=table,
            summary_metrics=summary_metrics,
            summary_text=summary_text,
            model_metrics=model_metrics,
            chart_bounds=chart_bounds,
            prediction_label=prediction_label,
            prediction_column=prediction_column,
            summary_raw=summary,
            error=error,
        )
    state_payload["error"] = error

    return render_template(
        "dashboard.html",
        table=table,
        summary=summary,
        error=error,
        summary_metrics=summary_metrics,
        model_metrics=model_metrics,
        popular_tickers=POPULAR_TICKERS,
        chart_bounds=chart_bounds,
        prediction_label=prediction_label,
        prediction_column=prediction_column,
        ui=ui,
        lang=lang,
        languages=LANG_CHOICES,
        page_title=ui["title"],
        active_route="dashboard",
        summary_text=summary_text,
        state_payload=state_payload,
    )


def _create_chart_image(df, prediction_label):
    chart_stream = io.BytesIO()
    fig, ax = plt.subplots(figsize=(7, 3))
    dates = pd.to_datetime(df["Date"])
    ax.plot(dates, df["Close"], label="Close", color="#2563eb", linewidth=2)
    series_config = [
        ("Pred_LSTM", prediction_label or "LSTM", "#16a34a"),
        ("Pred_XGB", "Pred_XGB", "#f97316"),
        ("Pred_RF", "Pred_RF", "#a855f7"),
    ]
    for column, label, color in series_config:
        if column in df and df[column].notna().any():
            ax.plot(dates, df[column], label=label, color=color, linewidth=2, linestyle="--")
    ax.set_title("Price vs forecast")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_stream, format="png", dpi=150)
    plt.close(fig)
    chart_stream.seek(0)
    return chart_stream


def _build_pdf(ticker, start, end, summary_text, df, prediction_label, metrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Forecast report for {ticker}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Period: {start} — {end}", styles["Normal"]),
        Spacer(1, 6),
        Paragraph(summary_text or "No summary available.", styles["Normal"]),
        Spacer(1, 16),
    ]

    if metrics:
        def format_metric(value, suffix=""):
            if value is None:
                return "—"
            return f"{value:.2f}{suffix}"

        table_data = [["Model", "MAPE %", "RMSE", "MAE", "MSE"]]
        for key, stats in metrics.items():
            if not stats:
                continue
            name = key.replace("Pred_", "") or key
            table_data.append(
                [
                    name,
                    format_metric(stats.get("mape"), "%"),
                    format_metric(stats.get("rmse")),
                    format_metric(stats.get("mae")),
                    format_metric(stats.get("mse")),
                ]
            )
        if len(table_data) > 1:
            story.append(Paragraph("Model errors", styles["Heading3"]))
            metrics_table = Table(table_data, repeatRows=1)
            metrics_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            story.append(metrics_table)
            story.append(Spacer(1, 12))

    chart_bytes = _create_chart_image(df, prediction_label)
    story.append(RLImage(chart_bytes, width=460, height=220))
    story.append(Spacer(1, 16))

    table_df = df.tail(10)
    columns = ["Date", "Close"]
    column_labels = ["Date", "Close"]
    for column, label in [
        ("Pred_LSTM", prediction_label or "LSTM"),
        ("Pred_XGB", "Pred_XGB"),
        ("Pred_RF", "Pred_RF"),
    ]:
        if column in df and df[column].notna().any():
            columns.append(column)
            column_labels.append(label)

    table_data = [column_labels]
    for _, row in table_df.iterrows():
        formatted = []
        for column in columns:
            value = row.get(column)
            if column == "Date":
                formatted.append(str(value))
            elif value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                formatted.append("—")
            else:
                formatted.append(f"{value:.2f}")
        table_data.append(formatted)

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(Paragraph("Last records", styles["Heading3"]))
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.post("/run_forecast")
def run_forecast_api():
    lang = resolve_language()
    ui = _get_ui(lang)
    payload = request.get_json(silent=True)
    payload = payload if isinstance(payload, dict) else {}
    ticker = payload.get("ticker") or request.form.get("ticker")
    start = payload.get("start") or request.form.get("start")
    end = payload.get("end") or request.form.get("end")
    try:
        result = _generate_forecast_bundle(ticker, start, end, lang)
        response_payload = dict(result.get("state") or {})
        response_payload["error"] = None
        return jsonify({"ok": True, "data": response_payload})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Forecast API failed for %s", ticker, exc_info=exc)
        return jsonify({"ok": False, "error": ui["forecast_generic_error"]}), 500


@app.post("/export_pdf")
def export_pdf():
    lang = resolve_language()
    ui = _get_ui(lang)
    ticker = request.form.get("ticker")
    start = request.form.get("start")
    end = request.form.get("end")
    try:
        ticker_clean, _, _ = _validate_request_params(ticker, start, end, ui)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        df, summary, metrics = predict_stock(ticker_clean, start, end)
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Date"] = df["Date"].astype(str)

        prediction_label = PREDICTION_COLUMN.replace("Pred_", "")
        pdf_stream = _build_pdf(ticker_clean, start, end, summary, df, prediction_label, metrics)
        filename = f"report_{ticker_clean}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        return send_file(pdf_stream, as_attachment=True, download_name=filename, mimetype="application/pdf")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Failed to export PDF for %s", ticker_clean, exc_info=exc)
        return jsonify({"error": ui["forecast_generic_error"]}), 500


@app.get("/search_tickers")
def search_tickers_api():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])

    try:
        key = query.lower()
        cached = SEARCH_CACHE.get(key)
        now = time.time()
        if cached and now - cached["timestamp"] < SEARCH_CACHE_TTL:
            results = cached["data"]
        else:
            results = search_tickers(query, limit=8)
            SEARCH_CACHE[key] = {"timestamp": now, "data": results}
    except Exception:
        results = []
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)


