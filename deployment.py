from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse


BASE_DIR = Path(__file__).resolve().parent
PREDICTION_DB_DIR = BASE_DIR / "Prediction database"


class ModelChoice(str, Enum):
    logistic_regression = "logistic_regression"
    xgboost = "xgboost"
    weibull_tte_rnn = "weibull_tte_rnn"


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


MODEL_FILES: Dict[ModelChoice, Path] = {
    ModelChoice.logistic_regression: PREDICTION_DB_DIR / "logreg_predictions_all.csv",
    ModelChoice.xgboost: PREDICTION_DB_DIR / "xgb_predictions_all.csv",
    ModelChoice.weibull_tte_rnn: PREDICTION_DB_DIR / "weibull_tte_predictions_all.csv",
}


app = FastAPI(
    title="First-Time Investor Predictions API",
    description=(
        "Select one of three models, choose Top X customers, and return results as JSON or CSV."
    ),
    version="1.0.0",
)


def _load_predictions(model_choice: ModelChoice) -> pd.DataFrame:
    file_path = MODEL_FILES[model_choice]

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prediction file not found for model '{model_choice.value}': {file_path.name}",
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read prediction file '{file_path.name}': {error}",
        ) from error

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction file '{file_path.name}' is empty.",
        )

    return df


def _top_rows(df: pd.DataFrame, model_choice: ModelChoice, top_x: int) -> pd.DataFrame:
    if "rank" in df.columns:
        ranked = df.sort_values("rank", ascending=True).head(top_x)
    elif model_choice in {ModelChoice.logistic_regression, ModelChoice.xgboost}:
        if "prob_first_time_investor" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="Expected column 'prob_first_time_investor' was not found.",
            )
        ranked = df.sort_values("prob_first_time_investor", ascending=False).head(top_x)
    else:
        if "risk_6m" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="Expected column 'risk_6m' was not found.",
            )
        ranked = df.sort_values("risk_6m", ascending=False).head(top_x)

    return ranked.reset_index(drop=True)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/predictions")
def get_predictions(
    model: ModelChoice = Query(..., description="Model to use"),
    top_x: int = Query(1000, ge=1, le=100000, description="How many top customers to return"),
    output: OutputFormat = Query(OutputFormat.json, description="Return format: json or csv"),
):
    df = _load_predictions(model)
    top_df = _top_rows(df, model, top_x)

    if output == OutputFormat.csv:
        csv_content = top_df.to_csv(index=False)
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; filename={model.value}_top_{top_x}.csv"
                )
            },
        )

    return JSONResponse(
        content={
            "model": model.value,
            "top_x": top_x,
            "count": int(len(top_df)),
            "results": top_df.to_dict(orient="records"),
        }
    )


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>First time investor customer predictions</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                margin: 2rem;
                background: #FFFFFF;
                color: #003F63;
            }
            .wrap { max-width: 560px; }
            h1 {
                margin-bottom: 0.25rem;
                color: #003F63;
            }
            p { color: #003F63; }
            form {
                margin-top: 1rem;
                padding: 1rem;
                border: 1px solid #003F63;
                border-radius: 8px;
                background: #003F63;
            }
            label {
                display: block;
                margin-top: 0.75rem;
                font-weight: 600;
                color: #FFFFFF;
            }
            select, input {
                width: 100%;
                margin-top: 0.35rem;
                padding: 0.5rem;
                box-sizing: border-box;
                border: 1px solid #999EDC;
                border-radius: 6px;
                color: #FFFFFF;
                background: #999EDC;
            }
            select option { color: #FFFFFF; background: #999EDC; }
            button {
                margin-top: 1rem;
                padding: 0.6rem 0.9rem;
                border: 1px solid #999EDC;
                border-radius: 6px;
                cursor: pointer;
                background: #999EDC;
                color: #FFFFFF;
                font-weight: 600;
            }
            .hint { margin-top: 1rem; font-size: 0.92rem; color: #003F63; }
            em { color: #003F63; font-style: italic; }
            code { color: #003F63; }
        </style>
    </head>
    <body>
        <div class="wrap">
            <h1><em>First time investor</em> customer predictions</h1>
            <p>Select a model, choose how many customers to return, and pick output format.</p>

            <form action="/predictions" method="get">
                <label for="model">Model</label>
                <select id="model" name="model">
                    <option value="logistic_regression">Logistic Regression</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="weibull_tte_rnn">Weibull TTE RNN</option>
                </select>

                <label for="top_x">Top X customers</label>
                <input id="top_x" name="top_x" type="number" min="1" max="100000" value="1000" required />

                <label for="output">Output format</label>
                <select id="output" name="output">
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                </select>

                <button type="submit">Get predictions</button>
            </form>

            <div class="hint">API endpoint: <code>/predictions</code></div>
        </div>
    </body>
</html>
"""


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")
