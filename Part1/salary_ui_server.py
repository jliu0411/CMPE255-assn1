import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import joblib

from fair_value import estimate_fair_value

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "05_X_train_optimal.csv")
RAW_DATA_PATH = os.path.join(DATA_DIR, "01_raw_data.csv")
SCALER_PATH = os.path.join(DATA_DIR, "feature_scaler.json")

# Load model and training data for normalization
TRAIN_FEATURES = pd.read_csv(FEATURES_PATH)
MODEL_FEATURE_NAMES = TRAIN_FEATURES.columns.tolist()
with open(SCALER_PATH, "r", encoding="utf-8") as scaler_file:
    FEATURE_SCALER = json.load(scaler_file)
MODEL = joblib.load(MODEL_PATH)
REFERENCE_PLAYERS = pd.read_csv(RAW_DATA_PATH)

# Default values for user-facing stats
DEFAULTS = {
    "GP": 75,
    "MIN": 27.0,
    "PTS": 15.0,
    "FG_Percent": 0.450,
    "ThreePT_Percent": 0.350,
    "FT_Percent": 0.780,
    "REB": 4.5,
    "AST": 2.5,
    "STL": 0.9,
    "BLK": 1.2,
    "TO": 1.8,
    "PF": 2.0,
    "Plus_Minus": 0.0,
    "Age": 27,
    "Years_in_League": 5,
    "Injury_Prone": "No",
    "Stats_Year": 2025,
    "Contract_Start_Year": 2025,
    "MVP": "No",
    "All_NBA": "No",
    "All_Star": "No",
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NBA Fair Value Estimator</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #1d2433;
      margin: 0;
      padding: 32px;
      min-height: 100vh;
    }
    .container {
      max-width: 1000px;
      margin: 0 auto;
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.15);
      padding: 32px 36px 40px;
    }
    h1 {
      margin-top: 0;
      margin-bottom: 8px;
      font-size: 32px;
      color: #1d2433;
    }
    .subtitle {
      color: #51607d;
      margin-bottom: 28px;
      font-size: 15px;
    }
    .section-title {
      font-size: 16px;
      font-weight: 700;
      color: #1d2433;
      margin-top: 24px;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e5eaf0;
    }
    form {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }
    .full-width {
      grid-column: 1 / -1;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    label {
      font-weight: 600;
      font-size: 13px;
      color: #374151;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    input, select {
      padding: 10px 12px;
      border-radius: 8px;
      border: 1.5px solid #d1d9e0;
      font-size: 14px;
      background: #fbfcfe;
      transition: border-color 0.2s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: #667eea;
      background: #fff;
      box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .help-text {
      font-size: 12px;
      color: #9ca3af;
      margin-top: 4px;
    }
    button {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 14px 32px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 24px;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    button:active {
      transform: translateY(0);
    }
    .result {
      margin-top: 32px;
      padding: 24px;
      border-radius: 12px;
      background: linear-gradient(135deg, #d4fc79 0%, #96f794 100%);
      border: 2px solid #a8f56a;
      font-size: 18px;
      line-height: 1.6;
    }
    .result strong {
      color: #15803d;
      font-size: 24px;
    }
    .salary-amount {
      font-size: 28px;
      font-weight: 700;
      color: #166534;
      margin: 8px 0;
    }
    .note {
      margin-top: 20px;
      padding: 14px 16px;
      background: #fef3c7;
      border-left: 4px solid #f59e0b;
      border-radius: 6px;
      color: #92400e;
      font-size: 13px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🏀 NBA Fair Value Estimator</h1>
    <div class="subtitle">Estimate cap-aware market value separately from a player's existing contract.</div>

    <form method="POST" action="/predict">
      {fields}
      <div class="full-width">
        <button type="submit">💰 Predict Salary</button>
      </div>
    </form>

    {result_block}
    <div class="note"><strong>⚠️ Disclaimer:</strong> This model provides rough market-value estimates based on historical data. It is intended for benchmarking and decision support, not for exact contract prediction. Actual salaries are influenced by many non-statistical factors including brand value, market demand, agent negotiation, and team circumstances.</div>
  </div>
</body>
</html>
"""


def build_field_html(name: str, label: str, value, field_type: str = "number", help_text: str = "", step: str = "any"):
    if field_type == "select":
        options = "".join(
            f'<option value="{opt}" {"selected" if value == opt else ""}>{opt}</option>'
            for opt in ["No", "Yes"]
        )
        return f'''
        <div class="field">
          <label for="{name}">{label}</label>
          <select id="{name}" name="{name}">
            {options}
          </select>
          {f'<div class="help-text">{help_text}</div>' if help_text else ''}
        </div>
        '''
    return f'''
    <div class="field">
      <label for="{name}">{label}</label>
      <input id="{name}" name="{name}" type="{field_type}" step="{step}" value="{value}" required />
      {f'<div class="help-text">{help_text}</div>' if help_text else ''}
    </div>
    '''


def render_page(result_block: str = "", user_input: dict = None):
    if user_input is None:
        user_input = {}
    
    fields_html = ""
    
    # Basic Stats Section
    fields_html += '<div class="full-width section-title">📊 Basic Performance Stats</div>'
    fields_html += build_field_html("GP", "Games Played", user_input.get("GP", DEFAULTS["GP"]), "number", "Total games in season", "1")
    fields_html += build_field_html("MIN", "Minutes Per Game", user_input.get("MIN", DEFAULTS["MIN"]), "number", "Average minutes played")
    fields_html += build_field_html("PTS", "Points Per Game", user_input.get("PTS", DEFAULTS["PTS"]), "number", "Average points scored")
    fields_html += build_field_html("FG_Percent", "Field Goal %", user_input.get("FG_Percent", DEFAULTS["FG_Percent"]), "number", "Overall shooting %, 0.0-1.0")
    fields_html += build_field_html("ThreePT_Percent", "3-Point %", user_input.get("ThreePT_Percent", DEFAULTS["ThreePT_Percent"]), "number", "3-point shooting %, 0.0-1.0")
    fields_html += build_field_html("FT_Percent", "Free Throw %", user_input.get("FT_Percent", DEFAULTS["FT_Percent"]), "number", "Free throw %, 0.0-1.0")
    fields_html += build_field_html("REB", "Rebounds Per Game", user_input.get("REB", DEFAULTS["REB"]), "number", "Average rebounds per game")
    fields_html += build_field_html("AST", "Assists Per Game", user_input.get("AST", DEFAULTS["AST"]), "number", "Average assists per game")
    fields_html += build_field_html("STL", "Steals Per Game", user_input.get("STL", DEFAULTS["STL"]), "number", "Average steals per game")
    fields_html += build_field_html("BLK", "Blocks Per Game", user_input.get("BLK", DEFAULTS["BLK"]), "number", "Average blocks per game")
    fields_html += build_field_html("TO", "Turnovers Per Game", user_input.get("TO", DEFAULTS["TO"]), "number", "Average turnovers per game")
    fields_html += build_field_html("PF", "Personal Fouls", user_input.get("PF", DEFAULTS["PF"]), "number", "Average fouls per game")
    fields_html += build_field_html("Plus_Minus", "Plus/Minus", user_input.get("Plus_Minus", DEFAULTS["Plus_Minus"]), "number", "Team +/- while player is on court")
    
    # Player Profile Section
    fields_html += '<div class="full-width section-title">👤 Player Profile</div>'
    fields_html += build_field_html("Age", "Current Age", user_input.get("Age", DEFAULTS["Age"]), "number", "Player's current age in years", "1")
    fields_html += build_field_html("Years_in_League", "Years in NBA", user_input.get("Years_in_League", DEFAULTS["Years_in_League"]), "number", "Years of professional experience", "1")
    fields_html += build_field_html("Injury_Prone", "Injury Prone?", user_input.get("Injury_Prone", DEFAULTS["Injury_Prone"]), "select", "History of significant injuries")
    fields_html += '<div class="full-width section-title">Contract Market Context</div>'
    fields_html += build_field_html("Stats_Year", "Stats Season (ending year)", user_input.get("Stats_Year", DEFAULTS["Stats_Year"]), "number", "Year the entered performance statistics were recorded (2010-2025)", "1")
    fields_html += build_field_html("Contract_Start_Year", "Contract Start Year", user_input.get("Contract_Start_Year", DEFAULTS["Contract_Start_Year"]), "number", "Year the proposed contract begins; may be later than the stats season", "1")
    fields_html += build_field_html("MVP", "MVP Qualified?", user_input.get("MVP", DEFAULTS["MVP"]), "select", "MVP honors can unlock designated-veteran supermax eligibility")
    fields_html += build_field_html("All_NBA", "All-NBA Qualified?", user_input.get("All_NBA", DEFAULTS["All_NBA"]), "select", "All-NBA honors can affect maximum-contract eligibility")
    fields_html += build_field_html("All_Star", "All-Star?", user_input.get("All_Star", DEFAULTS["All_Star"]), "select", "Used as a transparent market-value adjustment")
    
    return HTML_TEMPLATE.replace("{fields}", fields_html).replace("{result_block}", result_block)


def calculate_engineered_features(stats: dict) -> dict:
    """
    Calculate engineered features from basic NBA stats.
    Maps user-friendly stats to the features the model was trained on.
    This function is the deployment counterpart of Chunk 3 feature engineering.
    """
    # Extract basic stats
    gp = float(stats.get("GP", 75))
    min_per_game = float(stats.get("MIN", 27))
    pts = float(stats.get("PTS", 15))
    fg_pct = float(stats.get("FG_Percent", 0.45))
    three_pct = float(stats.get("ThreePT_Percent", 0.35))
    ft_pct = float(stats.get("FT_Percent", 0.78))
    reb = float(stats.get("REB", 4.5))
    ast = float(stats.get("AST", 2.5))
    stl = float(stats.get("STL", 0.9))
    blk = float(stats.get("BLK", 1.2))
    to = float(stats.get("TO", 1.8))
    age = float(stats.get("Age", 27))
    years_in_league = float(stats.get("Years_in_League", 5))
    injury_prone = stats.get("Injury_Prone", "No") == "Yes"
    
    # Calculate engineered features to match the model's training data
    engineered = {
        "Year": int(float(stats.get("Stats_Year", stats.get("Valuation_Year", 2025)))),
        "Age": age,
        "Points_Efficiency": pts / (min_per_game + 1),
        "Years_In_League": years_in_league,
        "Field_Goal_Percent": fg_pct,
        "Free_Throw_Percent": ft_pct,
        "Steals_Per_Game": stl,
        "Assists_Per_Game": ast,
        "Blocks_Per_Game": blk,
        "Turnovers_Per_Game": to,
        "Turnover_Usage_Ratio": to / (min(40, (pts + to) / (min_per_game * 0.2)) + 1) if min_per_game > 0 else 0,
        "Age_Experience_Gap": max(0, age - years_in_league - 20),  # Approximate career start age
        "Shooting_Accuracy": (fg_pct + three_pct + ft_pct) / 3,
        "Prime_Age_Factor": max(0, (28 - abs(age - 28)) / 28),
        "Rebounds_Per_Game": reb,
        "Rebound_Assist_Ratio": reb / (ast + 1),
        "Points_Per_Game": pts,
        "Minutes_Per_Game": min_per_game,
        "Three_Point_Percent": three_pct,
        "Games_Played": gp * (0.8 if injury_prone else 1.0),  # Injury factor
        "Defensive_Impact": stl + blk,
        "Usage_Percent": min(40, (pts + to) / (min_per_game * 0.2)),  # Estimate usage
        "Experience_Level": (
            years_in_league if years_in_league <= 3 else 
            (1 if years_in_league <= 6 else 
             (2 if years_in_league <= 10 else 3))
        ),
    }
    
    return engineered


def predict_salary(payload: dict):
    # Calculate engineered features from user input
    engineered_stats = calculate_engineered_features(payload)
    
    # Apply the exact feature scaling learned during data preparation.
    standardized_input = {}
    for feature in MODEL_FEATURE_NAMES:
        if feature not in engineered_stats:
            raise ValueError(f"Missing feature: {feature}")
        raw_value = float(engineered_stats[feature])
        params = FEATURE_SCALER.get(feature)
        standardized_input[feature] = (
            (raw_value - params["mean"]) / params["scale"] if params else raw_value
        )
    
    row = pd.DataFrame([standardized_input], columns=MODEL_FEATURE_NAMES)
    contract_benchmark = max(0.5, float(MODEL.predict(row)[0]))
    fair_value = estimate_fair_value(payload, contract_benchmark, REFERENCE_PLAYERS)
    return contract_benchmark, fair_value["fair_value_millions"], engineered_stats, fair_value


class SalaryPredictHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html_page = render_page("")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_page.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_page.encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        form_data = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
        payload = {k: v[0] for k, v in form_data.items()}

        try:
            contract_benchmark, estimated_salary, engineered_stats, fair_value = predict_salary(payload)
            
            # The model target is stored directly in USD millions.
            result_text = (
                f"<div class=\"result\">"
                f"<div style=\"margin-bottom: 12px;\">Estimated fair annual value:</div>"
                f"<div class=\"salary-amount\">${estimated_salary:.2f}M</div>"
                f"<div style=\"font-size: 14px; color: #374151;\">"
                f"Historical contract benchmark: ${contract_benchmark:.2f}M<br>"
                f"Performance percentile: {fair_value['performance_percentile']:.1f}%<br>"
                f"Stats season: {fair_value['stats_year']} | Contract start: {fair_value['contract_start_year']}<br>"
                f"Service at contract start: {fair_value['contract_years_of_service']:.1f} years<br>"
                f"Salary cap ({fair_value['contract_start_year']}): ${fair_value['salary_cap_millions']:.2f}M "
                f"{'(projected)' if fair_value['cap_is_projected'] else ''}<br>"
                f"Eligible max tier: {fair_value['max_salary_rate']:.1f}% of cap "
                f"(${fair_value['eligible_first_year_max_millions']:.2f}M)<br>"
                f"Honors adjustment: {fair_value['honors_adjustment']}<br>"
                f"Illustrative 4-year value with 8% raises: ${fair_value['four_year_total_millions']:.2f}M "
                f"(${fair_value['four_year_aav_millions']:.2f}M AAV)"
                f"</div>"
                f"</div>"
            )
            html_page = render_page(result_text, payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_page.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_page.encode("utf-8"))
        except Exception as exc:
            error_html = f'<div class="result" style="background:#fff1f0;border-color:#f2b9b6;color:#b42620;">❌ Error: {html.escape(str(exc))}</div>'
            html_page = render_page(error_html, payload)
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_page.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_page.encode("utf-8"))

    def log_message(self, format, *args):
        return


def main():
    host = "127.0.0.1"
    port = 8006
    server = ThreadingHTTPServer((host, port), SalaryPredictHandler)
    print(f"NBA Fair Value Estimator running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
