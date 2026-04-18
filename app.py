from flask import Flask, render_template, request
import pandas as pd
import os
import plotly.express as px
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shutil
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CHART_FOLDER = 'static/charts'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# ------------------ CLEAN DATA ------------------
def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    df.columns = df.columns.str.strip()
    return df

# ------------------ COLUMN DETECTION ------------------
def detect_columns(df):
    col_map = {}
    for col in df.columns:
        name = col.lower()

        if 'party' in name:
            col_map['party'] = col
        elif 'vote' in name:
            col_map['votes'] = col
        elif 'year' in name:
            col_map['year'] = col
        elif 'winner' in name:
            col_map['winner'] = col

    return col_map

# ------------------ CHARTS ------------------
def generate_charts(df, col_map):

    # 🔥 Clear old charts
    if os.path.exists("static/charts"):
        shutil.rmtree("static/charts")
    os.makedirs("static/charts")

    charts = []
    timestamp = str(int(time.time()))

    party = col_map.get('party')
    votes = col_map.get('votes')
    year = col_map.get('year')

    if not party or not votes:
        return charts

    # 🔹 Aggregated Data
    top = df.groupby(party)[votes].sum().reset_index()

    # 1️⃣ Party-wise Votes (Bar)
    fig1 = px.bar(top, x=party, y=votes, title="Total Votes by Party", template="plotly_dark")
    file1 = f"charts/bar_{timestamp}.html"
    fig1.write_html(f"static/{file1}")
    charts.append(file1)

    # 2️⃣ Vote Share (Pie)
    fig2 = px.pie(top, names=party, values=votes, title="Vote Share", template="plotly_dark")
    file2 = f"charts/pie_{timestamp}.html"
    fig2.write_html(f"static/{file2}")
    charts.append(file2)

    # 3️⃣ Year Trend (Line)
    if year:
        trend = df.groupby(year)[votes].sum().reset_index()
        fig3 = px.line(trend, x=year, y=votes, title="Year Trend", template="plotly_dark")
        file3 = f"charts/line_{timestamp}.html"
        fig3.write_html(f"static/{file3}")
        charts.append(file3)

    # 4️⃣ Raw Data (Top 10)
    fig4 = px.bar(df.head(10), x=party, y=votes, title="Top 10 Rows", template="plotly_dark")
    file4 = f"charts/raw_{timestamp}.html"
    fig4.write_html(f"static/{file4}")
    charts.append(file4)

    # 5️⃣ Top 10 Parties
    top10 = top.nlargest(10, votes)
    fig5 = px.bar(top10, x=party, y=votes, title="Top 10 Parties", template="plotly_dark")
    file5 = f"charts/top10_{timestamp}.html"
    fig5.write_html(f"static/{file5}")
    charts.append(file5)

    # 6️⃣ Votes Distribution (Histogram)
    fig6 = px.histogram(df, x=votes, title="Votes Distribution", template="plotly_dark")
    file6 = f"charts/hist_{timestamp}.html"
    fig6.write_html(f"static/{file6}")
    charts.append(file6)

    # 7️⃣ Votes vs Party Scatter
    fig7 = px.scatter(df.head(100), x=party, y=votes, title="Votes vs Party", template="plotly_dark")
    file7 = f"charts/scatter_{timestamp}.html"
    fig7.write_html(f"static/{file7}")
    charts.append(file7)

    # 8️⃣ Box Plot (Vote Spread)
    fig8 = px.box(df, x=party, y=votes, title="Vote Spread by Party", template="plotly_dark")
    file8 = f"charts/box_{timestamp}.html"
    fig8.write_html(f"static/{file8}")
    charts.append(file8)

    return charts

# ------------------ SENTIMENT (FIXED) ------------------
def sentiment_analysis(df, party_col):
    sample_text = {
        "BJP": "Strong leadership and development",
        "INC": "Facing criticism and internal issues",
        "AAP": "Focus on governance and reforms",
        "DMK": "Strong regional political support",
        "AIADMK": "Mixed public opinion",
        "TMC": "Growing influence in region"
    }

    result = {}

    for party in df[party_col].unique():
        text = sample_text.get(party, "Neutral performance")
        polarity = TextBlob(text).sentiment.polarity
        result[party] = round(polarity, 2)

    return result

# ------------------ ACCURACY (FIXED) ------------------
def calculate_accuracy(df, col_map):
    try:
        if 'winner' not in col_map or 'votes' not in col_map:
            return 0

        df['WinnerFlag'] = df[col_map['winner']].astype(str).str.lower().map({'true':1,'false':0})

        X = df[[col_map['votes']]]
        y = df['WinnerFlag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        return round(model.score(X_test, y_test) * 100, 2)

    except:
        return 0

# ------------------ ROUTE ------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    charts = []
    sentiment = {}
    accuracy = 0

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            df = clean_data(df)

            print("NEW DATASET LOADED")
            print(df.head())

            col_map = detect_columns(df)

            charts = generate_charts(df, col_map)

            if 'party' in col_map:
                sentiment = sentiment_analysis(df, col_map['party'])

            accuracy = calculate_accuracy(df, col_map)

    return render_template("index.html",
                           charts=charts,
                           sentiment=sentiment,
                           accuracy=accuracy)

# ------------------ RUN ------------------
if __name__ == "__main__":
    print("Server running...")
    app.run(debug=True)