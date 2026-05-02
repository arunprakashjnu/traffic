from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), 'traffic_dataset.csv')
df = pd.read_csv(DATA_PATH)


@app.route('/', methods=['GET'])
def index():
    # Get filter values from query params
    congestion = request.args.get('congestion', '')
    min_count = request.args.get('min_count', '')
    max_count = request.args.get('max_count', '')
    min_speed = request.args.get('min_speed', '')
    max_speed = request.args.get('max_speed', '')
    vehicle_type = request.args.get('vehicle_type', '')

    filtered_df = df.copy()
    if congestion:
        filtered_df = filtered_df[filtered_df['Traffic Congestion Level'].str.lower() == congestion.lower()]
    if min_count:
        try:
            filtered_df = filtered_df[filtered_df['Vehicle Count'] >= int(min_count)]
        except ValueError:
            pass
    if max_count:
        try:
            filtered_df = filtered_df[filtered_df['Vehicle Count'] <= int(max_count)]
        except ValueError:
            pass
    if min_speed:
        try:
            filtered_df = filtered_df[filtered_df['Avg Speed (km/h)'] >= float(min_speed)]
        except ValueError:
            pass
    if max_speed:
        try:
            filtered_df = filtered_df[filtered_df['Avg Speed (km/h)'] <= float(max_speed)]
        except ValueError:
            pass
    if vehicle_type:
        filtered_df = filtered_df[filtered_df['Vehicle Types Detected'].str.contains(vehicle_type, case=False, na=False)]

    congestion_levels = sorted(df['Traffic Congestion Level'].unique())
    table_html = filtered_df.to_html(classes='data', index=False)
    return render_template(
        'index.html',
        table=table_html,
        congestion_levels=congestion_levels,
        selected=congestion,
        min_count=min_count,
        max_count=max_count,
        min_speed=min_speed,
        max_speed=max_speed,
        vehicle_type=vehicle_type
    )

if __name__ == '__main__':
    app.run(debug=True)
