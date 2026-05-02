# Traffic Data Web App

This project is a full-stack web application for visualizing and analyzing traffic data from a CSV file. It features a modern, dynamic user interface and robust backend processing.

## Structure
- `traffic_api/` — FastAPI backend (Python)
- `traffic_ui/` — React frontend (Material-UI)
- `pollution/traffic/traffic_dataset.csv` — Source data

## Features
- Interactive dashboard with charts and tables
- Data filtering, sorting, and summary statistics
- Responsive, modern UI
- API endpoints for data access and analysis

## Getting Started

### Backend (FastAPI)
1. Navigate to `traffic_api/`
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `uvicorn main:app --reload`

### Frontend (React)
1. Navigate to `traffic_ui/`
2. Install dependencies: `npm install`
3. Start the app: `npm start`

### Docker (optional)
- See `docker-compose.yml` for full-stack deployment

## Customization
- Update the CSV file at `pollution/traffic/traffic_dataset.csv` to use your own data
- Extend backend endpoints or frontend components as needed

## License
MIT
