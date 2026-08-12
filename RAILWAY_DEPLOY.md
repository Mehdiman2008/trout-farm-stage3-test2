# Railway deployment — Stage 3

This package is arranged so that `app.py` and `core/` are at the repository root.
Do not place all files inside another `trout_farm/` folder when uploading to GitHub.

## Fast deployment

1. Create a private GitHub repository.
2. Upload **all files/folders from this package root** to the repository root.
3. In Railway choose **New Project → Deploy from GitHub Repo**.
4. Railway reads `railway.json` and starts the app using `start_railway.sh`.
5. In Railway, open the service → **Settings → Networking → Generate Domain**.
6. Open `/api/solver` on the generated domain and confirm `available: true`.

No manual Start Command is required. The package binds to `0.0.0.0` and Railway's `$PORT` automatically.

## Persistent database (recommended for multi-day testing)

Without a Railway Volume, files written inside the deployment container can disappear after a redeploy.

1. Add a Railway Volume to this service.
2. Mount it at `/data`.
3. Add this service variable:

   `FARM_DB_PATH=/data/farm.db`

4. Redeploy.

The start script creates the database directory automatically.

## Useful checks

- Dashboard: `/`
- Solver status: `/api/solver`
- Validation: `/api/validate`
- SQLite backup: `/api/export/sqlite`
- JSON backup: `/api/export/json/download`

## Expected root layout

```text
app.py
core/
static/
data/
requirements.txt
railway.json
start_railway.sh
config.yaml
```
