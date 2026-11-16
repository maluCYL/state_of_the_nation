# State of the Nation (STON)

## What is this repository?

This is the repository for the **State of the Nation (STON)** project. It contains the scripts and notebooks used to analyse the project data.

The live data sits in a relational database on **Azure**. Before running any scripts, set up a local `.env` file with the credentials needed to connect to that database. You can find the relevant keys in Azure.

**Do not rename files** used by this repository. Notebooks and functions call files by their current names, and renaming them will cause errors.

## Repository setup

1. Download or clone this repository.
2. Open the repository in your IDE (this project has been developed in Cursor).
3. Create a `.env` file in the repository root and add the database credentials (see below).
4. In the terminal, from the repository root, run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. You are ready to use the notebooks in `scripts/`.

### `.env` variables

Create a `.env` file in the project root with at least:

```env
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=
```

Optional settings may also be used for SSL / table overrides, depending on the Azure connection (for example `DB_SSL`, `DB_SSL_CA`, `DB_TABLE`, `DB_OBSERVATIONS_TABLE`).

Never commit the `.env` file. It is already listed in `.gitignore`.

## Relevant notebooks and scripts

Open the `scripts/` folder for the main working files:

1. **`main.ipynb`** — primary indicator charts produced with the shared plotting functions.
2. **`executive_summary.ipynb`** — indicators used for the 2026 policy brief (30 pages).
3. **`single_graphs.ipynb`** — charts that do not follow the standard pattern used in `main.ipynb` or `executive_summary.ipynb`. These were created as one-off exceptions and may not appear in the final editorial outputs. Supporting data for some of these charts can be found in the `data/` folder (in the STON drive folder).
4. **`substack_blog.ipynb`** — graphs and animations used for the promotional Substack blog for the STON campaign.
5. **`functions.py`** — the shared Python module used by all notebooks. Every notebook imports from this file. Treat it carefully and do not remove it from the repository; it is the main script holding the project together.

## Repository logic

All reusable logic for the notebooks lives in **`functions.py`**.

- Do **not** define heavy or reusable functions inside analysis notebooks.
- Put building / reusable code in `functions.py`.
- Keep notebooks focused on calling those functions and producing outputs.

Generated charts and GIFs are saved to the `output/` folder by default.

## What is the `SQL_scripts/` folder?

This folder contains SQL scripts for database management. They were set aside as the project grew in size and complexity, but they are kept here in case they are useful later.

## Project folders

| Folder / file | Purpose |
|---|---|
| `scripts/` | Notebooks and `functions.py` |
| `data/` | Local supporting files (CSV / Excel / cached GeoJSON). Ignored by Git. |
| `output/` | Generated charts, GIFs, and other figure exports. Ignored by Git. |
| `SQL_scripts/` | Archived SQL helpers for database work. Ignored by Git. |
| `requirements.txt` | Python dependencies |
| `.env` | Local database credentials (not committed) |

## Notes

- Activate the virtual environment (`source .venv/bin/activate`) before running notebooks or installing packages.
- If the Azure firewall blocks your connection, you may need to allow your current IP address in Azure.
- The England regions heatmap may download and cache `data/england_regions.geojson` on first use.
