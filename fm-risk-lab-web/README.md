# fm-risk-lab-web

Next.js dashboard for the Mortgage Credit Risk and Stress Testing Lab. It reads from the
FastAPI backend in the parent repository (`src/api/main.py`) and shows the portfolio ECL
summary, a stress-test lab with scenario sliders, and a single-loan risk scorer.

## Pages

| Route | What it shows |
|---|---|
| `/` | Portfolio overview: total EAD and ECL, mean PD and LGD, vintage default curves, ECL by state, scenario comparison |
| `/risk-lab` | Scenario sliders that re-run the ECL engine through the API |
| `/loan-explorer` | Enter one loan's attributes and get its PD, LGD, ECL and risk level |
| `/methodology` | How the models were built and what data they use |

## Requirements

- Node.js 22 or newer (the CI workflow uses Node 22)
- The backend API running, by default on `http://localhost:8000`

## Run it

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

To point the dashboard at a backend on a different address, set `NEXT_PUBLIC_API_BASE_URL`
(for example in `.env.local`). It defaults to `http://localhost:8000`.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the development server |
| `npm run build` | Create a production build |
| `npm start` | Serve the production build |
| `npm test` | Run the unit tests in `lib/*.test.ts` with Node's built-in test runner |

## Layout

- `app/` : the App Router pages and global styles
- `components/` : shared UI pieces (KPI cards, risk gauge, scenario slider, charts)
- `lib/api.ts` : the single place that calls the backend; every request goes through it
- `lib/utils.ts` : formatting and risk-colour helpers, covered by `lib/utils.test.ts`
