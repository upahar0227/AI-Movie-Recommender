# FlimFlex

AI-based movie recommendation system. University research project for COMP-5112-FB (Research Methodology), Lakehead University, by Upahar Bhandari and Anup Burlakoti.

The project compares content-based, collaborative, hybrid, and sentiment-aware movie rankers, and delivers them in a small web application with accounts, ratings, a watchlist, and personalised recommendations. Metrics below are from `python -m src.recsys.train` on this machine. They are not hand-filled.

The reported run used Python 3.14.2 with the packages pinned in `requirements.txt`. If a package fails to install, try the same Python version. The random seed is 42.

## Features

- Models: content-based (TF-IDF), item-kNN, SVD, a hybrid with adaptive weighting, and a review-sentiment classifier.
- Evaluation: 5-fold per-user split, ranking and rating metrics, significance tests, and a cold-start experiment.
- Web app: sign-up and login, genre onboarding, half-star ratings, a watchlist, My ratings, a community rating on every movie card, similar movies, and movie search.
- Personalised Home: a blend of the long-term profile, recent activity, and recent searches, with stability rules and diversity re-ranking.
- Genres can be changed from Home at any time.
- Privacy controls: clear search history, delete one search, or delete the whole account.

## Project structure

```
api/            FastAPI app (entry point api/main.py) and route modules
src/recsys/     Models, training, evaluation, blending, posters, search profile
src/app/        Database setup, ORM models, security, settings
frontend/       Web page (plain HTML, CSS, and JavaScript, no build step)
alembic/        Database migrations
tests/          pytest tests
results/        Metrics tables and plots written by training
data/raw/       Datasets you download (not committed)
```

## Quick start

These steps use Windows PowerShell. On Mac or Linux, use `python3` to create the environment and `source .venv/bin/activate` to activate it.

1. Create and activate a virtual environment, then install the packages:

   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Download the datasets into `data/raw/` (see the Data section).

3. Create the settings file. Copy `.env.example` to `.env` and set `JWT_SECRET` to a long random string. You can make one with:

   ```powershell
   Copy-Item .env.example .env
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   The API refuses to start without `JWT_SECRET`. Never commit `.env`.

4. Train the models. This writes the files the app loads:

   ```powershell
   python -m src.recsys.train
   ```

5. Create the database tables:

   ```powershell
   python -m alembic upgrade head
   ```

6. Start the site and open http://127.0.0.1:8000 :

   ```powershell
   python -m uvicorn api.main:app --port 8000
   ```

7. Optional: run the tests with `python -m pytest`.

Run step 5 again whenever you pull code that changes the database. A "no such table" error means the migration has not been applied.

Training writes:

- `data/processed/movies.csv`
- `data/processed/artifacts/recommender.joblib`
- `results/*.csv` and `results/*.png`

## Data

Put these files in `data/raw/`. They are not committed.

| File | Source |
| --- | --- |
| `ml-latest-small/` (`ratings.csv`, `movies.csv`, `links.csv`) | [MovieLens latest small](https://grouplens.org/datasets/movielens/latest/) |
| `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` | [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| `IMDB Dataset.csv` (`review`, `sentiment`) | [IMDb 50K reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |

MovieLens may also sit directly in `data/raw/` instead of the `ml-latest-small` folder. Movies are kept only when `links.tmdbId` matches a TMDB id.

This run kept **3537 movies** and **70194 ratings** (from 9742 MovieLens movies and 100836 ratings). The IMDb file has **50000** reviews and no movie ids, so it trains the sentiment classifier only.

`TMDB_API_KEY` is optional. When it is unset, movie reception is a Bayesian average of MovieLens ratings (prior strength 20), scaled from the 0.5–5 range into 0–1. This run used that fallback. The code path that would average TMDB review sentiment is in `src/recsys/sentiment.py`.

## Posters

The TMDB 5000 file has no poster path. Posters are fetched by IMDb id from Metahub, cached under `data/processed/posters`, and served by `GET /posters/{imdb_id}`. When a poster is not available, the card shows a letter placeholder. Poster images are third-party content and are not stored in this repository.

Known limitation: the Metahub host had an expired certificate when this was written, so the download retries without certificate verification for that one URL. Use this only for local demos. A production version should fetch posters through the TMDB images API with a TMDB key and remove the verification bypass.

## Community rating

Each movie card shows one read-only score on a 0–10 scale, separate from the signed-in user's own stars. TMDB `vote_average` is used when that film has at least 10 TMDB votes. Otherwise the score is the MovieLens mean multiplied by 2. The number is a Bayesian average with prior strength `m = 50` (`COMMUNITY_PRIOR_M`): `(v / (v + m)) * R + (m / (v + m)) * C`, where `C` is the global mean of the source being shown. A film with no TMDB votes and no MovieLens ratings has `avg_rating` null, not 0. These values are attached once when the model loads.

## Models

- Content: TF-IDF, at most 5000 features, on Porter-stemmed genres, keywords, top 3 cast, director, and overview. User profile is the mean vector of movies rated at least 4.
- Item-kNN: cosine neighbors on the user-item matrix (40 neighbors, negative similarities dropped). The same scores are used for rating prediction and ranking.
- SVD: `TruncatedSVD` with 40 components, fit on raw ratings. Ranking uses the reconstructed value. RMSE uses that value clipped to [0.5, 5].
- Sentiment: TF-IDF (10000 features) plus logistic regression, multinomial naive Bayes, and a random forest. Stratified 80/20 split. Logistic regression won on F1 and is the saved model.
- Hybrid: `alpha * SVD + (1 - alpha) * content`, with `alpha = n_ratings / (n_ratings + k)`. A brand-new user (`n_ratings = 0`) gets popularity. Hybrid + sentiment adds `beta * reception`. `k` and `beta` were chosen on an inner hold-out of fold 0's training rows. Selected values: **k = 5**, **beta = 0.3**.

## Evaluation protocol

Each user's ratings are split into 5 disjoint folds (seed 42). A movie a user rated is never in both train and test. Ranking metrics use test ratings of at least 4 as relevant. NDCG uses the raw test rating as the gain. Users with no relevant test item are left out of precision, recall, F1, and NDCG. Coverage is the fraction of the 3537-movie catalog that appears in some user's top 10. RMSE and MAE are computed for popularity (item mean), item-kNN, and SVD, because content and hybrid scores are not on the star scale.

The cold-start curve uses fold 0 only. 117 users who had at least 10 training ratings and a relevant test item were sampled (cap 120, seed 42). For 1, 3, 5, and 10 known ratings, only that subset is visible to the model.

Paired tests compare each user's mean NDCG@10 for hybrid against each other model (Wilcoxon and paired t, alpha 0.05).

The evaluation covers the base models above. The personalised blend on the Home page (recent activity and search) is not evaluated offline.

## Results from this run

Ranking means across the five folds:

| Model | P@10 | R@10 | NDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.120 | 0.127 | 0.182 | 0.015 |
| Content | 0.038 | 0.049 | 0.062 | 0.271 |
| Item-kNN | 0.006 | 0.007 | 0.009 | 0.487 |
| SVD | 0.195 | 0.250 | 0.315 | 0.134 |
| Hybrid | 0.198 | 0.256 | 0.321 | 0.223 |
| Hybrid + sentiment | 0.201 | 0.260 | 0.320 | 0.121 |

Rating error on the held-out stars:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Popularity | 0.957 | 0.740 |
| Item-kNN | 0.912 | 0.686 |
| SVD | 2.842 | 2.636 |

Hybrid NDCG@10 is higher than popularity, content, item-kNN, and SVD. Those four Wilcoxon p-values are below 0.05 (609 users). Hybrid versus hybrid + sentiment is not: Wilcoxon p = 0.672. On this split, the sentiment term does not change full-history ranking by a detectable amount. The validation search still preferred beta 0.3, which is what the saved model uses.

Cold-start NDCG@10 (117 users, fold 0):

| Known ratings | Popularity | Content | Item-kNN | SVD | Hybrid | Hybrid + sentiment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.082 | 0.018 | 0.036 | 0.028 | 0.018 | 0.042 |
| 3 | 0.084 | 0.021 | 0.030 | 0.028 | 0.021 | 0.043 |
| 5 | 0.088 | 0.029 | 0.021 | 0.038 | 0.039 | 0.064 |
| 10 | 0.091 | 0.031 | 0.031 | 0.110 | 0.099 | 0.127 |

With one or three known ratings, popularity is the strongest list. By ten known ratings, hybrid + sentiment is ahead of SVD and popularity on this sample. That comparison was not given its own significance test.

Sentiment hold-out (10,000 reviews):

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.897 | 0.888 | 0.907 | 0.898 |
| Multinomial NB | 0.858 | 0.856 | 0.859 | 0.858 |
| Random forest | 0.847 | 0.837 | 0.861 | 0.849 |

Logistic-regression confusion matrix (rows are actual negative, positive): `[[4430, 570], [464, 4536]]`. The precision-recall curves are in `results/sentiment_pr.png`. The cold-start plot is `results/cold_start.png`.

## Accounts and personalised recommendations

The page keeps the access token in memory only. A new account picks genres first. With no ratings, `GET /recommendations/me` returns the highest-rated films in those genres (`mode` is `genre`). After the first rating the hybrid model is used. Under five ratings the mode is `warming_up`. At five or more it is `personalised`.

Home mixes three scores, each scaled to 0–1:

`final = w_base * base + w_recent * recent + w_search * search`

`base` is the long-term profile: chosen genres plus ratings, using hybrid SVD and content. `recent` is ratings, watchlist adds, and clicks from the last 14 days. `search` is the last 10 distinct submitted searches from the last 14 days. A missing signal has weight 0 and the other weights are rescaled so they still sum to 1.

`w_recent = min(R_MAX, n_recent / (n_recent + K_R))` with `R_MAX = 0.30` and `K_R = 5`. `w_search = min(S_MAX, n_search / (n_search + K_S))` with `S_MAX = 0.25` and `K_S = 3`. `w_base = 1 - w_recent - w_search`, so the original profile stays at least 0.45. A query adds `SEARCH_QUERY_STRENGTH = 0.3`, split across the top matched movies. A click on a result adds `SEARCH_CLICK_STRENGTH = 0.8` for that movie. Search half-life is `SEARCH_HALF_LIFE_DAYS = 3`. A search in the last 24 hours reserves `SEARCH_MIN_SLOTS = 2` of the top 10 for related movies. All of these live in `src/recsys/config.py`.

Ratings of 4 and above pull the content profile toward that film. Ratings of 2 and below push away from its genres, cast, and keywords. `CONTENT_DISLIKE_WEIGHT` in `src/recsys/config.py` is `0.5`. Clicks are not stored as ratings. `CLICK_SIGNAL_WEIGHT` is `0.15`. A submitted search is stored only in `search_events`: the query text, the normalised query, up to 10 matched movie ids, and the clicked movie id. Typing is not stored, and a search is never saved as a rating. The same normalised query inside 10 minutes updates that row instead of adding another. Recommendation responses do not include the query. Cards say "Based on your recent activity", or "Similar to {title}" when a result was clicked. Clear search history in the account menu, or call `DELETE /users/me/searches`, to delete those rows and rebuild the list.

Stability rules keep the list from jumping after one action: movies in the previous list get a small bonus (`SNAPSHOT_BONUS = 0.05`), at most 40% of the top 10 can change at once (`CHURN_MAX_FRACTION = 0.40`), and results are re-ranked for diversity (`MMR_LAMBDA = 0.7`).

## Privacy

The app stores an email, a username, a bcrypt password hash, movie ratings, a watchlist, interaction events (impression, click, rating, watchlist add), and submitted searches. A search row stores the query text, the matched movie ids, and the result that was clicked. It does not use third-party analytics. MovieLens files stay read-only. Delete one search or clear Search history on Home to drop that influence. Delete the account from the page, or call `DELETE /users/me` while signed in, to remove the user and every related row.

## API

Open data and model routes:

- `GET /health`
- `GET /movies/search?q=`
- `GET /recommendations/similar/{movie_id}?n=10`
- `GET /recommendations/user/{user_id}?n=10`
- `POST /recommendations/profile?n=10` with `{"ratings": [{"movie_id": 1, "rating": 5}]}`
- `GET /posters/{imdb_id}`

Accounts (sign in required for everything below `/auth`):

- `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`
- `GET /users/me`, `DELETE /users/me`

Genres and onboarding:

- `GET /genres` lists the allowed genres. `POST /genres` saves a first selection.
- `GET /users/me/genres` returns the user's genres, the allowed list, and the maximum. `PUT /users/me/genres` replaces the selection.
- `POST /onboarding` saves the first ratings and marks onboarding complete.

Ratings, watchlist, and recommendations:

- `POST /ratings`, `GET /ratings/me`, `DELETE /ratings/{movie_id}`
- `POST /watchlist/{movie_id}`, `GET /watchlist/me`, `DELETE /watchlist/{movie_id}`
- `GET /recommendations/me`
- `POST /interactions` logs a click or similar event.

Search history:

- `POST /search/events`, `GET /users/me/searches`, `DELETE /users/me/searches/{id}`, `DELETE /users/me/searches`

Unknown movies and users return 404. Ratings outside 0.5–5 return 422.

## Tests and manual checks

Run the automated tests with `python -m pytest`.

The page is plain HTML, so the star control is not run by pytest. After starting the server, check these by hand:

- Sign in, search for a title, and set 4.5 stars. The same film shows 4.5 on My ratings and on Recently rated.
- Click those stars again, or Clear. The rating disappears.
- With the stars focused, ArrowRight and ArrowLeft move by half a star.
- Sign out, search is behind the account form. A rating click before sign-in returns you to the form.
- After a rating, the notice says recommendations are updating, then the rated film is gone from Recommended for you.

## Troubleshooting

- **`no such table: search_events` (or any missing table):** run `python -m alembic upgrade head`.
- **API will not start, mentioning `JWT_SECRET`:** create `.env` from `.env.example` and set a long random `JWT_SECRET`.
- **Port 8000 already in use:** an earlier server is still running. Stop it, or start on another port with `--port 8001` and add `http://127.0.0.1:8001` to `CORS_ORIGINS` in `.env`.
- **Browser requests are blocked:** `CORS_ORIGINS` in `.env` must match the address and port you open. Use `127.0.0.1`, not `localhost`, unless you add it.
- **Missing data file when training:** check the file names and folders in the Data section.
- **Form data requires `python-multipart`:** it is pinned in `requirements.txt`. Run `pip install -r requirements.txt` again.

## Limitations

- The test is offline. A hit is a held-out movie the user already rated 4 or higher, so the numbers are not online click rates.
- The personalised blend (recent activity and search) is not evaluated offline.
- Popularity bias remains. The popularity list touches about 1.5% of the catalog because the same films win for almost every user.
- Only movies in both MovieLens and TMDB are kept, so a large part of MovieLens never enters the models.
- IMDb reviews are not linked to these movies. Without `TMDB_API_KEY`, "sentiment" at movie level is a normalized Bayesian rating, not text from that movie.
- Item-kNN has the best RMSE and the worst NDCG here. Its predicted stars sit near the user's mean, which is a weak ranking signal.
- SVD is fit on raw ratings with no user or item bias, so its reconstructed values are a poor star prediction (RMSE 2.84) even though the ordering is competitive.
- Hybrid parameters were tuned on one inner split, then frozen for all five folds.
- Posters depend on a third-party host and use a certificate-verification bypass for that host (see Posters).

## References and credits

- Shelake, V., Fernandes, S., & Shrungare, S. (2025). AI-driven personalized movie recommendations: A content and sentiment-aware model for streaming and digital   entrepreneurship. APTISI Transactions on Technopreneurship, 7(2), 306–317. 
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19.
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. Proceedings of ACL-HLT 2011, 142–150.
- Movie metadata comes from the TMDB 5000 dataset on Kaggle, built from data from The Movie Database (TMDB). This project is not endorsed or certified by TMDB.
