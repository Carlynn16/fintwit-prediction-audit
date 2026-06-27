# Data Dictionary — tweets_raw.csv

> Source: derived from `OLD project/validated_predictions.csv`.  
> All real account handles replaced with pseudonyms (see §Pseudonymisation below).

| Column | dtype | Meaning | Notes |
|---|---|---|---|
| `tweet_id` | str | Platform tweet identifier | **ID-precision caveat**: source stored as float64; least-significant digits may be lost. Treat as opaque label only — not a reliable join key. |
| `conversation_id` | str | Thread/conversation identifier | Same ID-precision caveat as `tweet_id`. |
| `tweet_type` | str | Thread role of the tweet | Values: `parent` (original post) or `reply`. |
| `author` | str | Account that posted the tweet | **Pseudonymised**: real handle replaced with `account_NN`. Mapping in `data/account_mapping.csv` (gitignored). |
| `text` | str | Raw tweet body text | Primary NLP input for all downstream phases. Not cleaned or truncated. |
| `created_at` | datetime (tz-aware UTC) | UTC timestamp of posting | Parsed with `pd.to_datetime(..., utc=True)`. |
| `created_date` | date | Calendar date of posting (UTC) | Derived from `created_at`; stored as Python `date` object. |
| `reply_to_tweet_id` | str | Tweet ID being replied to | Null for non-reply tweets. Same ID-precision caveat. |
| `reply_to_user` | str | Handle of the user being replied to | Pseudonymised if the target is a known author in this dataset; left as-is otherwise. Null for non-reply tweets. |
| `likes` | Int64 (nullable) | Number of likes at collection time | Missing values possible for older or deleted tweets. |
| `retweets` | Int64 (nullable) | Number of retweets at collection time | Missing values possible. |
| `replies_count` | Int64 (nullable) | Number of replies at collection time | Missing values possible. |
| `views` | Int64 (nullable) | Number of views at collection time | Missing values possible; view counts not available for all tweet vintages. |
| `author_followers` | Int64 (nullable) | Follower count at collection time | Snapshot at scrape time; not longitudinal. |
| `author_following` | Int64 (nullable) | Following count at collection time | Snapshot at scrape time. |
| `author_verified` | boolean (nullable) | Legacy blue-tick verified status | Original platform verified badge (pre-2023 scheme). |
| `author_blue_verified` | boolean (nullable) | Twitter Blue / X Premium subscriber status | Post-2023 paid verification badge. |

## Pseudonymisation

Each of the 51 unique author handles was mapped to `account_01` … `account_51` by sorting all unique handles alphabetically and assigning sequential numbers. The same mapping was applied to `reply_to_user` wherever the target handle matched a known author; unrecognised reply targets were left unchanged. The full `handle → pseudonym` mapping is stored in `data/account_mapping.csv`, which is listed in `.gitignore` and must never be committed to the public repository.

## ID-precision caveat

The source CSV exported `tweet_id`, `conversation_id`, and `reply_to_tweet_id` as `float64` in scientific notation (e.g. `1.8924e+18`). Twitter/X tweet IDs are 64-bit integers; float64 has only 53 bits of mantissa, so values above ~9 × 10¹⁵ lose precision. The affected columns are retained as string identifiers for thread-structure reference only — they should not be used to join against any external tweet dataset where exact ID matching is required.
