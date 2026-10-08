# Dataset construction

These scripts build `../data/citation_arrays.json`. They query live services (CSRankings, OpenAlex, Semantic Scholar) and take hours. The services' contents change over time, so a re-run reflects their contents at that time.

Requires `requests`. The OpenAlex scripts need a contact address for OpenAlex's polite pool:

```bash
export OPENALEX_EMAIL=you@example.org
```

Run the scripts in order, from this directory. Intermediate files are written to `../build_output/`.

| script | what it does | writes |
|---|---|---|
| `01_select_researchers.py` | downloads the CSRankings data and selects the top 50 faculty at each of the top 20 North American departments | `researchers_list.json` |
| `02_match_profiles.py` | matches each researcher to an OpenAlex author profile, falling back to Semantic Scholar, and cross-checks each profile's number of works against the researcher's CSRankings publication count | `profiles.json` |
| `03_fetch_citation_arrays.py` | fetches the citation count of every paper of each matched profile | `citation_arrays.json` |
| `04_export_dataset.py` | writes the dataset file: the author ID and citation array of each researcher | `../data/citation_arrays.json` |

`common.py` holds the code shared by the scripts: the API client, name matching and file helpers. Options are listed by `python <script> --help`; `02_match_profiles.py --limit N` processes only the first N researchers, which is useful for a trial run.
