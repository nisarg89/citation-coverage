# Dataset

`citation_arrays.json` holds citation records for 1,034 researchers: for each researcher, the citation count of every paper.

```json
[{"author_id": "https://openalex.org/A5068811427", "citation_array": [3494, 531, 515, ...]}, ...]
```

| field | meaning |
|---|---|
| `author_id` | the researcher's public author ID: an OpenAlex author URL, or a Semantic Scholar author ID (a numeric string) |
| `citation_array` | citation counts of the researcher's papers, in non-increasing order (so `c_k = citation_array[k-1]`) |

Records are listed in order of `author_id`.

## Source

Citation counts come from OpenAlex, or from Semantic Scholar where no OpenAlex profile was matched, and were retrieved on 2026-05-01 (UTC). The code that produces the file is in [`../build`](../build).

## License

[CC BY 4.0](LICENSE.md). Please attribute as: Nisarg Shah and Vijay Vazirani, *Why Leave Citations on the Table?* (dataset).
