# How often did Telohelm call a model during an episode?

On the 20 matched scenario positions in each of four streams, Telohelm's learning condition made between 34 and 86 in-episode model calls. The same condition met the goal without help on 7 to 17 of those 20 positions per stream. A position can require several calls. Reading the two columns together keeps a low call count from being mistaken for a successful outcome. [Frozen study records](https://doi.org/10.5281/zenodo.23993339)

| Stream | Matched positions | In-episode model calls | Goals met without help |
|---|---:|---:|---:|
| 4301 | 20 | 58 | 7 |
| 4302 | 20 | 83 | 10 |
| 4303 | 20 | 86 | 9 |
| 4304 | 20 | 34 | 17 |

A stream is one schedule of simulated scenarios. The learning condition is labelled `L` in the records. The comparison uses five positions in each of four phases. In stream 4301 the phase counts are 43, 15, 0 and 0 model calls. Their sum gives 58. Calls used to build candidate procedures between episodes are recorded separately. These four rows describe one simulated task.

## Check the numbers

Download the [shared study ZIP (322 KiB)](https://zenodo.org/api/records/23993339/files/shared-study-files-and-frozen-analysis.zip/content) and the [reader](read_model_calls.py) into the same folder. Run this command there using Python 3.9 or newer.

```sh
python3 read_model_calls.py shared-study-files-and-frozen-analysis.zip
```

The reader verifies the ZIP and the frozen analysis by SHA-256, adds the phase counts within each stream and compares the call and goal totals with the archive's cross-stream summary. The table above gives the expected output. For an independent rebuild from the original episode records, follow the repository's [table reproduction instructions](../README.md#reproduce-the-papers-tables).

Source and attribution. Simon Vajda, Telohelm study records, CC BY 4.0. [Data and licence](https://doi.org/10.5281/zenodo.23993339) · [Software](https://doi.org/10.5281/zenodo.23993336)
