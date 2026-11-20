# HAR70+ recordings

The study uses the HAR70+ dataset by Aleksej Logacjov and Astrid Ustad. Two synchronized triaxial Axivity AX3 accelerometers, attached to the lower back and right thigh, recorded 18 older adults at a nominal 50 Hz. Each CSV belongs to one participant. [UCI dataset record](https://archive.ics.uci.edu/dataset/780/har70) · [Dataset DOI](https://doi.org/10.24432/C5CW3D) · [Dataset license, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Prepare the original inputs

From the repository root, run the following command with Python 3. It downloads the public UCI archive (42.2 MiB) and installs the 18 verified CSVs in `data/raw/` (195.6 MiB in total).

```sh
python data/prepare.py
```

For an offline installation with the study's original `HAR.zip` or a downloaded UCI ZIP, use the local archive.

```sh
python data/prepare.py --archive /path/to/HAR.zip
```

Use `--output /path/to/new_directory` to choose a different destination. The command verifies an existing directory and reports a mismatch if any expected file is missing or altered. To check an existing download without creating files, run the following.

```sh
python data/prepare.py --verify --output data/raw
```

## What was verified

All 18 research CSVs (IDs 501–518) matched the public UCI download byte for byte, including row order and numeric formatting. Each has the same eight columns in this order: `timestamp`, `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`, `label`. The six acceleration channels are measured in g. Across the 2,259,597 rows, all fields were populated, all six channels were finite, and timestamps increased strictly within each participant. The raw count refers to timestamped samples before any preprocessing or windowing.

| Label | Activity | Samples |
| ---: | --- | ---: |
| 1 | Walking | 1,079,312 |
| 3 | Shuffling | 66,058 |
| 4 | Ascending stairs | 4,560 |
| 5 | Descending stairs | 4,978 |
| 6 | Standing | 418,055 |
| 7 | Sitting | 483,452 |
| 8 | Lying | 203,182 |

The [manifest](manifest.csv) records each participant's original byte count, SHA-256, row count and class counts. Its last two columns give the number of successive timestamp differences unequal to 20 ms and the longest difference in milliseconds. The files retain their original timing, including short jitter and larger gaps. The UCI ZIP checked during packaging had SHA-256 `5db89b85c1a4f494f51fcc6661eb2ad41dd7efbc4799c776e794c56fb439e5c3`. The preparation script checks participant-level hashes so the same inputs can be restored from either archive layout.
