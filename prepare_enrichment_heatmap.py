from sys import argv, stdout, executable
import os
import shlex
import subprocess
import argparse
import gzip
import logging
import inspect
from typing import List, Dict, Tuple, Optional

import numpy as np
import pyBigWig


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"  # test how it looks in bash with e.g.: echo -e "\x1b[34;20m text\e[0m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    pink = "\x1b[35;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "###\t[%(asctime)s] %(filename)s:%(lineno)d: %(name)s %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configureLogging(analysisPrefix=(os.path.basename(__file__)).replace(".py", "")):
    logger = logging.getLogger()
    logger.disabled = False
    logger.setLevel(logging.INFO)

    streamhdlr = logging.StreamHandler(stdout)
    filehdlr = logging.FileHandler(f"{analysisPrefix}.log")

    logger.addHandler(streamhdlr)
    logger.addHandler(filehdlr)

    streamhdlr.setLevel(logging.INFO)
    filehdlr.setLevel(logging.INFO)

    lgrPlainFormat = logging.Formatter('###\t[%(asctime)s] %(filename)s:%(lineno)d: %(name)s %(levelname)s: %(message)s')
    filehdlr.setFormatter(lgrPlainFormat)
    streamhdlr.setFormatter(CustomFormatter())


def str2bool(v):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        lgr.critical("Unrecognized parameter was set for '{}'. Program was aborted.".format(v))
        exit()


def parse_region(region_string: str) -> Tuple[str, int, int]:
    clean_region = region_string.replace(",", "").replace(".", "")
    chrom, positions = clean_region.split(":")
    start, end = positions.split("-")
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        raise ValueError(f"Region end must be larger than start: {region_string}")
    return chrom, start_i, end_i


def parseArgs():
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Current working directory: {}".format(os.getcwd()))
    lgr.info("Command used to run the program: python {}".format(' '.join(str(x) for x in argv)))

    parser = argparse.ArgumentParser(description="Prepare a TSV matrix for the EnrichmentHeatmap track by binning multiple bigWig files over a single region.")
    parser.add_argument("-b", "--bigwigs", help="Space separated list of bigWig files to include.", nargs="+", required=True)
    parser.add_argument("-r", "--region", help="Genomic region to extract, format chr:start-end.", required=True)
    parser.add_argument("-o", "--output", help="Output TSV file.", default="enrichment_heatmap.tsv")
    parser.add_argument("--bin-size", help="Bin size in bases.", type=int, required=False, default=None)
    parser.add_argument("--number-of-bins", help="Total number of bins to use.", type=int, required=False, default=None)
    parser.add_argument("--summary-method", help="pyBigWig summary method.", default="mean",
                        choices=["mean", "average", "max", "min", "stdev", "dev", "coverage", "cov", "sum"])
    parser.add_argument("--nan-to-zero", help="Replace missing values with zeros.", action="store_true")
    parser.add_argument("--metadata", help="Optional TSV with a 'sample' column to attach metadata to each bigWig.", default=None)
    parser.add_argument("--sample-labels", help="Optional comma-separated labels matching the --bigwigs order.", default=None)
    parser.add_argument("--sort-by", help="How to sort rows of the output matrix. Options: input, mean_desc, mean_asc, max_desc, max_asc, metadata:<column>.",
                        default="mean_desc")

    args = parser.parse_args()

    lgr.info(f"List the versions of imported packages:")
    lgr.info(f"numpy: {np.__version__}")
    lgr.info(f"pyBigWig: {pyBigWig.__version__ if hasattr(pyBigWig, '__version__') else 'NA'}")

    lgr.info("Current working directory: {}".format(os.getcwd()))
    command_used = ' '.join(shlex.quote(arg) for arg in [os.path.basename(executable)] + argv)
    lgr.info("Command used to run script: {}".format(command_used))

    # for each argument, log its value:
    lgr.info("Region (-r flag): {}".format(args.region))
    lgr.info("Output (--output flag): {}".format(args.output))
    lgr.info("Bin size (--bin-size flag): {}".format(args.bin_size))
    lgr.info("Number of bins (--number-of-bins flag): {}".format(args.number_of_bins))
    lgr.info("Summary method (--summary-method flag): {}".format(args.summary_method))
    lgr.info("Sort (--sort-by flag): {}".format(args.sort_by))

    if args.bin_size is None and args.number_of_bins is None:
        parser.error("One of --bin-size or --number-of-bins must be provided.")
    if args.bin_size is not None and args.number_of_bins is not None:
        parser.error("Use only one of --bin-size or --number-of-bins.")

    if args.bin_size is not None and args.bin_size <= 0:
        parser.error("--bin-size must be > 0.")
    if args.number_of_bins is not None and args.number_of_bins <= 0:
        parser.error("--number-of-bins must be > 0.")

    return args


def load_metadata(metadata_path: Optional[str]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Returns a mapping sample_name -> metadata dict and an ordered list of metadata field names.
    """
    if metadata_path is None:
        return {}, []
    opener = gzip.open if metadata_path.endswith(".gz") else open
    with opener(metadata_path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if "sample" not in header:
            raise ValueError("Metadata file must contain a 'sample' column.")
        sample_idx = header.index("sample")
        meta_fields = [h for i, h in enumerate(header) if i != sample_idx]
        mapping = {}
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts.extend([""] * (len(header) - len(parts)))
            sample_name = parts[sample_idx]
            mapping[sample_name] = {field: parts[idx] for idx, field in enumerate(header) if idx != sample_idx}
    return mapping, meta_fields


def pick_sample_labels(bigwig_files: List[str], explicit_labels: Optional[str]) -> List[str]:
    if explicit_labels:
        labels = [l.strip() for l in explicit_labels.split(",") if l.strip() != ""]
        if len(labels) != len(bigwig_files):
            raise ValueError("--sample-labels must have the same number of entries as --bigwigs.")
        return labels
    return [os.path.splitext(os.path.basename(path))[0] for path in bigwig_files]


def compute_bin_edges(start: int, end: int, bin_size: Optional[int], number_of_bins: Optional[int]) -> np.ndarray:
    if bin_size:
        edges = np.arange(start, end, bin_size, dtype=int)
        if edges[-1] != end:
            edges = np.append(edges, end)
        return edges
    else:
        return np.linspace(start, end, num=number_of_bins + 1, dtype=int)


def summarize_bigwig(path: str, chrom: str, start: int, end: int, n_bins: int, summary_method: str) -> List[float]:
    with pyBigWig.open(path) as bw:
        stats = bw.stats(chrom, start, end, nBins=n_bins, type=summary_method)
    # pyBigWig returns None for empty bins
    return [np.nan if v is None else float(v) for v in stats]


def sort_rows(matrix: np.ndarray, sort_by: str, labels: List[str], metadata: List[Dict[str, str]]):
    """
    Sort matrix rows in-place, returning updated (matrix, labels, metadata).
    """
    sort_by = sort_by.lower()
    if sort_by == "input":
        return matrix, labels, metadata

    reverse = sort_by.endswith("desc")
    if sort_by.startswith("metadata:"):
        field = sort_by.split("metadata:", 1)[1]
        if field == "":
            return matrix, labels, metadata

        def key_fn(idx):
            return metadata[idx].get(field, "")
    else:
        axis = 1
        if sort_by.startswith("mean"):
            key_values = np.nanmean(matrix, axis=axis)
        elif sort_by.startswith("max"):
            key_values = np.nanmax(matrix, axis=axis)
        else:
            return matrix, labels, metadata

        def key_fn(idx):
            return key_values[idx]

    order = sorted(range(matrix.shape[0]), key=key_fn, reverse=reverse)
    matrix = matrix[order, :]
    labels = [labels[i] for i in order]
    metadata = [metadata[i] for i in order]
    return matrix, labels, metadata


def main():
    configureLogging()
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    args = parseArgs()

    chrom, start, end = parse_region(args.region)
    bin_edges = compute_bin_edges(start, end, args.bin_size, args.number_of_bins)
    n_bins = len(bin_edges) - 1

    metadata_map, metadata_fields = load_metadata(args.metadata)
    sample_labels = pick_sample_labels(args.bigwigs, args.sample_labels)

    rows = []
    labels = []
    row_metadata = []
    for bw_path, sample_name in zip(args.bigwigs, sample_labels):
        if not os.path.exists(bw_path):
            raise FileNotFoundError(f"bigWig file not found: {bw_path}")
        values = summarize_bigwig(bw_path, chrom, start, end, n_bins, args.summary_method)
        rows.append(values)
        labels.append(sample_name)
        row_metadata.append(metadata_map.get(sample_name, {}))

    matrix = np.array(rows, dtype=float)
    if args.nan_to_zero:
        matrix = np.nan_to_num(matrix, nan=0.0)

    matrix, labels, row_metadata = sort_rows(matrix, args.sort_by, labels, row_metadata)

    header = ["sample"] + metadata_fields + [f"bin_{i:05d}" for i in range(matrix.shape[1])]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as handle:
        handle.write("#enrichment_heatmap_version\t1\n")
        handle.write(f"#region\t{chrom}:{start}-{end}\n")
        handle.write(f"#chrom\t{chrom}\n")
        handle.write(f"#start\t{start}\n")
        handle.write(f"#end\t{end}\n")
        handle.write(f"#bin_edges\t{','.join(map(str, bin_edges.tolist()))}\n")
        handle.write(f"#summary_method\t{args.summary_method}\n")
        handle.write(f"#sort_by\t{args.sort_by}\n")
        handle.write(f"#nan_to_zero\t{args.nan_to_zero}\n")
        handle.write("\t".join(header) + "\n")
        for idx, label in enumerate(labels):
            meta_values = [row_metadata[idx].get(field, "") for field in metadata_fields]
            row_values = [f"{v:.6g}" if not np.isnan(v) else "nan" for v in matrix[idx]]
            handle.write("\t".join([label] + meta_values + row_values) + "\n")

    lgr.info("All done, thank you!")
    lgr.info(f"Wrote matrix with {matrix.shape[0]} samples and {matrix.shape[1]} bins to {args.output}")


if __name__ == "__main__":
    main()
