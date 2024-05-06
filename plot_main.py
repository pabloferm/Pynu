import argparse

from pynu import Plot


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "hdf5_file",
        type=str,
        nargs="?",
        default=None,
        help="Output analysis file in hdf5 format.",
    )
    parse.add_argument(
        "-xml",
        "--xml_file",
        type=str,
        nargs="?",
        default=None,
        help="Input analysis file in xml format.",
    )
    parse.add_argument(
        "-dir",
        "--directory",
        type=str,
        nargs="?",
        default="",
        help="Path to folder to store the analysis.",
    )

    args = parse.parse_args()

    pynuplot = Plot(
        args.hdf5_file, directory=args.directory, analysis_input_file=args.xml_file
    )

    pynuplot.ResultPlotsMatrix()

    pynuplot.NuisancePlots()


if __name__ == "__main__":
    main()
