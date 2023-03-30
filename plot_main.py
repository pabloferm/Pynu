import sys
import os
import argparse

from PyNu import Plot

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "hdf5_file",
        type=str,
        nargs='?',
        default=None,
        help='Output analysis file in hdf5 format.')
    args = parse.parse_args()

    pynuplot = Plot(args.hdf5_file)

if __name__ == '__main__':
	main()