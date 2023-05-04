import sys
import os
import argparse

from pynu import Report


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "hdf5_file",
        type=str,
        nargs='?',
        default=None,
        help='Output analysis file in hdf5 format.')
    parse.add_argument(
        "-xml",
        '--xml_file',
        type=str,
        nargs='?',
        default=None,
        help='Input analysis file in xml format.')
    parse.add_argument(
        "-dir",
        '--directory',
        type=str,
        nargs='?',
        default='',
        help='Path to folder to store the analysis.')

    args = parse.parse_args()

    pynureport = Report(
        args.hdf5_file,
        args.xml_file,
        directory = args.directory,
        doctype = 'article',
        # doctype = 'beamer',
        author = 'Pablo')

    pynureport.make_title()
    pynureport.make_introduction()
    pynureport.make_results()
    pynureport.make_nuisance()

    pynureport.write_report()


if __name__ == '__main__':
    main()
