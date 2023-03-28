import sys
import os
import argparse

from PyNu import PyNu


def main():

    print('=============================================================\n' +
          '===================== Analysis setup ========================\n' +
          '=============================================================')

    # Read arguments
    ############################
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "xml_file",
        type=str,
        nargs='?',
        default=os.environ['PYNU'] +
        '/examples/AnalysisFiles/test.xml',
        help='Input analysis file in xml format.')
    parse.add_argument(
        '-p',
        '--point',
        nargs='+',
        type=int,
        default=None,
        help='Specify analysis point or points (p0 p1 p2 p3) to run.')
    parse.add_argument(
        '-rp',
        '--range_of_points',
        nargs='+',
        type=int,
        default=None,
        help='Specify range (start and end) of analysis points to run, p0 p3 = p0 p0+1 p0+2 ... p3-1 p3. Edges are included.')
    parse.add_argument(
        '-o',
        '--outfile',
        nargs='?',
        type=str,
        default='outfile.hdf5',
        help='Analysis output file.')
    parse.add_argument(
        "--multi",
        dest='multiproc',
        default=False,
        action='store_true',
        help='Option for running the analysis with multiprocessing (recommended locally).')
    parse.add_argument(
        "--cluster",
        dest='cluster',
        default=False,
        action='store_true',
        help='Option for submitting jobs to a cluster.')
    parse.add_argument(
        "--mcmc",
        dest='mcmc',
        default=False,
        action='store_true',
        help='Option for sampling parameter space using Markov Chain Monte Carlo.')
    args = parse.parse_args()

    # Setup analysis from xml file
    ################################
    pynu = PyNu(args.xml_file, verbosity=False)

    # Setup running points and options
    ####################################
    if args.mcmc is False:
        if args.range_of_points is None and args.point is not None:
            points = [*set(args.point)]
            if points >= pynu.Analysis.NumberOfPhysPoints:
                sys.exit('Point out of range for this analysis.')
        elif args.range_of_points is not None and args.point is None:
            points = list(
                range(
                    int(args.range_of_points[0]),
                    1 + int(args.range_of_points[-1])))
            if points[-1] >= pynu.Analysis.NumberOfPhysPoints:
                sys.exit('Point out of range for this analysis.')
        else:  # run over all analysis points
            points = range(0, pynu.Analysis.NumberOfPhysPoints)

    if args.multiproc:
        import multiprocessing

    if args.mcmc:
        import emcee

    # Setup output file
    ############################
    if (args.cluster and (points[0] == 0 or not os.path.isfile(
            args.outfile))) or not args.cluster or not os.path.isfile(args.outfile):
        pynu.CreateOutFile(args.outfile)

    # Set analysis
    ################

    ''' Parallelization '''
    if args.multiproc:
        import multiprocessing
        cores = multiprocessing.cpu_count()

        ''' Markov chain wandering '''
        if args.mcmc:
            import emcee
            import numpy as np
            nwalkers = 2**4
            ndim = pynu.Analysis.NumberOfPhys
            nsteps = 200
            initial = np.zeros((nwalkers, ndim))

    # Loop over specified points of analysis
    else:
        for p in points:
        # Compute weights at a given point of the physics grid (fixed part for
        # nuisance minimisation)
            print(f'Analyzing point {p} of {pynu.Analysis.NumberOfPhysPoints} points.')
            pynu.FitBinnedLLH(p)


if __name__ == '__main__':
    import cProfile
    from pstats import SortKey
    import pstats
    cProfile.run('main()', 'output.dat')

    with open('output_time.txt', 'w') as f:
        p = pstats.Stats('output.dat', stream=f)
        p.sort_stats('time').print_stats()

    with open('output_calls.txt', 'w') as f:
        p = pstats.Stats('output.dat', stream=f)
        p.sort_stats('calls').print_stats()
