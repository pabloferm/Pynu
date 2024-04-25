import sys
import os
import argparse

from pynu import PyNuFit


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
        '/../examples/AnalysisFiles/test.xml',
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
        "--ncores",
        nargs='?',
        type=int,
        default=None,
        help='Option for especifying the number of cores to be used by multiprocessing.')
    parse.add_argument(
        "--cluster",
        dest='cluster',
        default=False,
        action='store_true',
        help='Option for submitting jobs to a cluster.')
    parse.add_argument(
        "--spherical_grid",
        dest='spherical_grid',
        default=False,
        action='store_true',
        help='Option for running the analysis only over the grid points inside a n-dimensional elipse.')
    parse.add_argument(
        '--radius',
        nargs='?',
        type=float,
        default=1.0,
        help='Relative radius for the n-dimensional elipse if the spherical_grid option is true.')
    parse.add_argument(
        "--mcmc",
        dest='mcmc',
        default=False,
        action='store_true',
        help='Option for sampling parameter space using Metropolis-Hastings Markov Chain Monte Carlo.')
    parse.add_argument(
        "--hmc",
        dest='hmc',
        default=False,
        action='store_true',
        help='Option for sampling parameter space using Hamiltonian Markov Chain Monte Carlo.')
    parse.add_argument(
        "--svgd",
        dest='svgd',
        default=False,
        action='store_true',
        help='Option for sampling parameter space using Stein Variation Gradient Descent (SVGD).')
    args = parse.parse_args()

    # Setup analysis from xml file
    ################################
    pynufit = PyNuFit(args.xml_file, verbosity=True)

    # Setup running points and options
    ####################################
    if args.mcmc is False:
        if args.range_of_points is None and args.point is not None:
            points = [*set(args.point)]
            if points[-1] >= pynufit.Analysis.NumberOfPhysPoints:
                sys.exit('Point out of range for this analysis.')
        elif args.range_of_points is not None and args.point is None:
            points = list(
                range(
                    int(args.range_of_points[0]),
                    1 + int(args.range_of_points[-1])))
            if points[-1] >= pynufit.Analysis.NumberOfPhysPoints:
                sys.exit('Point out of range for this analysis.')
        else:  # run over all analysis points
            points = range(0, pynufit.Analysis.NumberOfPhysPoints)

    if args.multiproc:
        import multiprocessing

    if args.mcmc:
        pass

    if args.spherical_grid:
        pynufit.Analysis.set_spherical_grid(radius=args.radius)


    # Setup output file
    ############################
    # if (args.cluster and (points[0] == 0 or not os.path.isfile(
        # args.outfile))) or (not args.cluster) or
        # (os.path.isfile(args.outfile)):
    if not os.path.isfile(args.outfile):
        print("Creating new analysis file.")
        pynufit.CreateOutFile(args.outfile)
    else:
        print("Analysis file already exists.")
        pynufit.SetOutFile(args.outfile)

    # Set analysis
    ################
    ''' Parallelization '''
    if args.multiproc:
        import multiprocessing
        if args.ncores:
            cores = args.ncores
        else:
            cores = multiprocessing.cpu_count()

        ''' Markov chain wandering '''
        if args.mcmc:
            import numpy as np
            nwalkers = 2**4
            ndim = pynufit.Analysis.NumberOfPhys
            nsteps = 200
            initial = np.zeros((nwalkers, ndim))

        # elif args.hmc:
        #     samples = 100
        #     for s in range(points):

        else:
            processes = []
            for i,p in enumerate(points):
                print(
                    f'Processing point {p} of {pynufit.Analysis.NumberOfPhysPoints} points in the analysis.')
                if (i+1) % cores == 0:
                    for proc in processes:
                        proc.join()
                    processes = []
                proc = multiprocessing.Process(target=pynufit.FitModel, args=[p,])
                proc.start()
                processes.append(proc)
                

    # Loop over specified points of analysis
    else:
        for p in points:
            # Compute weights at a given point of the physics grid (fixed part for
            # nuisance minimisation)
            print(
                f'Processing point {p} of {pynufit.Analysis.NumberOfPhysPoints} points in the analysis.')
            pynufit.FitModel(p)
            print('=====================================================')


if __name__ == '__main__':
    import cProfile
    import pstats
    cProfile.run('main()', 'hmc.dat')

    with open('hmc_time.txt', 'w') as f:
        p = pstats.Stats('hmc.dat', stream=f)
        p.sort_stats('time').print_stats()

    with open('hmc_calls.txt', 'w') as f:
        p = pstats.Stats('hmc.dat', stream=f)
        p.sort_stats('calls').print_stats()

    # main()
