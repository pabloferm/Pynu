from PyNu import Plot


class Report:
    def __init__(self,
                 analysis_output_file,
                 analysis_input_file=False,
                 directory=''):

        self.pynuplot = Plot(args.hdf5_file, directory=args.directory)

    def Make(self):
        pass
