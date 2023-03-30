import AnalysisReader as AR  # contains parse class to read and setup the analysis


class Plot:

	def __init__(self, analysis_input_file, analysis_output_file, directory=None):

		''' Set up basic analysis variables and structure to build full analysis '''
        self.Analysis = AR.parse(analysis_file, check=self.verbosity)