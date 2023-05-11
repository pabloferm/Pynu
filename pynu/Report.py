import numpy as np
import pylatex
from pylatex import Command, NoEscape, Section, Figure, Subsection
# from pylatex.utils import italic
import os
from pynu import Plot


class Report:
    def __init__(self,
                 analysis_output_file,
                 analysis_input_file,
                 directory='',
                 doctype='article',
                 author=''):

        self.pynuplot = Plot(
            analysis_output_file,
            analysis_input_file=analysis_input_file,
            directory=directory)

        self.reportfile = directory + 'report_' + doctype
        self.author = author
        self.directory = directory

        self.analysis = self.pynuplot.AnalysisInput
        self.analysis.get_analysis()

        geometry_options = {"tmargin": "2cm", "lmargin": "2cm"}
        self.doc = pylatex.Document(
            documentclass=doctype,
            geometry_options=geometry_options)

        self.analysis_interpreter()

    def write_report(self):
        self.doc.generate_pdf(self.reportfile, clean_tex=False)
        self.doc.generate_tex()

    def make_title(self):
        self.title = self.source + self.scenario + self.analysis_type + 'with the ' + self.det + self.detector_type
        self.doc.preamble.append(Command('title', self.title))
        self.doc.preamble.append(Command('author', self.author))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

    def make_introduction(self):
        introduction = 'We present a ' + self.analysis_data + 'using the ' + self.source + \
            'neutrino sources, measured by the ' + self.det + self.detector_type + '.\n'
        introduction += 'In this analysis, we fit the '
        with self.doc.create(Section('Introduction')):
            self.doc.append(introduction)

    def make_results(self):
        self.pynuplot.ResultPlotsMatrix()
        with self.doc.create(Section('Results')):
            with self.doc.create(Figure(position='h!')) as results_plot:
                results_plot.add_image(
                    self.directory +
                    '/ResultPlotsMatrix.png',
                    width='15 cm')
                results_plot.add_caption(
                    'Grid of plots summarizing the results of this analysis.')

    def make_nuisance(self):
        self.pynuplot.NuisancePlots()
        with self.doc.create(Section('Nuisance Parameters')):
            with self.doc.create(Figure(position='h!')) as results_plot:
                results_plot.add_image(
                    self.directory +
                    '/NuisancePlots.png',
                    width='15 cm')
                results_plot.add_caption(
                    'Grid of plots summarizing the nuisance of this analysis.')

    def analysis_interpreter(self):
        self.source, sources = self.list_items('sources')
        self.det, dets = self.list_items('detectors')

        if dets * sources == 1:
            self.analysis_type = 'analysis '
        elif dets * sources > 1:
            self.analysis_type = 'combined analysis '

        self.detector_type = 'detector'
        if dets > 1:
            self.detector_type += s

        self.scenario = self.which_oscillation()

        if self.analysis.DataFiles.values():
            self.analysis_data = 'data fit '
        else:
            self.analysis_data = 'sensitivity analysis '

    def which_oscillation(self):
        if 'Osc' in self.analysis.SCENARIO:
            if self.analysis.Flavors == 3:
                return 'three-flavor neutrino oscillation '
            elif self.analysis.Flavors > 3:
                return 'sterile neutrino '

    def list_items(self, block):
        if block == 'sources':
            array = self.analysis.sources
        elif block == 'detectors':
            array = self.analysis.detectors

        number = len(array)

        if len(array) == 1:
            items = array[0] + ' '
        else:
            items = ''
            for s in array:
                if s == array[-1]:
                    items = items[:-2] + 'and ' + s
                items += s + ', '

        return items, number
