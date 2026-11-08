import xml.etree.ElementTree as ET
import sys


class ParseXML:
    """Reads the experiment files and exposures and the oscillation point of an event XML."""

    def __init__(self, xmlfile):
        self.tree = ET.parse(xmlfile)  # create element tree object
        self.root = self.tree.getroot()  # get root element of XML file

        self.Fixed = []
        self.FixedValue = {}

        self.Experiments = {}

    def get_analysis(self):
        self.read_experiments()
        self.read_oscillations()

        del self.root

    def read_experiments(self):
        """Reads each experiment's MC and data files and exposures, keyed by detector and source."""
        for source in self.root.iter("NeutrinoExperiment"):
            src = source.find("source")
            self.Experiments[source.attrib["name"]] = {
                src.attrib["name"]: {
                    "MCFiles": [fi.attrib["name"] for fi in src.findall("MCFiles")],
                    "TotalMCexposure": float(src.find("MCexposure").text),
                    "DataFiles": [fi.attrib["name"] for fi in src.findall("DataFiles")],
                    "Exposure": float(src.find("exposure").text),
                }
            }

    def read_oscillations(self):
        """Reads the number of flavors and the oscillation parameter values, in XML order."""
        oscillations = list(self.root.iter("NeutrinoOscillations"))
        if len(oscillations) != 1:
            sys.exit("An event XML must define exactly one NeutrinoOscillations block.")
        self.Flavors = int(oscillations[0].find("flavors").text)
        for fix in oscillations[0].findall("fixed"):
            s = fix.attrib["name"]
            self.FixedValue[s] = float(fix.find("value").text)
            self.Fixed.append(s)
