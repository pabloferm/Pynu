"""
Neutrino physics XML configuration parser.

This module handles parsing and validation of XML analysis files for neutrino
physics experiments, organizing parameters into physics, nuisance, and fixed categories.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import collections
import numpy as np
import itertools


class ParameterType(Enum):
    """Enumeration of parameter types in the analysis."""
    PHYSICS = "physics"
    NUISANCE = "nuisance"
    FIXED = "fixed"


class XMLParseError(Exception):
    """Custom exception for XML parsing errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


@dataclass
class NuisanceParameter:
    """Container for nuisance parameter information."""
    name: str
    nominal: float
    sigma: float
    distribution: str


@dataclass
class PhysicsParameter:
    """Container for physics parameter information."""
    name: str
    true_value: Union[float, str]
    min_value: float
    max_value: float
    num_points: int
    edges: List[float]
    grid: Optional[np.ndarray] = None


@dataclass
class FixedParameter:
    """Container for fixed parameter information."""
    name: str
    value: Union[float, str]


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""
    name: str
    target: str
    sources: Dict[str, Dict[str, Union[List[str], float]]]


class ParseXML:
    """Parse and manage neutrino physics analysis XML configuration files.
    
    This class reads an XML analysis file, extracts physics parameters (to be fitted),
    nuisance parameters (systematic uncertainties), and fixed parameters, organizing
    them by source/scenario.
    
    Attributes:
        check: Whether to perform consistency checks during parsing
        tree: Parsed XML element tree
        root: Root element of XML tree
    """

    # XML element names
    NEUTRINO_SOURCE = "NeutrinoSource"
    NEUTRINO_TARGET = "NeutrinoTarget"
    NEUTRINO_EXPERIMENT = "NeutrinoExperiment"
    NEUTRINO_OSCILLATIONS = "NeutrinoOscillations"
    
    # Parameter names
    ORDERING = "Ordering"
    MASS_ORDERINGS = {"normal", "inverted"}
    
    # Attributes and elements
    ATTR_NAME = "name"
    ATTR_STATUS = "status"
    ATTR_TARGET = "target"

    def __init__(self, xmlfile: Union[str, Path] = "AnalysisFiles/test.xml", check: bool = False) -> None:
        """Initialize XML parser with analysis configuration file.
        
        Args:
            xmlfile: Path to XML analysis input file
            check: Whether to perform consistency checks
            
        Raises:
            FileNotFoundError: If xmlfile does not exist
            XMLParseError: If XML file is invalid
        """
        self.check = check
        self._xmlfile = Path(xmlfile)
        
        if not self._xmlfile.exists():
            raise FileNotFoundError(f"XML file not found: {self._xmlfile}")
        
        try:
            self.tree = ET.parse(self._xmlfile)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            raise XMLParseError(f"Failed to parse XML file: {e}")
        
        # Initialize parameter containers
        self._init_containers()
        
        # Spherical grid parameters
        self.n_sphere = False
        self.n_sphere_cut: Optional[np.ndarray] = None

    def _init_containers(self) -> None:
        """Initialize all parameter dictionaries and lists."""
        # Nuisance parameters
        self.nuisance_params: Dict[str, List[NuisanceParameter]] = {}
        self.nuisance_by_source: Dict[str, Dict[str, NuisanceParameter]] = {}
        
        # Physics parameters
        self.physics_params: Dict[str, List[PhysicsParameter]] = {}
        self.physics_by_source: Dict[str, Dict[str, PhysicsParameter]] = {}
        
        # Fixed parameters
        self.fixed_params: Dict[str, List[FixedParameter]] = {}
        self.fixed_by_source: Dict[str, Dict[str, FixedParameter]] = {}
        
        # Experiments
        self.experiments: Dict[str, ExperimentConfig] = {}
        
        # Analysis metadata
        self.sources: List[str] = []
        self.targets: List[str] = []
        self.detectors: List[str] = []
        self.oscillations: List[str] = []
        self.flavors: Optional[int] = None
        self.scenario: Optional[str] = None
        
        # Summary statistics
        self.num_physics_params = 0
        self.num_nuisance_params = 0
        self.num_physics_points = 0
        self.full_physics_grid: Optional[np.ndarray] = None

    def get_analysis(self) -> None:
        """Parse and initialize all analysis variables.
        
        Reads sources, detectors, experiments, and oscillation parameters from XML,
        constructs the physics grid, and optionally performs consistency checks.
        
        Raises:
            ValidationError: If consistency checks fail
        """
        try:
            self.sources = self._read_block(self.NEUTRINO_SOURCE)
            self.targets = self._read_block(self.NEUTRINO_TARGET)
            self.detectors = self._read_block(self.NEUTRINO_EXPERIMENT)
            self.oscillations = self._read_block(self.NEUTRINO_OSCILLATIONS)
            
            self._validate_oscillations()
            
            # Build physics grids
            physics_grids = self._build_physics_grids()
            self.full_physics_grid = self._build_cartesian_grid(physics_grids)
            
            # Calculate statistics
            self.num_physics_params = sum(
                len(params) for params in self.physics_params.values()
            )
            self.num_nuisance_params = sum(
                len(params) for params in self.nuisance_params.values()
            )
            self.num_physics_points = int(np.prod([p.num_points for params in self.physics_params.values() 
                                                   for p in params]))
            
            if self.check:
                self._validate_all()
            
        except (XMLParseError, ValidationError) as e:
            raise
        except Exception as e:
            raise XMLParseError(f"Error during analysis parsing: {e}") from e
        finally:
            del self.root

    def _read_block(self, block_name: str) -> List[str]:
        """Read a named block from XML and return list of names.
        
        Args:
            block_name: Name of XML block to read
            
        Returns:
            List of item names in the block
        """
        items = []
        for element in self.root.iter(block_name):
            if self._is_enabled(element):
                name = element.attrib.get(self.ATTR_NAME)
                if name:
                    items.append(name)
                    
                    if block_name == self.NEUTRINO_OSCILLATIONS:
                        self.flavors = self._parse_int(element, "flavors")
                    elif block_name == self.NEUTRINO_EXPERIMENT:
                        self._parse_experiment(element, name)
                    else:
                        self._parse_parameters(element, name)
        
        self._print_block_summary(block_name, items)
        return items

    def _parse_experiment(self, element: ET.Element, name: str) -> None:
        """Parse experiment configuration from XML element.
        
        Args:
            element: XML element containing experiment data
            name: Name of the experiment
        """
        target = element.find(self.ATTR_TARGET)
        target_name = target.attrib.get(self.ATTR_NAME) if target is not None else None
        
        sources_dict = {}
        for src_elem in element.findall("source"):
            if self._is_enabled(src_elem):
                src_name = src_elem.attrib.get(self.ATTR_NAME)
                if src_name:
                    sources_dict[src_name] = {
                        "MCFiles": self._parse_file_list(src_elem, "MCFiles"),
                        "DataFiles": self._parse_file_list(src_elem, "DataFiles"),
                        "Exposure": self._parse_float(src_elem, "exposure"),
                        "MCExposure": self._parse_float(src_elem, "MCexposure"),
                    }
        
        self.experiments[name] = ExperimentConfig(
            name=name,
            target=target_name or "",
            sources=sources_dict
        )

    def _parse_parameters(self, element: ET.Element, source_name: str) -> None:
        """Parse all parameter types from a source element.
        
        Args:
            element: XML element containing parameters
            source_name: Name of the source/scenario
        """
        self.nuisance_params[source_name] = self._parse_nuisance_parameters(element)
        self.physics_params[source_name] = self._parse_physics_parameters(element)
        self.fixed_params[source_name] = self._parse_fixed_parameters(element)

    def _parse_nuisance_parameters(self, element: ET.Element) -> List[NuisanceParameter]:
        """Parse nuisance parameters from XML element.
        
        Args:
            element: XML element containing nuisance parameters
            
        Returns:
            List of NuisanceParameter objects
        """
        params = []
        for nuis_elem in element.findall("nuisance"):
            if self._is_enabled(nuis_elem):
                name = nuis_elem.attrib.get(self.ATTR_NAME)
                
                if name == self.ORDERING:
                    raise ValidationError(
                        "Neutrino mass ordering cannot be a nuisance parameter. "
                        "Please test both ordering hypotheses separately."
                    )
                
                params.append(NuisanceParameter(
                    name=name,
                    nominal=self._parse_float(nuis_elem, "nominal"),
                    sigma=self._parse_float(nuis_elem, "sigma"),
                    distribution=self._parse_string(nuis_elem, "distribution"),
                ))
        
        return params

    def _parse_physics_parameters(self, element: ET.Element) -> List[PhysicsParameter]:
        """Parse physics parameters from XML element.
        
        Args:
            element: XML element containing physics parameters
            
        Returns:
            List of PhysicsParameter objects
        """
        params = []
        for phys_elem in element.findall("physics"):
            param = self._parse_single_physics_parameter(phys_elem)
            if param:
                params.append(param)
        
        return params

    def _parse_single_physics_parameter(self, element: ET.Element) -> Optional[PhysicsParameter]:
        """Parse a single physics parameter from XML element.
        
        Args:
            element: XML element containing a physics parameter
            
        Returns:
            PhysicsParameter object or None if should be fixed
        """
        name = element.attrib.get(self.ATTR_NAME)
        min_val = self._parse_float(element, "min")
        max_val = self._parse_float(element, "max")
        num_points = self._parse_int(element, "points")
        true_val = self._parse_mixed(element, "true")
        
        # Handle mass ordering specially
        if name == self.ORDERING:
            return self._handle_mass_ordering(element, name, num_points, true_val)
        
        # Check if should be fixed
        if min_val == max_val or num_points <= 1:
            self._log_parameter_fixed(name)
            return None
        
        edges = [min_val, max_val]
        return PhysicsParameter(
            name=name,
            true_value=true_val,
            min_value=min_val,
            max_value=max_val,
            num_points=num_points,
            edges=edges,
        )

    def _handle_mass_ordering(self, element: ET.Element, name: str, 
                             num_points: int, true_val: str) -> Optional[PhysicsParameter]:
        """Handle special case of neutrino mass ordering parameter.
        
        Args:
            element: XML element
            name: Parameter name
            num_points: Number of points in grid
            true_val: True value
            
        Returns:
            PhysicsParameter for ordering or None if fixed
            
        Raises:
            ValidationError: If ordering is improperly specified
        """
        min_str = element.find("min").text or ""
        max_str = element.find("max").text or ""
        
        has_normal = "norm" in min_str or "norm" in max_str
        has_inverted = "inv" in min_str or "inv" in max_str
        
        if has_normal and has_inverted and num_points == 2:
            return PhysicsParameter(
                name=name,
                true_value=true_val,
                min_value=0,
                max_value=1,
                num_points=2,
                edges=list(self.MASS_ORDERINGS),
            )
        elif (has_normal or has_inverted) and num_points == 1:
            # Should be fixed, not returned as physics parameter
            self._log_parameter_fixed(name)
            return None
        elif not (has_normal or has_inverted) or num_points == 0:
            raise ValidationError("Please specify a neutrino mass ordering")
        else:
            raise ValidationError("Ordering parameter is improperly specified")

    def _parse_fixed_parameters(self, element: ET.Element) -> List[FixedParameter]:
        """Parse fixed parameters from XML element.
        
        Args:
            element: XML element containing fixed parameters
            
        Returns:
            List of FixedParameter objects
        """
        params = []
        for fix_elem in element.findall("fixed"):
            if self._is_enabled(fix_elem):
                name = fix_elem.attrib.get(self.ATTR_NAME)
                params.append(FixedParameter(
                    name=name,
                    value=self._parse_mixed(fix_elem, "value"),
                ))
        
        return params

    def _build_physics_grids(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Build interpolation grids for all physics parameters.
        
        Returns:
            Dictionary mapping source -> parameter -> grid array
        """
        grids = {}
        for source, params in self.physics_params.items():
            grids[source] = {}
            for param in params:
                if param.name == self.ORDERING:
                    grids[source][param.name] = np.array(param.edges)
                else:
                    # Use logarithmic spacing for mass-squared differences (names ending in digit > 3)
                    if param.name[-1].isdigit() and int(param.name[-1]) > 3:
                        grid = np.geomspace(param.min_value, param.max_value, param.num_points)
                    else:
                        grid = np.linspace(param.min_value, param.max_value, param.num_points)
                    
                    grids[source][param.name] = grid
                    param.grid = grid
        
        return grids

    def _build_cartesian_grid(self, physics_grids: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Build Cartesian product of all physics parameter grids.
        
        Args:
            physics_grids: Dictionary of parameter grids by source
            
        Returns:
            Array of all grid point combinations
        """
        all_grids = []
        for source in physics_grids:
            for grid_array in physics_grids[source].values():
                all_grids.append(grid_array)
        
        return np.array(list(itertools.product(*all_grids)))

    def set_spherical_grid(self, radius: float = 1.0) -> None:
        """Apply n-dimensional spherical cut to analysis grid.
        
        Removes corners of the parameter grid, keeping only points within
        an n-dimensional ellipsoid. Useful for reducing computation in grid search.
        
        Args:
            radius: Relative radius of ellipsoid in each dimension
        """
        if self.full_physics_grid is None:
            raise ValueError("Physics grid not initialized. Call get_analysis() first.")
        
        self.n_sphere_cut = self._apply_n_sphere(radius)
        self.n_sphere = True
        
        num_original = len(self.full_physics_grid)
        num_filtered = np.sum(self.n_sphere_cut)
        
        print(f"\n{'*' * 65}")
        print(f"**** Analysis grid reduced from {num_original} to {num_filtered} points ****")
        print(f"{'*' * 65}\n")

    def _apply_n_sphere(self, radius: float = 1.0) -> np.ndarray:
        """Compute boolean mask for points within n-sphere.
        
        Args:
            radius: Relative radius of ellipsoid
            
        Returns:
            Boolean array indicating points within sphere
        """
        num_params = self.num_physics_params
        if self.ORDERING in self.physics_params.get(self.scenario or "", []):
            num_params -= 1  # Don't count ordering in distance calculation
        
        centers = np.zeros(num_params)
        radii = np.zeros(num_params)
        
        param_idx = 0
        for source_params in self.physics_params.values():
            for param in source_params:
                if param.name == self.ORDERING:
                    continue
                
                centers[param_idx] = 0.5 * (param.min_value + param.max_value)
                radii[param_idx] = param.max_value - centers[param_idx]
                param_idx += 1
        
        normalized_grid = (self.full_physics_grid - centers) / radii
        return np.sum(normalized_grid**2, axis=1) <= radius**2

    def do_point(self, point_idx: int) -> bool:
        """Check if a point should be analyzed.
        
        Args:
            point_idx: Index of point in analysis grid
            
        Returns:
            True if point should be analyzed
        """
        if not self.n_sphere or self.n_sphere_cut is None:
            return True
        
        return bool(self.n_sphere_cut[point_idx])

    def get_parameter_value(self, param_name: str) -> Union[float, str]:
        """Get value of a parameter (fixed, nominal, or true value).
        
        Args:
            param_name: Name of parameter
            
        Returns:
            Parameter value
            
        Raises:
            ValueError: If parameter not found
        """
        for source_params in self.fixed_params.values():
            for param in source_params:
                if param.name == param_name:
                    return param.value
        
        for source_params in self.nuisance_params.values():
            for param in source_params:
                if param.name == param_name:
                    return param.nominal
        
        for source_params in self.physics_params.values():
            for param in source_params:
                if param.name == param_name:
                    return param.true_value
        
        raise ValueError(f"Parameter '{param_name}' not found in analysis")

    def _validate_all(self) -> None:
        """Perform all consistency checks on analysis configuration.
        
        Raises:
            ValidationError: If any check fails
        """
        self._validate_sources()
        self._validate_physics()
        self._validate_parameters()

    def _validate_sources(self) -> None:
        """Validate that sources match between definitions and experiments."""
        declared_sources = set(self.sources)
        experiment_sources = set()
        
        for exp_config in self.experiments.values():
            experiment_sources.update(exp_config.sources.keys())
        
        if declared_sources != experiment_sources:
            raise ValidationError(
                f"Source mismatch: declared {declared_sources}, "
                f"but experiments use {experiment_sources}"
            )
        
        self._print_section("Source Validation")
        print("✓ All experiment files properly configured:")
        for exp_name, exp_config in self.experiments.items():
            for src_name, src_data in exp_config.sources.items():
                print(f"  - {src_name} in {exp_name}")
                print(f"    MC files: {src_data['MCFiles']}")
                print(f"    Data files: {src_data['DataFiles']}")

    def _validate_physics(self) -> None:
        """Validate physics parameters configuration."""
        if self.num_physics_params == 0:
            raise ValidationError("No physics parameters specified for fitting")
        
        self._print_section("Physics Parameters")
        for source, params in self.physics_params.items():
            param_names = [p.name for p in params]
            print(f"  - {source}: {param_names}")

    def _validate_parameters(self) -> None:
        """Validate nuisance and fixed parameters."""
        self._print_section("Nuisance Parameters")
        for source, params in self.nuisance_params.items():
            param_names = [p.name for p in params]
            print(f"  - {source}: {param_names}")
        
        self._print_section("Fixed Parameters")
        for source, params in self.fixed_params.items():
            param_names = [p.name for p in params]
            print(f"  - {source}: {param_names}")

    def _validate_oscillations(self) -> None:
        """Validate oscillation scenario specification.
        
        Raises:
            ValidationError: If multiple scenarios specified
        """
        if len(self.oscillations) != 1:
            raise ValidationError(
                "Please specify exactly one oscillation scenario. "
                f"Found {len(self.oscillations)}: {self.oscillations}"
            )
        
        self.scenario = self.oscillations[0]
        self._print_section("Oscillation Scenario")
        print(f"  - {self.scenario} (flavors: {self.flavors})")

    # Helper parsing methods
    def _is_enabled(self, element: ET.Element) -> bool:
        """Check if XML element is enabled (status=1)."""
        status_elem = element.find(self.ATTR_STATUS)
        if status_elem is not None:
            return bool(int(status_elem.text or "0"))
        return False

    def _parse_float(self, element: ET.Element, tag: str) -> float:
        """Parse float value from element child."""
        child = element.find(tag)
        if child is not None and child.text:
            return float(child.text)
        raise XMLParseError(f"Missing or invalid float tag '{tag}' in {element.attrib}")

    def _parse_int(self, element: ET.Element, tag: str) -> int:
        """Parse integer value from element child."""
        child = element.find(tag)
        if child is not None and child.text:
            return int(child.text)
        raise XMLParseError(f"Missing or invalid int tag '{tag}' in {element.attrib}")

    def _parse_string(self, element: ET.Element, tag: str) -> str:
        """Parse string value from element child."""
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        raise XMLParseError(f"Missing string tag '{tag}' in {element.attrib}")

    def _parse_mixed(self, element: ET.Element, tag: str) -> Union[float, str]:
        """Parse value that could be numeric or string."""
        child = element.find(tag)
        if child is not None and child.text:
            text = child.text.strip()
            try:
                return float(text)
            except ValueError:
                return text
        raise XMLParseError(f"Missing value tag '{tag}' in {element.attrib}")

    def _parse_file_list(self, element: ET.Element, tag: str) -> List[str]:
        """Parse list of file references from element."""
        files = []
        for file_elem in element.findall(tag):
            if self._is_enabled(file_elem):
                name = file_elem.attrib.get(self.ATTR_NAME)
                if name:
                    files.append(name)
        return files

    @staticmethod
    def _print_section(title: str) -> None:
        """Print formatted section header."""
        print(f"\n{title}:")
        print("-" * 50)

    @staticmethod
    def _print_block_summary(block_name: str, items: List[str]) -> None:
        """Print summary of parsed block."""
        block_labels = {
            ParseXML.NEUTRINO_SOURCE: "Neutrino sources",
            ParseXML.NEUTRINO_TARGET: "Neutrino targets",
            ParseXML.NEUTRINO_EXPERIMENT: "Detectors",
            ParseXML.NEUTRINO_OSCILLATIONS: "Oscillation scenario",
        }
        
        label = block_labels.get(block_name, block_name)
        print(f"\n{label}:")
        for item in items:
            print(f"  + {item}")

    @staticmethod
    def _log_parameter_fixed(param_name: str) -> None:
        """Log when a parameter is moved to fixed."""
        print(f"Notice: Parameter '{param_name}' has been moved to fixed.")