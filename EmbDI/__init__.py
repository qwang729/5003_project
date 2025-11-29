# EmbDI package initialization
__version__ = "1.0"
__author__ = "Riccardo CAPPUZZO (original), Your Name (reproduction)"

# Import core modules for convenience
from .utils import Node, Graph, RandomWalk, load_config, validate_config, build_graph, execute_walks_generation
from .entity_resolution import entity_resolution_pipeline
from .edgelist import generate_edgelist