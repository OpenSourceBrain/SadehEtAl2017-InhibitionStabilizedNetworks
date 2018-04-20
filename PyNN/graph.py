import neuromllite
import sys

import logging

file_name = sys.argv[1]

   
logging.basicConfig(level=logging.INFO, format="%(name)-19s %(levelname)-5s - %(message)s")
    
from neuroml.hdf5.NeuroMLXMLParser import NeuroMLXMLParser


'''
from neuroml.hdf5.DefaultNetworkHandler import DefaultNetworkHandler
nmlHandler = DefaultNetworkHandler()   

#currParser = NeuroMLHdf5Parser(nmlHandler) 

currParser.parse(file_name)'''


from neuromllite.GraphVizHandler import GraphVizHandler

level = int(sys.argv[2])

handler = GraphVizHandler(level, None)

currParser = NeuroMLXMLParser(handler)

currParser.parse(file_name)

handler.finalise_document()

print("Done with GraphViz...")
