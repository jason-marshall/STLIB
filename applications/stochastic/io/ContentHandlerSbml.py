"""SBML content handler."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Model import Model
from state.Value import Value
from state.ParameterEvaluation import KineticLawDecoratorSbml
from state.Species import Species
from state.Reaction import Reaction
from state.SpeciesReference import SpeciesReference

import xml.sax.handler

class ContentHandlerSbml(xml.sax.ContentHandler):
    """An SBML content handler.

    Inherit from the XML SAX content handler."""

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        # The stack of the enclosing elements.
        self.elements = []
        # The stack of unhandled enclosing elements.
        self.unhandled = []
        self.errors = ''
        self.warnings = ''
        self.level = 0
        self.operators = []
        self.firstTerm = False
        self.content = ''
        # Define the handled elements as a set for efficient checking.
        self.handled = set(['sbml',
                            'model',
                            'listOfParameters', 'listOfCompartments',
                            'listOfSpecies', 'listOfReactions',
                            'parameter', 'compartment', 'species', 'reaction',
                            'listOfReactants', 'listOfProducts',
                            'listOfModifiers', 'speciesReference',
                            'modifierSpeciesReference', 'kineticLaw',
                            'math', 'apply', 'times', 'divide', 'plus', 'minus',
                            'ci', 'cn'])

    def startElement(self, name, attributes):
        if name == 'cain':
            # Fatal error.
            raise Exception('This is a Cain file. Directly open the file instead of using File->Import SBML.')

        if name in self.handled:
            self.elements.append(name)
        else:
            self.unhandled.append(name)
            self.warnings += 'Unhandled tag: ' + name + '\n'

        # If we are processing an unhandled tag, do nothing.
        if self.unhandled:
            return

        #
        # Kinetic law elements.
        #
        if name != 'kineticLaw' and 'kineticLaw' in self.elements:
            if name == 'listOfParameters':
                return
            elif name == 'parameter':
                # Append a parameter.
                if 'id' in attributes.keys():
                    id = attributes['id']
                else:
                    if 'name' in attributes.keys():
                        id = attributes['name']
                        self.errors += 'Missing id attribute in reaction parameter. Using name for id.\n'
                    else:
                        self.errors += 'Missing id attribute in reaction parameter.\n'
                        return
                # Ignore the name attribute.
                if 'value' in attributes.keys():
                    expression = attributes['value']
                else:
                    expression = ''
                self.kineticLawParameters[id] = Value('', expression)
            elif name == 'math':
                if self.elements[-2] != 'kineticLaw':
                    self.errors == 'Badly placed math tag.\n'
                    return
            elif name == 'apply':
                self.level += 1
                if self.level != 1:
                    if not self.firstTerm:
                        if not self.operators:
                            self.errors += 'No operator for apply block.\n'
                        else:
                            self.propensity += self.operators[-1]
                    self.propensity += '('
            elif name == 'times':
                self.operators.append('*')
                self.firstTerm = True
            elif name == 'divide':
                self.operators.append('/')
                self.firstTerm = True
            elif name == 'plus':
                self.operators.append('+')
                self.firstTerm = True
            elif name == 'minus':
                self.operators.append('-')
                self.firstTerm = True
            # Content Identifier
            elif name == 'ci':
                # The content should be empty.
                if self.content:
                    self.errors += 'Mishandled content in ci tag.\n'
            # Content Number
            elif name == 'cn':
                # The content should be empty.
                if self.content:
                    self.errors += 'Mishandled content in cn tag.\n'
            return
        
        #
        # The top level element.
        #
        if name == 'sbml':
            if 'level' in attributes.keys() and attributes['level'] == 1:
                self.errors += 'Error: SBML level 1 is not supported. Use level 2 or higher.\n'
            return
        #
        # Elements for the model.
        #
        elif name == 'model':
            # Start a new model.
            self.model = Model()
            # If an ID was specified, use it.
            if 'id' in attributes.keys():
                self.model.id = attributes['id']
            # Otherwise try the name.
            elif 'name' in attributes.keys():
                self.model.id = attributes['name']
            # If there is no ID or name, just use 'model'.
            else:
                self.model.id = 'model'
            if 'name' in attributes.keys():
                self.model.name = attributes['name']
        elif name == 'listOfParameters':
            return
        elif name == 'listOfCompartments':
            # Start the dictionary of compartments.
            self.model.compartments = {}
        elif name == 'listOfSpecies':
            # Start the dictionary of species and list of species identifiers.
            self.model.species = {}
            self.model.speciesIdentifiers = []
        elif name == 'listOfReactions':
            # Start the list of reactions.
            self.model.reactions = []
        elif name == 'parameter':
            # Append a parameter.
            if 'id' in attributes.keys():
                id = attributes['id']
            else:
                if 'name' in attributes.keys():
                    id = attributes['name']
                    self.errors += 'Missing id attribute in parameter. Using name for id.\n'
                else:
                    self.errors += 'Missing id attribute in parameter.\n'
                    return
            if 'name' in attributes.keys():
                parameterName = attributes['name']
            else:
                parameterName = ''
            if 'value' in attributes.keys():
                expression = attributes['value']
            else:
                expression = ''
            self.model.parameters[id] = Value(parameterName, expression)
        elif name == 'compartment':
            # Append a compartment.
            if not 'id' in attributes.keys():
                self.errors += 'Missing id attribute in compartment.\n'
                return
            if not attributes['id']:
                self.errors += 'Compartment identifier is empty.\n'
                return
            # No default name. Default size of 1.
            compartment = Value('', '1')
            if 'name' in attributes.keys():
                compartment.name = attributes['name']
            if 'size' in attributes.keys():
                compartment.expression = attributes['size']
            else:
                # Cain uses compartment sizes like parameter values. Thus
                # a value for the size is required.
                compartment.expression = '1'
            # Ignore spatialDimensions, constant, and outside.
            self.model.compartments[attributes['id']] = compartment
        elif name == 'species':
            # Append a species.
            if 'id' in attributes.keys():
                id = attributes['id']
            else:
                if 'name' in attributes.keys():
                    id = attributes['name']
                    self.errors += 'Missing id attribute in species. Using name for id.\n'
                else:
                    self.errors += 'Missing id attribute in species.\n'
                    return
            if not 'compartment' in attributes.keys():
                self.errors += 'Missing compartment attribute in species' +\
                    id + '.\n'
                return
            if 'name' in attributes.keys():
                speciesName = attributes['name']
            else:
                speciesName = ''
            if 'initialAmount' in attributes.keys():
                initialAmount = attributes['initialAmount']
            else:
                initialAmount = ''
            self.model.species[id] = \
                Species(attributes['compartment'], speciesName, initialAmount)
            self.model.speciesIdentifiers.append(id)
        elif name == 'reaction':
            # Append a reaction.
            if 'id' in attributes.keys():
                id = attributes['id']
            else:
                if 'name' in attributes.keys():
                    id = attributes['name']
                    self.errors += 'Missing id attribute in reaction. Using name for id.\n'
                else:
                    self.errors += 'Missing id attribute in reaction.\n'
                    return
            if 'name' in attributes.keys():
                reactionName = attributes['name']
            else:
                reactionName = ''
            # Construct the reaction.
            r = Reaction(id, reactionName, [], [], False, '0')
            # Note if the reaction is reversible. The reversible field is true
            # by default.
            if not 'reversible' in attributes.keys() or\
                    attributes['reversible'] == 'true':
                r.reversible = True
            self.model.reactions.append(r)
        elif name == 'listOfReactants':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfReactants tag.\n'
                return
        elif name == 'listOfProducts':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfProducts tag.\n'
                return
        elif name == 'listOfModifiers':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfModifiers tag.\n'
                return
        elif name == 'speciesReference':
            # Add the reactant, product, or modifier to the current reaction.
            if not self.model.reactions:
                self.errors += 'Badly placed speciesReference tag.\n'
                return
            if not 'species' in attributes.keys():
                self.errors +=\
                    'Missing species attribute in speciesReference.\n'
                return
            if 'stoichiometry' in attributes.keys():
                stoichiometry = int(attributes['stoichiometry'])
            else:
                stoichiometry = 1
            # No need to record if the stoichiometry is zero.
            if stoichiometry != 0:
                sr = SpeciesReference(attributes['species'], stoichiometry)
                if self.elements[-2] == 'listOfReactants':
                    self.model.reactions[-1].reactants.append(sr)
                elif self.elements[-2] == 'listOfProducts':
                    self.model.reactions[-1].products.append(sr)
                else:
                    self.errors += 'Badly placed speciesReference tag.\n'
                    return
        elif name == 'modifierSpeciesReference':
            # Add to the reactants and products of the current reaction.
            if not self.model.reactions:
                self.errors += 'Badly placed modifierSpeciesReference tag.\n'
                return
            if not 'species' in attributes.keys():
                self.errors +=\
                    'Missing species attribute in modifierSpeciesReference.\n'
                return
            if self.elements[-2] != 'listOfModifiers':
                self.errors += 'Badly placed modifierSpeciesReference tag.\n'
                return
            sr = SpeciesReference(attributes['species'])
            self.model.reactions[-1].reactants.append(sr)
            self.model.reactions[-1].products.append(sr)
        elif name == 'kineticLaw':
            if self.elements[-2] != 'reaction':
                self.errors == 'Badly placed kineticLaw tag.\n'
                return
            # Start a propensity expression.
            self.propensity = ''
            # Start the dictionary of parameters for the kinetic law.
            self.kineticLawParameters = {}
            
    def endElement(self, name):
        # If we are processing an unhandled tag, do nothing.
        if self.unhandled and name == self.unhandled[-1]:
            del self.unhandled[-1]
            return

        del self.elements[-1]

        #
        # Kinetic law elements.
        #
        if 'kineticLaw' in self.elements:
            if name == 'apply':
                del self.operators[-1]
                if self.level != 1:
                    self.propensity += ')'
                self.level -= 1
                self.firstTerm = False
            # Content Identifier or Content Number
            elif name == 'ci' or name == 'cn':
                if self.operators and not self.firstTerm:
                    self.propensity += self.operators[-1]
                self.propensity += self.content
                self.firstTerm = False
                self.content = ''
            return

        if name == 'listOfParameters':
            # The list of parameters may be empty.
            return
        elif name == 'listOfCompartments':
            if not self.model.compartments:
                self.errors += 'No compartments were defined.\n'
        elif name == 'listOfSpecies':
            if not (self.model.species and self.model.speciesIdentifiers):
                self.errors += 'No species were defined.\n'
        elif name == 'listOfReactions':
            if not self.model.reactions:
                self.errors += 'No reactions were defined.\n'
        elif name == 'kineticLaw':
            if not self.model.reactions:
                self.errors += 'Kinetic law without a reaction.\n'
                return
            # Remove unnecessary spaces.
            self.propensity = self.propensity.replace(' ', '')
            # Insert the parameters values.
            decorator = KineticLawDecoratorSbml(self.kineticLawParameters)
            self.model.reactions[-1].propensity = decorator(self.propensity)
                
    def characters(self, content):
        # If we are processing an unhandled tag, do nothing.
        if self.unhandled:
            return
        # Content Identifier or Content Number
        if self.elements[-1] == 'ci' or self.elements[-1] == 'cn':
            self.content += content

def main():
    from xml.sax import parse
    from io.XmlWriter import XmlWriter
    from glob import glob
    
    for name in glob('../examples/sbml/dsmts31/*.xml'):
        handler = ContentHandlerSbml()
        parse(open(name, 'r'), handler)
        if handler.warnings:
            print('\nWarning for ' + name)
            print(handler.warnings)
        if handler.errors:
            print('\nError for ' + name)
            print(handler.errors)
        assert not handler.errors

    handler = ContentHandlerSbml()
    parse(open('../examples/sbml/dsmts31/dsmts-001-01.xml', 'r'),
          handler)
    assert not handler.errors
    writer = XmlWriter()
    writer.beginDocument()
    handler.model.writeXml(writer)
    writer.endDocument()

if __name__ == '__main__':
    main()
