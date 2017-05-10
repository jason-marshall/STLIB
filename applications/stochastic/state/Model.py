"""Implements the Model class."""
# CONTINUE: Fixed non-unique names when reactions or species are added.

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from Utilities import getUniqueName
from Reaction import Reaction
from TimeEvent import TimeEvent
from TriggerEvent import TriggerEvent
from Value import Value
from ParameterEvaluation import evaluateModel, evaluateModelInhomogeneous,\
    KineticLawDecorator, KineticLawDecoratorMathematica, getParameters,\
    getIdentifiers
from io.XmlWriter import XmlWriter
from io.MathematicaWriter import MathematicaWriter, mathematicaForm, mathematicaCoefficient

import math

class Model:
    """The model describes the compartments, species, and reactions."""

    # CONTINUE: Maybe I shouldn't store the id here.
    def __init__(self):
        """Make an empty model.

        Member data:
        - id: An optional identifier for the model.
        - name: An optional descriptive name for the model.  Used for 
        information only.
        - compartments: A dictionary of compartments. The keys are the 
        compartment identifiers.
        - species: A dictionary of species. The keys are the species 
        identifiers.
        - speciesIdentifiers: A list of the species identifiers.
        - reactions: A list of reactions.
        - timeEvents: A list of the time events.
        - triggerEvents: A list of the trigger events.
        - parameters: A dictionary of parameters. The keys are the parameter
        identifiers.

        CONTINUE:
        Note that compartments are represented, but not really used. They are
        not needed to identify species; to conform with SBML the species 
        identifiers in a model must be unique. Because we only use substance
        units, we don't need to track the size or spatial dimensions of the
        compartments."""
        self.id = ''
        self.name = ''
        self.compartments = {}
        self.species = {}
        self.speciesIdentifiers = []
        self.reactions = []
        self.timeEvents = []
        self.triggerEvents = []
        self.parameters = {}

    def hasErrors(self, isDiscrete):
        """Return None if the model is valid. Otherwise return an error 
        message. An empty model is not considered to be valid."""
        if not self.id:
            return 'Empty identifier.'
        # Check the compartments.
        # Note that there may be no compartments.
        for id in self.compartments:
            if not id:
                return 'Compartment has an empty identifier.'
            error = self.compartments[id].hasErrors()
            if error:
                return 'Error in compartment ' + id + '.\n' + error
        # Check the number of species.
        if len(self.species) < 1:
            return 'There are no species.'
        if len(self.speciesIdentifiers) != len(self.species):
            return 'Internal error: len(self.speciesIdentifiers) != len(self.species).'
        # Check that there is at least one reaction or event.
        if not self.reactions and not self.timeEvents and\
               not self.triggerEvents:
            return 'There are no reactions or events.'

        # Check the species.
        compartmentIdentifiers = self.compartments.keys()
        for id in self.speciesIdentifiers:
            error = self.species[id].hasErrors(compartmentIdentifiers)
            if error:
                return 'Error in species ' + id + '.\n' + error
        # Check that the species identifiers are unique.
        if len(set(self.speciesIdentifiers)) != len(self.speciesIdentifiers):
            return 'The species identifiers are not unique.'

        # Check the reactions.
        for reaction in self.reactions:
            error = reaction.hasErrors(self.speciesIdentifiers, isDiscrete)
            if error:
                return 'Error in reaction ' + reaction.id + '.\n' + error
        # Check that the reaction identifiers are unique.
        reactionIdentifiers = [x.id for x in self.reactions]
        if len(set(reactionIdentifiers)) != len(reactionIdentifiers):
            return 'The reaction identifiers are not unique.'

        # Check the time events.
        for event in self.timeEvents:
            error = event.hasErrors()
            if error:
                return 'Error in time event ' + event.id + '.\n' + error
        # Check that the identifiers are unique.
        timeEventIdentifiers = [x.id for x in self.timeEvents]
        if len(set(timeEventIdentifiers)) != len(timeEventIdentifiers):
            return 'The time event identifiers are not unique.'

        # Check the trigger events.
        for event in self.triggerEvents:
            error = event.hasErrors()
            if error:
                return 'Error in trigger event ' + event.id + '.\n' + error
        # Check that the identifiers are unique.
        triggerEventIdentifiers = [x.id for x in self.triggerEvents]
        if len(set(triggerEventIdentifiers)) != len(triggerEventIdentifiers):
            return 'The trigger event identifiers are not unique.'

        # Check the parameters.
        for id in self.parameters:
            error = self.parameters[id].hasErrors()
            if error:
                return 'Error in parameter ' + id + '.\n' + error
        # Check that the parameter identifiers are unique.
        parameterIdentifiers = self.parameters.keys()
        if len(set(parameterIdentifiers)) != len(parameterIdentifiers):
            return 'The parameter identifiers are not unique.'

        # Check that all model identifiers are unique.
        identifiers = compartmentIdentifiers + self.speciesIdentifiers +\
            reactionIdentifiers + timeEventIdentifiers +\
            triggerEventIdentifiers + parameterIdentifiers
        if len(set(identifiers)) != len(identifiers):
            # CONTINUE: Report the ones that are not unique.
            return 'The model identifiers are not unique.'
        return None

    # CONTINUE REMOVE
    #def getReactionIdentifiers(self):
    #    return [r.id for r in self.reactions]

    def hasOnlyMassActionKineticLaws(self):
        return not False in \
            [reaction.massAction for reaction in self.reactions]

    def hasIntegerInitialAmounts(self):
        """Return True if all of the initial amounts are integer-valued."""
        for id in self.species:
            value = self.species[id].initialAmountValue
            if value <= 2**53 and int(value) != value:
                return False
        return True

    def evaluate(self):
        """Evaluate the parameters, the species initial amounts, and the
        reaction propensities for the mass action kinetic laws. Return
        None if successful. Otherwise return an error message."""
        return evaluateModel(self)

    def evaluateInhomogeneous(self):
        """Evaluate the parameters and the species initial amounts. Return
        None if successful. Otherwise return an error message."""
        return evaluateModelInhomogeneous(self)

    # CONTINUE: Adapt.
    def setVolume(self, volume, doUpdate):
        """Set the volume. Update the mass action propensities if indicated."""
        # Disable until adapted.
        assert False
        assert volume > 0
        assert doUpdate == True or doUpdate == False
        if doUpdate:
            ratio = self.volume / volume
            for reaction in self.reactions:
                if reaction.massAction:
                    order = reaction.order()
                    if order >= 2:
                        factor = ratio
                        for i in range(order-2):
                            factor *= ratio
                        reaction.propensity += '*' + str(factor)
                        reaction.simplify()
        self.volume = volume

    def addReverseReactions(self):
        """If this model has any reactions that have been marked as reversible,
        then add the reverse reaction. Leave the propensities for these 
        reactions blank."""
        for r in self.reactions:
            if r.reversible:
                r.reversible = False
                id = getUniqueName(r.id + '_reverse', 
                                   [x.id for x in self.reactions])
                if r.name:
                    name = r.name + ' reverse'
                else:
                    name = ''
                # Use the new identifier and name. Switch the reactants and
                # products. Leave the propensity blank.
                self.reactions.append(Reaction(id, name, r.products[:],
                                               r.reactants[:], r.massAction,
                                               ''))

    def makePropensitiesNumberOfReactions(self):
        return 'static const std::size_t NumberOfReactions = %d;\n'\
            % len(self.reactions)

    def makePropensitiesConstructor(self):
        result = 'Propensities(const ReactionSetType& reactionSet) :\n'
        result += '  Base(reactionSet) {\n'
        for i in range(len(self.reactions)):
            result += '  _propensityFunctions[%d] = &Propensities::f%d;\n' % \
                (i, i)
        result += '}\n'
        return result

    def makePropensitiesMemberFunctions(self, isDiscrete):
        # While pi and e are defined in the math module, they are not built-in
        # C++ constants. Thus we need to temporarily add pi and e.
        temporary = []
        for id, value in [('pi', math.pi), ('e', math.e)]:
            if not id in self.parameters and\
                    not id in self.speciesIdentifiers:
                temporary.append(id)
                self.parameters[id] = Value('', value)
        
        prefix = '__p_'
        decorator = KineticLawDecorator(prefix, self.parameters.keys(),
                                        'x', self.speciesIdentifiers)
        result = ''
        for i in range(len(self.reactions)):
            result += 'Number\n'
            result += 'f%d(const PopulationType* x) const {\n' % i

            # Propensity function.
            if self.reactions[i].massAction:
                f = self.reactions[i].makeMassActionPropensityFunction(
                    'x', self.speciesIdentifiers, isDiscrete)
            else:
                result += '  using namespace std;\n'
                expression = self.reactions[i].propensity
                # Parameters.
                for id in getParameters(expression, self.parameters.keys()):
                    result += '  const Number ' + prefix + id + ' = ' +\
                        repr(self.parameters[id].value) + ';\n'
                f = decorator(expression)

            # For discrete methods, we do not check for negative populations
            # in the propensity function. For continuous methods we do.
            if isDiscrete:
                result +=\
                    '  return ' + f + ';\n' +\
                    '}\n\n'
            else:
                # If each of the reactant populations are positive, return the
                # propensity function. Otherwise return 0.
                condition = self.reactions[i].makePositiveReactantsCondition(\
                    'x', self.speciesIdentifiers)
                result +=\
                    '  if(' + condition + ') {\n' +\
                    '    return ' + f + ';\n' +\
                    '  }\n' +\
                    '  else {\n' +\
                    '    return 0;\n' +\
                    '  }\n' +\
                    '}\n\n'
        # Remove pi and e if they were temporarily added.
        for id in temporary:
            del self.parameters[id]
        return result

    def makeInhomogeneousPropensities(self, isDiscrete):
        # While pi and e are defined in the math module, they are not built-in
        # C++ constants. Thus we need to temporarily add pi and e.
        temporary = []
        for id, value in [('pi', math.pi), ('e', math.e)]:
            if not id in self.parameters and\
                    not id in self.speciesIdentifiers:
                temporary.append(id)
                self.parameters[id] = Value('', value)
        
        prefix = '__p_'
        decorator = KineticLawDecorator(prefix, self.parameters.keys(),
                                        'x', self.speciesIdentifiers)
        lines = ['inline',
                 'void',
                 'computePropensities(std::vector<double>* propensities, const std::vector<double>& x, const double t) {',
                 '  using namespace std;']
        # The parameters.
        for id in self.parameters.keys():
            lines.append('  const double ' + prefix + id + ' = ' +\
                             repr(self.parameters[id].value) + ';')
        # The propensities.
        for i in range(len(self.reactions)):
            reaction = self.reactions[i]
            if reaction.massAction:
                expression = reaction.makeInhomogeneousMassActionPropensityFunction(
                    'x', self.speciesIdentifiers, isDiscrete)
            else:
                expression = reaction.propensity
            expression = decorator(expression)
            # For discrete methods, we do not check for negative populations
            # in the propensity function. For continuous methods we do.
            assignment = '  (*propensities)[' + str(i) + '] = '
            if isDiscrete:
                assignment += expression + ';'
            else:
                assignment += 'max(0., ' + expression + ');'
            lines.append(assignment)
        lines.append('}')
        # Remove pi and e if they were temporarily added.
        for id in temporary:
            del self.parameters[id]
        return '\n'.join(lines)

    def convertCustomToMassAction(self):
        """Try to convert the custom rate laws to mass action ones."""
        parameters = self.compartments.keys() + self.parameters.keys()
        for reaction in self.reactions:
            reaction.convertCustomToMassAction(self.speciesIdentifiers,
                                               parameters)
        
    def writeXml(self, writer):
        # Don't check stochastic-specific validity.
        assert not self.hasErrors(False)
        attributes = {}
        attributes['id'] = self.id
        if self.name:
            attributes['name'] = self.name
        writer.beginElement('model', attributes)
        # listOfParameters
        if self.parameters:
            writer.beginElement('listOfParameters')
            for id in self.parameters:
                self.parameters[id].writeParameterXml(writer, id)
            writer.endElement()
        # listOfCompartments
        # Note: Do not write the unnamed compartment.
        if self.compartments:
            writer.beginElement('listOfCompartments')
            for id in self.compartments:
                self.compartments[id].writeCompartmentXml(writer, id)
            writer.endElement()
        # listOfSpecies
        writer.beginElement('listOfSpecies')
        for id in self.speciesIdentifiers:
            self.species[id].writeXml(writer, id)
        writer.endElement()
        # listOfReactions
        if self.reactions:
            writer.beginElement('listOfReactions')
            for reaction in self.reactions:
                reaction.writeXml(writer)
            writer.endElement()
        # listOfTimeEvents
        if self.timeEvents:
            writer.beginElement('listOfTimeEvents')
            for event in self.timeEvents:
                event.writeXml(writer)
            writer.endElement()
        # listOfTriggerEvents
        if self.triggerEvents:
            writer.beginElement('listOfTriggerEvents')
            for event in self.triggerEvents:
                event.writeXml(writer)
            writer.endElement()
        writer.endElement() # model

    def doUseUnnamedCompartment(self):
        """Return True if a species uses the unnamed compartment."""
        for id in self.species:
            if self.species[id].compartment == '':
                return True
        return False

    def writeSbml(self, writer, version):
        # Don't check stochastic-specific validity.
        assert not self.hasErrors(False)
        assert version in range(1, 4)
        # sbml
        writer.beginElement\
            ('sbml', 
             {'xmlns':'http://www.sbml.org/sbml/level2/version%d' % version,
              'level':'2', 'version':str(version)})
        # model
        attributes = {}
        if self.id:
            attributes['id'] = self.id
        if self.name:
            attributes['name'] = self.name
        writer.beginElement('model', attributes)

        # listOfUnitDefinitions
        writer.beginElement('listOfUnitDefinitions')
        writer.beginElement('unitDefinition', {'id':'substance'})
        writer.beginElement('listOfUnits')
        writer.writeEmptyElement('unit', {'kind':'item'})
        writer.endElement()
        writer.endElement()
        writer.endElement()

        # listOfCompartments
        # Note that there must be either a named or unnamed compartment in use.
        writer.beginElement('listOfCompartments')
        # Write the named compartments.
        for id in self.compartments:
            self.compartments[id].writeCompartmentSbml(writer, id)
        unnamedCompartment = ''
        # If any of the species use the unnamed compartment.
        if self.doUseUnnamedCompartment():
            # Get a unique name.
            unnamedCompartment = getUniqueName('Unnamed',
                                               self.compartments.keys())
            # Write the unnamed compartment.
            c = Value('', '1')
            c.value = 1.
            c.writeCompartmentSbml(writer, unnamedCompartment)
        writer.endElement()

        # listOfSpecies
        writer.beginElement('listOfSpecies')
        for id in self.speciesIdentifiers:
            self.species[id].writeSbml(writer, id, unnamedCompartment)
        writer.endElement()

        # listOfParameters
        if self.parameters:
            writer.beginElement('listOfParameters')
            for id in self.parameters:
                self.parameters[id].writeParameterSbml(writer, id)
            writer.endElement()

        # listOfReactions
        writer.beginElement('listOfReactions')
        n = 0
        for reaction in self.reactions:
            reaction.writeSbml(writer)
            n += 1
        writer.endElement()

        # CONTINUE: Add events.

        writer.endElement() # model
        writer.endElement() # sbml

    def writeCmdl(self, outputFile):
        """Write a CMDL file that Dizzy can import."""
        # The parameter values.
        for id in self.parameters:
            outputFile.write(id + '=' + str(self.parameters[id].value) + ';\n')
        # The species and initial amounts.
        for id in self.species:
            outputFile.write(id + '='
                             + str(self.species[id].initialAmountValue) + ';\n')
        for r in self.reactions:
            outputFile.write(r.id + ',' + r.stringCmdl() + ',')
            if r.massAction:
                outputFile.write(r.propensity + ';\n')
            else:
                # Non-mass action propensities are enclosed in brackets.
                outputFile.write('[' + r.propensity + '];\n')
        # CONTINUE: Add events.

    def writeMathematica(self, writer, method, recordedSpecies,
                         recordedReactions):
        """In writing the Mathematica file I remove all of the underscores
        from the identifiers."""
        assert not self.hasErrors(False)
        writer.begin('Notebook')
        writer.begin('Title', self.id)
        writer.begin('Input', r'Needs[\"PlotLegends`\"]')
        writer.end()

        #
        # Compartments.
        #
        if self.compartments:
            writer.begin('Section', 'Compartments')
            # CONTINUE: Instead of using the value, translate the Python
            # expression to Mathematica.
            writer.begin('Input',
                         r'\n'.join(['%s:=%s;' %
                                     (id.replace('_',''),
                                      mathematicaForm(self.compartments[id].value))
                                     for id in self.compartments]))
            writer.end()
            writer.end() # Compartments

        #
        # Parameters.
        #
        if self.parameters:
            writer.begin('Section', 'Parameters')
            # CONTINUE: Instead of using the value, translate the Python
            # expression to Mathematica.
            writer.begin('Input',
                         r'\n'.join(['%s:=%s;' %
                                     (id.replace('_',''),
                                      mathematicaForm(self.parameters[id].value))
                                     for id in self.parameters]))
            writer.end()
            writer.end() # Parameters

        #
        # Species.
        #
        writer.begin('Section', 'Species')
        # species
        content = r'species={'
        content += r','.join([r'%s[t]' % id.replace('_','') for id in 
                                self.speciesIdentifiers])
        content += r'};\n'
        # speciesIdentifiers
        content += r'speciesIdentifiers=Table[Head[species[[i]]],{i,Length[species]}];\n'
        # speciesEquations
        content += r'speciesEquations={'
        equations = []
        for id in self.speciesIdentifiers:
            lhs = r"%s'[t]==" % id.replace('_','')
            rhs = r''
            for r in self.reactions:
                influence = r.influence(id)
                if influence != 0:
                    rhs += r"%s %s'[t]" % \
                        (mathematicaCoefficient(influence),
                         r.id.replace('_',''))
            if not rhs:
                rhs = r'0'
            equations.append(lhs+rhs)
        content += r','.join(equations)
        content += r'};'
        # speciesInitialConditions
        content += r'speciesInitialConditions={'
        content += r','.join([r'%s[0]==%s' %
                                (id.replace('_',''),
                                 mathematicaForm(self.species[id].initialAmountValue))
                                for id in self.species])
        content += r'};'
       
        writer.begin('Input', content)
        writer.end()
        writer.end() # Species

        #
        # Reactions.
        #
        decorator = KineticLawDecoratorMathematica(self.speciesIdentifiers)
        writer.begin('Section', 'Reactions')
        # reactions
        content = r'reactions={'
        content += r','.join([r'%s[t]' % r.id.replace('_','') for r in 
                              self.reactions])
        content += r'};\n'
        # reactionIdentifiers
        content += r'reactionIdentifiers=Table[Head[reactions[[i]]],{i,Length[reactions]}];\n'
        # reactionEquations
        content += r'reactionEquations={'
        equations = []
        for r in self.reactions:
            eqn = r"%s'[t]==" % r.id.replace('_','')
            if r.massAction:
                equations.append(r.id.replace('_','') + "'[t]==" +
                                 r.makeMassActionPropensityFunctionMathematica\
                                     (self.speciesIdentifiers).replace('_',''))
            else:
                # Use ToExpression[] to convert most mathematical expressions
                # to standard Mathematica form. For example "sin(x)" will be
                # converted to "Sin[x]". Note that "pow(x,y)" is not 
                # correctly interpreted.
                equations.append(r.id.replace('_','') +\
                                     r"'[t]==ToExpression[\"" +\
                                     decorator(r.propensity).replace('_','') +\
                                     r'\",TraditionalForm]')
        content += r','.join(equations)
        content += r'};\n'
        # reactionInitialConditions
        content += r'reactionInitialConditions=Table[reactions[[i]][[0]][0]==0,{i,Length[reactions]}];'
       
        writer.begin('Input', content)
        writer.end()
        writer.end() # Reactions

        #
        # Time interval.
        #
        writer.begin('Section', 'Time Interval')
        writer.begin('Input', r'startTime=%s;\nequilibrationTime=%s;\nrecordingTime=%s;\nnumberOfFrames=%s;' %
                     (mathematicaForm(method.startTime),
                      mathematicaForm(method.equilibrationTime),
                      mathematicaForm(method.recordingTime),
                      mathematicaForm(method.numberOfFrames)))
        writer.end()
        writer.end() # Time Interval

        #
        # Numerically Solve
        #
        writer.begin('Section', 'Numerically Solve')
        writer.begin('Input', r'initialTime=startTime+equilibrationTime;\nfinalTime=initialTime+recordingTime;\nframeTimes=Table[initialTime+(i-1)recordingTime/numberOfFrames,{i,numberOfFrames}];\nsolution=NDSolve[Join[speciesEquations,reactionEquations,speciesInitialConditions,reactionInitialConditions],Join[species,reactions],{t,startTime,finalTime}][[1]];')
        writer.end()
        writer.end() # Numerically Solve

        #
        # Plot the Species Populations
        #
        if recordedSpecies:
            writer.begin('Section', 'Plot the Species Populations')
            # Note: Mathematica indices start at 1.
            writer.begin('Input', r'recordedSpecies={' +
                         ','.join([str(_i+1) for _i in recordedSpecies]) + r'};')
            writer.end()
            writer.begin('Text', r'Plot all of the species with tooltips.')
            writer.end()
            writer.begin('Input', r'Plot[Evaluate[Table[Tooltip[species[[recordedSpecies[[i]]]]/.solution,speciesIdentifiers[[recordedSpecies[[i]]]]],{i,Length[recordedSpecies]}]],{t,initialTime,finalTime},PlotRange->All]')
            writer.end()
            writer.begin('Text', r'Plot all of the species with a legend.')
            writer.end()
            writer.begin('Input', r'Plot[Evaluate[species[[recordedSpecies]]/.solution],{t,initialTime,finalTime},PlotRange->All,PlotLegend->speciesIdentifiers[[recordedSpecies]],LegendPosition->{1.1,-0.5}]')
            writer.end()
            writer.begin('Text', r'Plot each of the species.')
            writer.end()
            writer.begin('Input', r'GraphicsGrid[Table[{Plot[Evaluate[species[[recordedSpecies[[i]]]]/.solution],{t,initialTime,finalTime},PlotRange->All,PlotLabel->speciesIdentifiers[[recordedSpecies[[i]]]]]},{i,Length[recordedSpecies]}],ImageSize->400]')
            writer.end()
            writer.end() # Plot the Species Populations

        #
        # Plot the Reaction Counts
        #
        if recordedReactions:
            writer.begin('Section', 'Plot the Reaction Counts')
            # Note: Mathematica indices start at 1.
            writer.begin('Input', r'recordedReactions={' +
                         ','.join([str(_i+1) for _i in recordedReactions]) + r'};')
            writer.end()
            writer.begin('Text', r'Plot all of the reactions with tooltips.')
            writer.end()
            writer.begin('Input', r'Plot[Evaluate[Table[Tooltip[reactions[[recordedReactions[[i]]]]/.solution,reactionIdentifiers[[recordedReactions[[i]]]]],{i,Length[recordedReactions]}]],{t,initialTime,finalTime},PlotRange->All]')
            writer.end()
            writer.begin('Text', r'Plot all of the reactions with a legend.')
            writer.end()
            writer.begin('Input', r'Plot[Evaluate[reactions[[recordedReactions]]/.solution],{t,initialTime,finalTime},PlotRange->All,PlotLegend->reactionIdentifiers[[recordedReactions]],LegendPosition->{1.1,-0.5}]')
            writer.end()
            writer.begin('Text', r'Plot each of the reactions.')
            writer.end()
            writer.begin('Input', r'GraphicsGrid[Table[{Plot[Evaluate[reactions[[recordedReactions[[i]]]]/.solution],{t,initialTime,finalTime},PlotRange->All,PlotLabel->reactionIdentifiers[[recordedReactions[[i]]]]]},{i,Length[recordedReactions]}],ImageSize->400]')
            writer.end()
            writer.end() # Plot the Reaction Counts

        #
        # Write a Trajectory file that Cain can Import
        #
        writer.begin('Section', 'Write a Trajectory file that Cain can Import')
        writer.begin('Text', r'The current directory. You can change the directory with SetDirectory[].')
        writer.end()
        writer.begin('Input', r'Directory[]')
        writer.end()
        writer.begin('Text', r'Write the trajectory data to %s.txt.' % self.id)
        writer.end()
        # CONTINUE: put in dictionary.
        #r'(*The number of species.*)',
        #r'Write[file,Length[species]];',
        #r'(*The number of reactions.*)',
        #r'Write[file,Length[reactions]];',
        #r'(*The number of frames.*)',
        #r'Write[file,numberOfFrames];',
        #r'(*The list of frame times.*)',
        #r'If[numberOfFrames==1,frameTimes={finalTime},frameTimes=Table[initialTime+(i-1)recordingTime/(numberOfFrames-1),{i,numberOfFrames}]];',
        #r'For[i=1,i<=numberOfFrames,++i,WriteString[file,CForm[frameTimes[[i]]],\" \"]];',
        #r'WriteString[file,\"\\n\"];',
        inputs = [r'file=OpenWrite[\"%s.txt\"];' % self.id,
                  r'(*Blank line for the Python dictionary of information.*)',
                  r'WriteString[file,\"\\n\"];',
                  r'(*The number of trajectories.*)',
                  r'Write[file,1];',
                  r'(*Blank line for the initial Mersenne twister state.*)',
                  r'WriteString[file,\"\\n\"];',
                  r'(*The solver was successful.*)',
                  r'WriteString[file,\"\\n\"];',
                  r'(*The species populations at each frame.*)',
                  r'For[i=1,i<=numberOfFrames,++i,For[j=1,j<=Length[recordedSpecies],++j,WriteString[file,CForm[species[[recordedSpecies[[j]]]]/.solution/.t->frameTimes[[i]]],\" \"]]];',
                  r'WriteString[file,\"\\n\"];',
                  r'(*The reaction counts at each frame.*)',
                  r'For[i=1,i<=numberOfFrames,++i,For[j=1,j<=Length[recordedReactions],++j,WriteString[file,CForm[reactions[[recordedReactions[[j]]]]/.solution/.t->frameTimes[[i]]],\" \"]]];',
                  r'WriteString[file,\"\\n\"];',
                  r'(*Blank line for the final Mersenne twister state.*)',
                  r'WriteString[file,\"\\n\"];',
                  r'Close[file];']
        writer.begin('Input', r'\n'.join(inputs))
        writer.end()
        writer.end() # Write a Trajectory file that Cain can Import

        writer.end() # Title
        writer.end() # Notebook

    def makeAsciiSpeciesReferenceList(self, species):
        x = ''
        first = True
        for speciesReference in species:
            if first:
                first = False
            else:
                x = x + ' + '
            if speciesReference.stoichiometry != 1:
                x = x + '%d ' % speciesReference.stoichiometry
            x = x + speciesReference.species
        return x

    # CONTINUE REMOVE
    def makeAsciiReaction(self, reaction, arrow='->'):
        return self.makeAsciiSpeciesReferenceList(reaction.reactants) +\
            ' ' + arrow + ' ' +\
            self.makeAsciiSpeciesReferenceList(reaction.products)

    def writeSpeciesTable(self):
        table = []
        for id in self.speciesIdentifiers:
            species = self.species[id]
            table.append([id, str(species.initialAmount), species.name,
                          species.compartment])
        return table

    def writeReactionsTable(self):
        table = []
        for reaction in self.reactions:
            if reaction.massAction:
                massAction = '1'
            else:
                massAction = ''
            table.append([reaction.id,
                          self.makeAsciiSpeciesReferenceList(reaction.reactants),
                          self.makeAsciiSpeciesReferenceList(reaction.products),
                          massAction, reaction.propensity, reaction.name])
        return table

    # CONTINUE: Convert the rest of the write*Table function to this style.
    def writeTimeEventsTable(self):
        return [[event.id, event.times, event.assignments, event.name] for
                event in self.timeEvents]

    def writeTriggerEventsTable(self):
        return [[event.id, event.trigger, event.assignments,
                 event.delay and str(event.delay) or '',
                 event.useValuesFromTriggerTime and '1' or
                 not event.useValuesFromTriggerTime and '', event.name] for
                event in self.triggerEvents]

    def writeParametersTable(self):
        table = []
        for id in self.parameters:
            p = self.parameters[id]
            table.append([id, p.expression, p.name])
        return table

    def writeCompartmentsTable(self):
        table = []
        for id in self.compartments:
            c = self.compartments[id]
            table.append([id, c.expression, c.name])
        return table

def duplicateModel(model, multiplicity, useScaling):
    """Duplicate the model by the specified multiplicity."""
    assert int(multiplicity) == multiplicity and multiplicity >= 2

    import re, copy
    if useScaling:
        from random import random

    duplicated = Model()
    duplicated.id = model.id
    duplicated.name = model.name
    # CONTINUE: This is not correct for events because the species id's change.
    duplicated.timeEvents = copy.deepcopy(model.timeEvents)
    duplicated.triggerEvents = copy.deepcopy(model.triggerEvents)
    duplicated.compartments = copy.deepcopy(model.compartments)
    duplicated.parameters = copy.deepcopy(model.parameters)
    for i in range(multiplicity):
        suffix = '_' + str(i + 1)
        if useScaling:
            factor = str(random()) + '*'
        # The species.
        for id in model.speciesIdentifiers:
            duplicated.species[id + suffix] = copy.deepcopy(model.species[id])
            duplicated.speciesIdentifiers.append(id + suffix)
        for r in model.reactions:
            # Copy the reaction.
            reaction = copy.deepcopy(r)
            # Add a suffix to the species and reaction identifiers.
            reaction.id += suffix
            for reactant in reaction.reactants:
                reactant.species += suffix
            for product in reaction.products:
                product.species += suffix
            if not reaction.massAction:
                for id in model.speciesIdentifiers:
                    reaction.propensity = re.sub(id, id + suffix,
                                                 reaction.propensity)
            if useScaling:
                reaction.propensity = factor + reaction.propensity
            # Append the reaction to the duplicated model.
            duplicated.reactions.append(reaction)
    # Return the duplicated model.
    return duplicated

    
def writeModelXml(model, out=None):
    if out:
        writer = XmlWriter(out)
    else:
        writer = XmlWriter()
    writer.beginDocument()
    model.writeXml(writer)
    writer.endDocument()

def writeModelSbml(model, out=None):
    if out:
        writer = XmlWriter(out)
    else:
        writer = XmlWriter()
    writer.beginDocument()
    model.writeSbml(writer, 3)
    writer.endDocument()

def writeModelMathematica(model, method, out=None):
    if out:
        writer = MathematicaWriter(out)
    else:
        writer = MathematicaWriter()
    recordedSpecies = range(len(model.species))
    recordedReactions = range(len(model.reactions))
    model.writeMathematica(writer, method, recordedSpecies, recordedReactions)

def main():
    from Species import Species
    from Method import Method
    from SpeciesReference import SpeciesReference

    print('-'*79)
    print('Time Homogeneous')
    model = Model()
    model.id = 'model'
    model.compartments['C1'] = Value('Cell', '1')
    model.speciesIdentifiers.append('s1')
    model.species['s1'] = Species('C1', 'species 1', '13')
    model.speciesIdentifiers.append('s2')
    model.species['s2'] = Species('C1', 'species 2', '17')
    model.reactions.append(
        Reaction('r1', 'reaction 1', [SpeciesReference('s1')], 
                 [SpeciesReference('s2')], True, '1.5'))
    model.reactions.append(
        Reaction('r2', 'reaction 2', 
                 [SpeciesReference('s1'), SpeciesReference('s2')], 
                 [SpeciesReference('s1', 2)], True, '2.5'))
    writeModelXml(model)
    print('')
    print(model.writeSpeciesTable())
    print('')
    print(model.writeReactionsTable())
    # The model must be evaluated for writing SBML or the C++ code.
    model.evaluate()
    writeModelSbml(model, open('model.xml', 'w'))
    method = Method()
    writeModelMathematica(model, method, open('model.nb', 'w'))
    print('')
    print(model.makePropensitiesNumberOfReactions())
    print(model.makePropensitiesConstructor())
    print(model.makePropensitiesMemberFunctions(True))
    print('')
    model.writeCmdl(sys.stdout)

    print('-'*79)
    print('Time Inhomogeneous')
    model = Model()
    model.id = 'model'
    model.speciesIdentifiers.append('s1')
    model.species['s1'] = Species('', 'species 1', '13')
    model.speciesIdentifiers.append('s2')
    model.species['s2'] = Species('', 'species 2', '17')
    model.reactions.append(
        Reaction('r1', 'reaction 1', [SpeciesReference('s1')], 
                 [SpeciesReference('s2')], True, '2+sin(t)'))
    model.reactions.append(
        Reaction('r2', 'reaction 2', 
                 [SpeciesReference('s1'), SpeciesReference('s2')], 
                 [SpeciesReference('s1', 2)], False, '1+exp(-t)'))
    writeModelXml(model)
    print('')
    print(model.writeSpeciesTable())
    print('')
    print(model.writeReactionsTable())
    # The model must be evaluated for writing SBML or the C++ code.
    model.evaluateInhomogeneous()
    writeModelSbml(model, open('model.xml', 'w'))
    print('')
    print(model.makeInhomogeneousPropensities(True))
    print('')

if __name__ == '__main__':
    main()
