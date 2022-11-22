import AnalysisReader as AR

test = AR.parse()
'''
test.readSources()
test.readDetectors()
test.readExperiments()
test.readOscillations()
print('Nuisance (marg) param.:')
print(test.Nuisance)
print(test.NuisanceList)
print(test.NuisNominal)
print('Fixed (model) param.:')
print(test.Fixed)
print(test.FixedValue)

print('Physics (fit) param.:')
print(test.Physics)
print(test.PhysGrid)
print(test.PhysEdges)

for i in test.Experiments.keys():
    print(test.Experiments[i].keys())
'''
