op2=>operation: import sys
op4=>operation: import os.path
op6=>operation: import numpy as np
op8=>operation: from itertools import product
op10=>operation: import argparse
op12=>operation: import Experiments
op14=>operation: import AnalysisReader
op16=>operation: from Analysis import Sensitivity
op18=>operation: input_arg = argparse.ArgumentParser()
sub20=>subroutine: input_arg.add_argument('xml_file', type=str, nargs='?', default='xmlAnalysis/AnalysisTemplate.xml', help='Input analysis file in xml format.')
sub22=>subroutine: input_arg.add_argument('-p', '--point', nargs='?', type=int, default=0, help="Specify analysis point to run. Only if 'cluster' option is enabled.")
sub24=>subroutine: input_arg.add_argument('-lp', '--list_of_points', nargs='?', type=list, default=0, help='Specify set of analysis points to run.')
sub26=>subroutine: input_arg.add_argument('-o', '--outfile', nargs='?', type=str, default='out.dat', help='Analysis output file.')
sub28=>subroutine: input_arg.add_argument('--multi', dest='multiproc', default=False, action='store_true', help='Option for running the analysis with multiprocessing (recommended locally).')
sub30=>subroutine: input_arg.add_argument('--cluster', dest='cluster', default=False, action='store_true', help='Option for submitting jobs to a cluster.')
sub32=>subroutine: input_arg.add_argument('--mcmc', dest='mcmc', default=False, action='store_true', help='Option for sampling parameter space using Markov Chain Monte Carlo.')
op34=>operation: args = input_arg.parse_args()
op36=>operation: multiproc = args.multiproc
op38=>operation: cluster = args.cluster
op40=>operation: markov = args.mcmc
cond43=>condition: if cluster
cond48=>condition: if (args.point is None)
op52=>operation: point = 0
op62=>operation: analysis_xml_file = args.xml_file
op64=>operation: outfile = args.outfile
op66=>operation: an = parse(analysis_xml_file)
sub68=>subroutine: an.readSources()
sub70=>subroutine: an.readExperiments()
sub72=>subroutine: an.readDetectors()
sub74=>subroutine: an.readPhysics()
sub76=>subroutine: an.readOscPar()
sub78=>subroutine: an.CheckSystematics()
op80=>operation: mcList = {}
cond83=>condition: for s in an.sources
cond113=>condition: for (i, (exp, fil, t)) in enumerate(zip(an.experiments, an.mcFiles, an.Exposure))
op126=>operation: mcList[exp] = rd(s, exp, t, fil)
sub128=>subroutine: mcList[exp].Binning()
sub130=>subroutine: mcList[exp].InitialFlux()
sub132=>subroutine: mcList[exp].BFOscillator(an.neutrinos, **an.OscParametersBest)
cond139=>operation: with open(outfile, 'w') as f:
    for par in an.parameters:
        f.write((par + ' '))
    if (an.NoSyst == 0):
        for sys in an.Systematics.values():
            for s in sys:
                f.write((s + ' '))
    f.write('X2 ')
    f.write('\n') if  ((cluster and ((point == 0) or (not os.path.isfile(outfile)))) or (not cluster))
sub149=>subroutine: print('=============================================================\n==================== Starting analysis ======================\n=============================================================')
cond152=>condition: if (an.physics[0] == 'Three Flavour')
op156=>operation: param = []
cond159=>operation: param.append(an.OscParametersGrid[oscpar]) while  oscpar in [*an.OscParametersGrid]
op171=>operation: parametersGrid = product(*param)
cond174=>condition: if cluster
op178=>operation: element = list(parametersGrid)[point]
sub180=>subroutine: print(f'Processing {element}')
sub182=>subroutine: Sensitivity(element[:(- 1)], element[(- 1)], an, mcList, outfile)
cond187=>condition: if multiproc
op191=>operation: import multiprocessing
op193=>operation: cores = multiprocessing.cpu_count()
cond196=>condition: if an.NoSyst
op200=>operation: cores = (3 * cores)
sub202=>subroutine: print('Analyzing with no systematics')
cond208=>condition: if markov
op212=>operation: ' Testing '
op214=>operation: import emcee
op216=>operation: nwalkers = (2 ** 4)
op218=>operation: ndim = (len(an.OscParametersEdges) - 1)
op220=>operation: nsteps = 200
op222=>operation: initial = np.zeros((nwalkers, ndim))
cond225=>condition: for (i, par) in enumerate(an.OscParametersEdges.values())
op248=>operation: param = list(an.OscParametersEdges.keys())[i]
cond251=>condition: if (param == 'Ordering')
op255=>operation: pass
op259=>operation: mu = an.OscParametersBest[param]
op261=>operation: sigma = min(abs((par[0] - mu)), abs((par[1] - mu)))
op263=>operation: initial[(:, i)] = np.random.uniform(par[0], par[1], nwalkers)
op268=>operation: mo = an.OscParametersBest['Ordering']
op270=>operation: with multiprocessing.Pool(processes=cores) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, Sensitivity, args=[mo, an, mcList, outfile], pool=pool)
    sampler.run_mcmc(initial, nsteps, progress=True, skip_initial_state_check=True)
op274=>operation: with multiprocessing.Pool(processes=cores) as pool:
    for element in parametersGrid:
        res = pool.apply_async(Sensitivity, args=(element[:(- 1)], element[(- 1)], an, mcList, outfile))
    pool.close()
    pool.join()
cond280=>operation: Sensitivity(element[:(- 1)], element[(- 1)], an, mcList, outfile) while  element in parametersGrid
op56=>operation: point = args.point

op2->op4
op4->op6
op6->op8
op8->op10
op10->op12
op12->op14
op14->op16
op16->op18
op18->sub20
sub20->sub22
sub22->sub24
sub24->sub26
sub26->sub28
sub28->sub30
sub30->sub32
sub32->op34
op34->op36
op36->op38
op38->op40
op40->cond43
cond43(yes)->cond48
cond48(yes)->op52
op52->op62
op62->op64
op64->op66
op66->sub68
sub68->sub70
sub70->sub72
sub72->sub74
sub74->sub76
sub76->sub78
sub78->op80
op80->cond83
cond83(yes)->cond113
cond113(yes)->op126
op126->sub128
sub128->sub130
sub130->sub132
sub132(left)->cond113
cond113(no)->cond83
cond83(no)->cond139
cond139->sub149
sub149->cond152
cond152(yes)->op156
op156->cond159
cond159->op171
op171->cond174
cond174(yes)->op178
op178->sub180
sub180->sub182
cond174(no)->cond187
cond187(yes)->op191
op191->op193
op193->cond196
cond196(yes)->op200
op200->sub202
sub202->cond208
cond208(yes)->op212
op212->op214
op214->op216
op216->op218
op218->op220
op220->op222
op222->cond225
cond225(yes)->op248
op248->cond251
cond251(yes)->op255
op255->cond225
cond251(no)->op259
op259->op261
op261->op263
op263->cond225
cond225(no)->op268
op268->op270
cond208(no)->op274
cond196(no)->cond208
cond187(no)->cond280
cond48(no)->op56
op56->op62
cond43(no)->op62

