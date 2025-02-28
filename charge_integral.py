import uproot as ur
import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('agg')

import pickle as pkl
import gzip

import sys, os

from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks

qrange = (0,1e4) # range for histogramming charge integral
outpref = ''

def run() :
    '''
    Main function, called if this module is run directly: python charge_integral.py input_fname.root out/prefix
    '''

    global outpref, qrange # use these from the global scope
    fname   = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    outdir=os.path.dirname(outpref)
    print('Output prefix:', outpref)
    if len(outdir) :
        os.makedirs(outdir, exist_ok=True)

    # these will accumulate data for each channel: charge histogram and number of entries in each histogram
    hists_by_chan  = dict()
    counts_by_chan = dict()

    # prepare output
    #outf = OpenGZ(outpref+'selected_zeroed_wfms.pkl.gz')


    N_WFMS=20000
    # Maximum number of waveforms to read in total
    NMAX=-1
    if len(sys.argv) > 3 :
        NMAX=int(sys.argv[3])

    counter       = 0
    total_wfms    = 0
    selected_wfms = 0
    # get data
    # tree = ur.open(fname+':raw_waveforms')
    # print(f'Input tree has {tree.num_entries} entries.')

    # iterate over batches in the input file; t is the input tree, only reading branches adcs and channel
    for t in ur.iterate(fname+':raw_waveforms',['adcs','channel'], step_size=N_WFMS) :
        if NMAX > -1 and total_wfms >= NMAX :
            break
        print(f'Starting processing batch of data: {counter}...')

        # sort by channel
        chan_sort = ak.argsort(t.channel)
        # channel runs: when sorted by channel, calculates how many waveforms of the same channel there are
        chan_runs = ak.run_lengths(t.channel[chan_sort])
        # stores only the first number of the sorted channels
        channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
        # group waveforms by channel
        wfms_all  = ak.unflatten(t.adcs[chan_sort], chan_runs)
        # make sure the waveforms are floats
        wfms_all  = ak.values_astype(wfms_all, np.float64)

        # do our stuff on the sorted waveforms
        ProcessBatch( hists_by_chan, counts_by_chan, channels, wfms_all)

        # usefull counting
        counter += 1
        total_wfms += ak.count(t.channel)

    selected_wfms = ak.sum(counts_by_chan.values())

    print(f'Number of input waveforms:    {total_wfms}')
    print(f'Number of selected waveforms: {selected_wfms}')
    print(f'Selection efficiency:         {selected_wfms/total_wfms:.3f}')


    # save the histograms in an output channel
    # create the output ROOT file; will be closed at the end of the 'with' block
    with ur.recreate(outpref+'charge_hists.root') as outf :
        for ch,counts in hists_by_chan.items() : # loop over each cahnnel and get the channel id and histogram counts
            bins = np.histogram_bin_edges(None, bins=len(counts), range=qrange)
            # store the counts and bins as a TH1D; this is taken care of by UpROOT
            outf[f'charge_hist_{ch}'] = (counts.to_numpy(), bins)

    #PlotHists(hists_by_chan, counts_by_chan)

def ProcessBatch(hists_by_chan, counts_by_chan, channels, wfms) :
    '''
    Function to process single batch of waveforms read from the input file.
    '''
    global qrange # get from the global scope, for histogram range

    # selection
    # clean pretrigger
    PRETRIGGER=125
    # ak.nanstd calculates standard deviation a.k.a. RMS in for each
    # waveform (over last index: axis=-1; range is limited to
    # 0..PRETRIGGER: ":PRETRIGGER"). The RMS is then requested to be
    # lower than 5.
    rms_mask = ak.nanstd(wfms[...,:PRETRIGGER],axis=-1) < 5
    wfms = wfms[rms_mask] # use the mask to filter out the waveforms
    # remove pedestal
    means = ak.nanmean(wfms[...,:PRETRIGGER],axis=-1)
    wfms = wfms - means
    channels = channels[ak.any(rms_mask, axis=-1)]

    # Get ROI
    start=PRETRIGGER
    stop =180
    wfms_for_charge = wfms[...,start:stop]

    # integrate
    charges = -ak.sum(wfms_for_charge[wfms_for_charge<0], axis=-1) # '-' used to correct for signal's negative polarity

    # Add new charge data to the histograms
    for chan,qs in zip(channels, charges) : # iterates over each channel and get the array of calculated charges
        if len(qs) == 0 : # make sure there were any waveforms
            continue
        hist,_ = np.histogram(qs.to_numpy().flatten(), bins=500, range=qrange) # create new histogram
        if chan not in hists_by_chan.keys() : # this is the first time we created a histogram for this channel
            # initialize the dictionaries
            hists_by_chan[chan]    = ak.zeros_like(hist)
            counts_by_chan[chan]   = 0

        # update total histogram for this channel
        hists_by_chan[chan]    = hists_by_chan[chan] + hist
        counts_by_chan[chan]   = counts_by_chan[chan] + len(qs)


    # store dict into the output pickle file
    #pkl.dump({'channels':channels, 'wfms':wfms, 'charges':charges}, outf)



def PlotHists(hists_by_chan, counts_by_chan):
    '''
    Help function which creates plots of the histograms.
    Takes:
      hists_by_channel ... is a dictionary of {channel: numpy.histogram}
      counts_by_channel .. dictionary of {channel: number of entries in the histogram}
    '''
    global qrange
    global outpref

    fig, ax = plt.subplots()

    print('Plotting charge histograms...')
    for chan,qh in hists_by_chan.items() :
        ax.cla()
        plt.title(f'Channel {chan} made of {counts_by_chan[chan]} waveforms')
        plt.xlabel(r'Integrated charge [ADC$\times$tick]')
        plt.ylabel('Count')
        plt.stairs(qh, np.histogram_bin_edges(None, len(qh), qrange))
        fig.savefig(outpref+f'charge_{chan}.png')
    print('Done.')

def OpenGZ(fname, compresslevel=3) :
    '''
    Help function to open gzip file to write to.
    '''
    return gzip.open(fname, 'wb', compresslevel=compresslevel)


# if this is the main running script, do run()
if __name__ == '__main__' :
    run()
