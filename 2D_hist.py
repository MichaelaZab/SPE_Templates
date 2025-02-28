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
binwidth = 20.

output_for_txt = []

#####################################################################################################################

data = np.loadtxt("output_file_2.txt", delimiter="\t", skiprows=1)

# Extract columns
channels_data = data[:, 0]  # First column (channel)
fit_mu = data[:, 1]    # Second column (fit_mu)
fit_sigma = data[:, 2]  # Third column (fit_sigma)

# Function to find values based on channel
def get_fit_values(channel_value):
    idx = np.where(channels_data == channel_value)[0]  # Find index
    if len(idx) > 0:
        return fit_mu[idx][0], fit_sigma[idx][0]  # Return corresponding values
    else:
        return None, None  # Return None if channel not found


def Create2DHists(wfms, channels, bins=None, range=None) :
    '''
    Expects:
      wfms - 3D array wfms[chanIt][wfmIt][wfmDigitIt] of ADCs.
      channels - list of channel IDs: channels[chanIt]
    '''

    hists_by_chan_2D = dict()
    bins_by_chan  = dict()
    for chan,chwfms in zip(channels, wfms) :
        nwfms = len(chwfms)
        if nwfms == 0  :
            continue
        size = len(chwfms[0])
        # determine number of bins and the range of the histogram
        if bins is None :
            # by default, create bin for each TDC tick and 600 bins for the ADC range
            bins  = (size,600)
        if range == None :
            # by default, use full TDC range and zoomed ADC range: ((TDCLow,TDCHi), (ADCLow,ADCHi))
            range = ((0,size), (-500,100))
        # build array of TDC ticks
        x = np.repeat( [np.arange(size,dtype=np.float64)], len(chwfms), axis=0 )
        # creates histogram
        hist, xedges, yedges = np.histogram2d( x.flatten(), chwfms.to_numpy().flatten(), bins=bins, range=range)
        hists_by_chan_2D[chan] = hist
        bins_by_chan[chan]  = (xedges, yedges)

    return hists_by_chan_2D, bins_by_chan


##########################################################################################################

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

    total_sum_wfms = dict()
    total_count_wfms = dict()

    hists_by_chan_2D = dict()
    bins_by_chan_2D = dict()

    norm_hists_by_chan_2D = dict()
    norm_bins_by_chan_2D = dict()

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
        ProcessBatch( hists_by_chan, counts_by_chan,
                      channels, wfms_all,
                      total_sum_wfms, total_count_wfms,
                      hists_by_chan_2D, bins_by_chan_2D,
                      norm_hists_by_chan_2D, norm_bins_by_chan_2D)

        # usefull counting
        counter += 1
        total_wfms += ak.count(t.channel)

    selected_wfms = ak.sum(counts_by_chan.values())

    print(f'Number of input waveforms:    {total_wfms}')
    print(f'Number of selected waveforms: {selected_wfms}')
    print(f'Selection efficiency:         {selected_wfms/total_wfms:.3f}')

    template_by_channel = dict()

    for chan, count in total_count_wfms.items():
        wf = total_sum_wfms[chan]
        template_by_channel[chan] = wf/count


    # save the histograms of the amplitudes in an output channel
    # create the output ROOT file; will be closed at the end of the 'with' block
    with ur.recreate(outpref+'MPE_amplitude_hists.root') as outf :
        for ch,counts in hists_by_chan.items() : # loop over each channel and get the channel id and histogram counts
            bins = np.histogram_bin_edges(None, bins=len(counts), range=(0,150))
            #store the counts and bins as a TH1D; this is taken care of by UpROOT
            outf[f'MPE_amplitude_hist_{ch}'] = (counts.to_numpy(), bins)


    # save the TEMPLATES in an output channel
    # create the output ROOT file; will be closed at the end of the 'with' block
    with ur.recreate(outpref+'SPE_templates.root') as outf :
        for chan,template in template_by_channel.items() : # loop over each channel and get the channel id and template
            if len(template)==0:
                continue
            #bins = np.histogram_bin_edges(None, bins=len(template.to_numpy()), range=(0,1024))
            bins = np.arange(1025)
            # store the counts and bins as a TH1D; this is taken care of by UpROOT
            outf[f'SPE_templates_{chan}'] = (template.to_numpy(), bins)

    # save the 2D histograms of the waveforms in an output channel
    with ur.recreate(outpref + '2D_histogramy.root') as outf:
        for ch, hist in hists_by_chan_2D.items():
            # Získáme bin edges z bins_by_chan
            bins_x, bins_y = bins_by_chan_2D[ch]
            # Uložíme histogram do ROOT souboru správně
            outf[f'2D_histogramy_{ch}'] = hist, bins_x, bins_y

    # save the 2D histograms of the norm_waveforms in an output channel
    with ur.recreate(outpref + 'norm_2D_histogramy.root') as outf:
        for ch, hist in norm_hists_by_chan_2D.items():
            # Získáme bin edges z bins_by_chan
            bins_x, bins_y = norm_bins_by_chan_2D[ch]

            # Uložíme histogram do ROOT souboru správně
            outf[f'norm_2D_histogramy_{ch}'] = hist, bins_x, bins_y


def ProcessBatch(hists_by_chan, counts_by_chan,
                 channels, wfms,
                 total_sum_wfms, total_count_wfms,
                 hists_by_chan_2D, bins_by_chan_2D,
                 norm_hists_by_chan_2D, norm_bins_by_chan_2D) :
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

    channel_charges_mask = []

    for chan,qs in zip(channels, charges):
        mu, sigma = get_fit_values(chan)
        charges_mask = 0
        if mu==None:
            charges_mask=ak.zeros_like(qs)!=0
        else:
            charge_window_l, charge_window_r = (mu-1.5*sigma)*binwidth, (mu+1.5*sigma)*binwidth
            #print("Nabojove okno je:", charge_window_l, charge_window_r)
            charges_mask = (qs >= charge_window_l) & (qs <= charge_window_r)
        channel_charges_mask.append(charges_mask)

    channel_charges_mask=ak.Array(channel_charges_mask)

    #na vše se aplikuje maska
    charges = charges[channel_charges_mask]
    wfms = wfms[channel_charges_mask]
    wfms_for_charge = wfms_for_charge[channel_charges_mask]

    #run the Create2DHists function to locally find hists and bins
    temp_hists_by_chan_2D, temp_bins_by_chan_2D = Create2DHists(wfms,channels, bins=None, range=None)

    #save the results of Create2DHists into global variable
    for chan, hist in temp_hists_by_chan_2D.items():
        if chan not in hists_by_chan_2D.keys() : # this is the first time we created a histogram for this channel
             # initialize the dictionaries
             hists_by_chan_2D[chan]    = np.zeros_like(hist)
             bins_by_chan_2D[chan]     = temp_bins_by_chan_2D[chan]
        # update total histogram for this channel
        hists_by_chan_2D[chan]    = hists_by_chan_2D[chan] + hist


    amplitudes = -ak.min(wfms_for_charge[wfms_for_charge<0], axis=-1)

    #výpočet normalizovaných wfms, normalizace na jedničku
    norm_wfms = wfms/amplitudes
    sum_wfms = ak.sum(norm_wfms, axis=-2)
    count_wfms = ak.num(norm_wfms, axis =-2)

    #run the Create2DHists function to locally find hists and bins
    temp_hists_by_chan_2D, temp_bins_by_chan_2D = Create2DHists(norm_wfms,channels, bins=(1024,600), range=((0,1024),(-3,3)))

    #save the results of Create2DHists into global variable
    for chan, hist in temp_hists_by_chan_2D.items():
        if chan not in norm_hists_by_chan_2D.keys() : # this is the first time we created a histogram for this channel
             # initialize the dictionaries
             norm_hists_by_chan_2D[chan]    = np.zeros_like(hist)
             norm_bins_by_chan_2D[chan]     = temp_bins_by_chan_2D[chan]
        # update total histogram for this channel
        norm_hists_by_chan_2D[chan]    = norm_hists_by_chan_2D[chan] + hist

    # Add new amplitude data to the histograms
    for chan,amp in zip(channels, amplitudes) : # iterates over each channel and get the array of calculated amplitudes
        if len(amp) == 0 : # make sure there were any waveforms
             continue
        hist,_ = np.histogram(amp.to_numpy().flatten(), bins=100, range=(0,50)) # create new histogram;
                                                        #range stanovit tak, aby odpovídal očekávané hodnotě aplitudy SPE
        if chan not in hists_by_chan.keys() : # this is the first time we created a histogram for this channel
             # initialize the dictionaries
             hists_by_chan[chan]    = ak.zeros_like(hist)
             counts_by_chan[chan]   = 0
        # update total histogram for this channel
        hists_by_chan[chan]    = hists_by_chan[chan] + hist
        counts_by_chan[chan]   = counts_by_chan[chan] + len(amp)


    for chan,sum,count in zip(channels, sum_wfms, count_wfms):
        if chan not in total_sum_wfms.keys() : # this is the first time we created a histogram for this channel
             # initialize the dictionaries
             total_sum_wfms[chan]    = np.zeros_like(sum)
             total_count_wfms[chan]  = 0
        total_sum_wfms[chan] = total_sum_wfms[chan] + sum
        total_count_wfms[chan] = total_count_wfms[chan] + count



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
