from __future__ import print_function
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
import sys, os

from scipy.optimize import curve_fit

#channel   = sys.argv[1]

def Gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

output = []

x_values = np.linspace(0, 80, 80)
f = ur.open('run_allcharge_hists.root')
for key,hist in f.items():

    hist = hist.to_numpy()

    h=hist[0][:80]
    peaks, _ = find_peaks(h, height=5, threshold=None, distance=6, width=1.5)
    plt.cla() #vyčistí canvas, clear axes
    plt.plot(h)

    if len(peaks) == 0: #pokud nemáme píky, přeskočíme soubor
        continue


    plt.plot(peaks, h[peaks], "x")
    if peaks[0] > 10: #pokud je první pík moc daleko, nebereme v úvahu
        continue

    for Npeak in range(len(peaks)):  #for loop přes všechny nalezené píky
        if Npeak < 3: # zajímá nás ale jen prvních 5 píků, lze změnit na 2, 3, cokoliv

            # Počáteční odhad parametrů pro fit
            A_guess = h[peaks[Npeak]]         # amplituda daného píku
            mu_guess = peaks[Npeak]  # pozice daného píku
            sigma_guess = 2.5+0.3*Npeak  # sigma daného píku + mechanismus, který rozšiřuje každý další pík
            #
            #
            p0 = [A_guess, mu_guess, sigma_guess]
            print("\nInitial guesses:",p0) # shrnutí počátečního odhadu


            # Define the range you want to fit within
            x_min, x_max = mu_guess-sigma_guess, mu_guess+sigma_guess  # rozsah hodnot, ve kterém fitujeme

            # Select only data within the range
            mask = (x_values >= x_min) & (x_values <= x_max)
            x_fit = x_values[mask]
            h_fit = h[mask]

            # Fit the curve
            fit_A = -1
            fit_mu = -1
            fit_sigma = -1
            try:
                (parameters, covariance) = curve_fit(Gauss, x_fit, h_fit, p0=p0, maxfev=5000)
                # Print results
                print("\nFitted parameters:", parameters)
                fit_A = parameters[0]
                fit_mu = parameters[1]
                fit_sigma = parameters[2]

                fit_y = Gauss(x_fit, fit_A, fit_mu, fit_sigma) # spočítám si y-hodnoty fitu v daném x rozsahu
                plt.plot(x_fit, fit_y, '-', label='fit')
            except:
                print("Fit failed" ,key ,Npeak)
            if Npeak == 1:
                left_boundary, right_boundary = fit_mu-(1.5*fit_sigma), fit_mu+1.5*(fit_sigma)
                channel = int(key[-7:-2])
                row = [channel, fit_mu, fit_sigma]
                print(fit_mu, fit_sigma)
                output.append(row) #výstupní seznam obsahující kanál, polohu píku a sigma
                print("\nLeft boundary = ",left_boundary, ", right boundary = ", right_boundary)
                plt.fill_between(x_values, -10, 1.1*h.max(), where=(x_values >= left_boundary) & (x_values <= right_boundary),
                     color='red', alpha=0.3, label="Highlighted Band")

    # plt.plot(xdata, ydata, 'o', label='data')
    plt.legend()

    key=key[:-2]
    plt.savefig(f"plots/{key}.png")


output = np.array(output)
fmt = ["%d", "%.4f", "%.4f"]  # Integer, 4 decimal places, 4 decimal places
np.savetxt("output_file_2.txt", output, fmt=fmt, delimiter="\t", header="channel\tfit_mu\tfit_sigma")
