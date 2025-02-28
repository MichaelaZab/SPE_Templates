# SPE_Templates

Pro zpracování waveforem a tvorbu SPE templatů jsou zásadní tři skripty:

1. charge_integral.py
    -vytvoří nábojové histogramy (charge histograms) pro jednotlivé kanály
   
3. Gauss_fit.py
    - nafituje píky ve vytvořených nábojových histogramech pomocí Gausse, pro nás je důležitý 2. pík, který představuje náboj jednoho fotoelektronu
    - vytvoří .png obrázky proložených nábojových histogramů
    - uloží parametry fitu do output_file_2.txt, tento soubor obsahuje čísla kanálů a příslušné hodnoty mu a sigma
      
4. 2D_hist.py
     - závěrečný skript, který normalizuje všechny waveformy a pomocí selekce na základě hodnot fitu z output_file.txt vytvoří SPE templaty
     - dále provádí normalizaci všech waveforem na jedničku
     - dále vytvoří histogra amplitud pro jednotilivé kanály
     - a také vytvoří 2D histogramy waveforem a normovaných waveforem
     - celkem 4 výstupní soubory (.root)

všechny výše zmíněné skripty se spouští jako:
python nazev_skriptu.py vstupni_soubor.root vystupni_soubor (optional: počet waveforem, které má zpracovat)

takže konkrétně například:
python 2D_hist.py run028369_merged.root all_data_ 5000        

