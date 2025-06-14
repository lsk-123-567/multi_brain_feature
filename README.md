multi_brain_fature

1.Feature Extraction of Inter-Brain Synchronization

2.The experimental data consist of resting-state and task-state conditions. Each group includes data from 5 participants, and each participant completed 5 rounds of experiments. The raw experimental data are saved in CSV format.

3.The saved CSV files are processed by preprocess2.py to extract data for each participant in each round, and the results are saved in MAT format. Subsequently, preprocess3.py performs filtering and other preprocessing steps on the raw experimental data, and the processed data are also saved as MAT files.

4.ISC2.py extracts the temporal-domain synchrony between each pair of participants within the same group across all electrodes. After that, ISC3.py computes the average ISC values across all participant pairs based on the results from ISC2.py.


5.arousal2.py calculates the power of three frequency bands and computes the ratio between the alpha and beta bands. This ratio is then used as a measure of emotional arousal.
wpli_1.py is designed to compute the weighted phase lag index (wPLI) between pairs of subjects across multiple frequency bands (theta, alpha, beta) from EEG data stored in .mat files. 

6.The three extracted features are visualized using ISC_plot_top.py, arousal_plot_top.py, and wpli_plot_top.py, which generate EEG topographic maps for the low-polarization and high-polarization groups. The visualization results are presented in result_top.png.7. In addition, overall average bar charts are plotted using ISC_plot_bar.py, arousal_plot_bar.py, and wpli_plot_bar.py, which display the group-level mean values to reflect the level of synchrony within each group. The results are shown in ISC_plot_bar.png, arousal_plot_bar.png, and wpli_plot_bar.png, respectively.





