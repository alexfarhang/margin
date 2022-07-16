### Setup:
cd sampled_v_trained_anon/
pip install -r requirements.txt

### To exactly generate figure 4:
(1) Open linear_run.sh in a text editor
(2) Set flags mnist_first_class=0 and mnist_second_class=1
(3) Change the list of SEEDS on line 5 to just be 160 i.e put for SEED in 160
(4) Change result_dir flag to the directory you want to save results in
(6) Run jupyter notebook, and open final_runs.ipynb
(7) Inside the definition of plot_graphs_just_one_for_paper() function, change the location of the filename to be the one produced from running linear_run.sh
(8) Run plot_graphs_just_one_for_paper() in the notebook 

### To exactly generate figure 8
(Generating the data)
Top row:
(1) Open linear_run.sh in a text editor
(2) Set flags mnist_first_class=0 and mnist_second_class=1
(3) Change the list of SEEDS on line 5 to just be 160, 161, 162 i.e put for SEED in 160 161 162
(4) Run ./linear_run.sh in your cmd line

Middle row:
Same as the top row, except: mnist_first_class=4, mnist_second_class=7, and the list of SEEDS should be 165, 166, 167

Bottom row:
Same as top row, except: mnist_first_class=3, mnist_second_class=8, and the list of SEEDS should be 170, 171, 172

(Generating the plots)
Run the first 6 cells of final_runs.ipynb, making sure to include the correct directories that include the data from running the instructions above.

### To exactly generate figure 9
(1) Open conv_run.sh in a text editor
(2) Change SEED=175 and run ./conv_run.sh
(3) Change SEED=176 and run ./conv_run.sh
(4) Change SEED=177 and run ./conv_run.sh
(5) Run the 1st, 2nd, 3rd, and 7th cell in final_runs.ipynb
