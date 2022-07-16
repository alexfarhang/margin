# margin
This repository accompanies the paper [Investigating Generalization by Controlling Normalized Margin](https://arxiv.org/abs/2205.03940) appearing at ICML 2022.

### Run Experiments and Plot Figures:
* Figure 1, 2: [generalized_spect_norm_margin_experiment.py](/generalized_spect_norm_margin_experiment.py)
* Figure 3, 7: [extreme_memorization_augmented_data_control.py](extreme_memorization_augmented_data_control.py)
* Figure 5, 10: [extreme_memorization_scale_or_margins.py](extreme_memorization_scale_or_margins.py)

Plot above figures with [generate_all_figures.ipynb](generate_all_figures.ipynb)

* Figure 4, 8, 9: refer to directions in [sampled_v_trained_anon](sampled_v_trained_anon)
* Figure 6: Experiments and figures in [average.ipynb](average.ipynb)

### Citation:
```bibtex
@InProceedings{farhang2022margin,
  title = 	 {Investigating Generalization by Controlling Normalized Margin},
  author =       {Farhang, Alexander R and Bernstein, Jeremy D and Tirumala, Kushal and Liu, Yang and Yue, Yisong},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {6324--6336},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/farhang22a/farhang22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/farhang22a.html},
}
```

